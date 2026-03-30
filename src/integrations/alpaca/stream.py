"""
Alpaca real-time market data stream manager.

Runs StockDataStream (equity) and CryptoDataStream (crypto) in dedicated
daemon threads, each with a supervisor/reconnect loop.
Public API is fully sync and thread-safe; the on_trade callback is called
from a stream thread so callers must handle thread-safety themselves.

No Discord imports. No asyncio dependency in the public API.
"""

import logging
import os
import threading
import time
from collections.abc import Callable

from config import ASSETS, PARAMS

logger = logging.getLogger(__name__)

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.live import CryptoDataStream, StockDataStream
    _STREAM_AVAILABLE = True
except ImportError:
    _STREAM_AVAILABLE = False


def is_stream_enabled() -> bool:
    """True if streaming is enabled and Alpaca keys are configured."""
    if os.environ.get("ALPACA_STREAM_ENABLED", "1") == "0":
        return False
    if not PARAMS.stream_enabled:
        return False
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


class AlpacaStreamManager:
    """Manages Alpaca WebSocket streams (equity + crypto) in dedicated daemon threads.

    Usage::

        def on_trade(ticker: str, price: float) -> None:
            ...  # called from stream thread — be thread-safe

        mgr = AlpacaStreamManager(on_trade=on_trade)
        mgr.start(symbols={"AAPL", "MSFT", "BTC"})
        # ... later:
        mgr.stop()
    """

    def __init__(self, on_trade: Callable[[str, float], None]) -> None:
        self._on_trade = on_trade
        self._throttle = float(
            os.environ.get(
                "STREAM_EVAL_THROTTLE_SECONDS",
                str(PARAMS.stream_eval_throttle_seconds),
            )
        )
        self._feed_name = os.environ.get("ALPACA_DATA_FEED", PARAMS.stream_data_feed)

        # --- Equity stream state ---
        self._symbols: set[str] = set()
        self._stream: StockDataStream | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # --- Crypto stream state ---
        self._crypto_symbols: set[str] = set()
        self._crypto_stream: CryptoDataStream | None = None
        self._crypto_thread: threading.Thread | None = None

        self._lock = threading.Lock()

        # Per-ticker last-fired timestamps for throttle (shared across streams)
        self._last_fired: dict[str, float] = {}

        # Equity connected: set True once first trade event arrives from StockDataStream
        self._connected = threading.Event()
        # Crypto connected: set True once first trade event arrives from CryptoDataStream
        self._crypto_connected = threading.Event()

        # Track last trade event time per ticker for stale detection
        self._last_trade_time: dict[str, float] = {}

        # Tickers already warned as stale; suppresses repeat WARNINGs until a fresh trade arrives
        self._known_stale: set[str] = set()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() or self._crypto_connected.is_set()

    def start(self, symbols: set[str]) -> None:
        """Launch equity and/or crypto streams in daemon threads. Thread-safe."""
        with self._lock:
            equity = {s for s in symbols if not self._is_crypto(s)}
            crypto = {s for s in symbols if self._is_crypto(s)}

            if not (self._thread and self._thread.is_alive()):
                self._symbols = equity
                self._stop_event.clear()
                self._connected.clear()
                self._thread = threading.Thread(
                    target=self._supervisor_loop, daemon=True, name="alpaca-stream",
                )
                self._thread.start()
                logger.info(
                    "Equity stream started for %d symbols (feed=%s, throttle=%.0fs)",
                    len(self._symbols), self._feed_name, self._throttle,
                )

            if not (self._crypto_thread and self._crypto_thread.is_alive()):
                self._crypto_symbols = crypto
                self._crypto_connected.clear()
                self._crypto_thread = threading.Thread(
                    target=self._crypto_supervisor_loop, daemon=True, name="alpaca-crypto-stream",
                )
                self._crypto_thread.start()
                logger.info(
                    "Crypto stream started for %d symbols (throttle=%.0fs)",
                    len(self._crypto_symbols), self._throttle,
                )

    def stop(self) -> None:
        """Signal both streams to stop. Thread-safe."""
        self._stop_event.set()
        self._connected.clear()
        self._crypto_connected.clear()

        with self._lock:
            equity_stream = self._stream
            crypto_stream = self._crypto_stream

        for stream in (equity_stream, crypto_stream):
            if stream:
                try:
                    stream.stop()
                except Exception as e:
                    logger.debug("Stream stop error (expected): %s", e)

        logger.info("Streams stopped")

    def update_subscriptions(self, symbols: set[str]) -> None:
        """Diff the symbol set and subscribe/unsubscribe as needed. Thread-safe."""
        equity_symbols = {s for s in symbols if not self._is_crypto(s)}
        crypto_symbols = {s for s in symbols if self._is_crypto(s)}

        with self._lock:
            current_equity = set(self._symbols)
            current_crypto = set(self._crypto_symbols)
            equity_stream = self._stream
            crypto_stream = self._crypto_stream

        # --- Equity diff ---
        eq_add = equity_symbols - current_equity
        eq_remove = current_equity - equity_symbols
        if eq_add or eq_remove:
            logger.info(
                "Equity stream subscriptions: +%d -%d (total %d)",
                len(eq_add), len(eq_remove), len(equity_symbols),
            )
            with self._lock:
                self._symbols = equity_symbols
            if equity_stream and self._connected.is_set():
                if eq_add:
                    try:
                        equity_stream.subscribe_trades(self._handle_trade, *eq_add)
                        logger.info("Subscribed equity: %s", sorted(eq_add))
                    except Exception as e:
                        logger.warning("Failed to subscribe equity %s: %s", eq_add, e)
                if eq_remove:
                    try:
                        equity_stream.unsubscribe_trades(*eq_remove)
                        logger.info("Unsubscribed equity: %s", sorted(eq_remove))
                    except Exception as e:
                        logger.warning("Failed to unsubscribe equity %s: %s", eq_remove, e)

        # --- Crypto diff ---
        cr_add = crypto_symbols - current_crypto
        cr_remove = current_crypto - crypto_symbols
        if cr_add or cr_remove:
            logger.info(
                "Crypto stream subscriptions: +%d -%d (total %d)",
                len(cr_add), len(cr_remove), len(crypto_symbols),
            )
            with self._lock:
                self._crypto_symbols = crypto_symbols
            if crypto_stream and self._crypto_connected.is_set():
                if cr_add:
                    try:
                        alpaca_syms = [self._to_alpaca_crypto_symbol(s) for s in cr_add]
                        crypto_stream.subscribe_trades(self._handle_crypto_trade, *alpaca_syms)
                        logger.info("Subscribed crypto: %s", sorted(cr_add))
                    except Exception as e:
                        logger.warning("Failed to subscribe crypto %s: %s", cr_add, e)
                if cr_remove:
                    try:
                        alpaca_syms = [self._to_alpaca_crypto_symbol(s) for s in cr_remove]
                        crypto_stream.unsubscribe_trades(*alpaca_syms)
                        logger.info("Unsubscribed crypto: %s", sorted(cr_remove))
                    except Exception as e:
                        logger.warning("Failed to unsubscribe crypto %s: %s", cr_remove, e)

    def get_stale_tickers(self, stale_seconds: float = 300.0) -> tuple[list[str], list[str]]:
        """Return (new_stale, known_stale) for tickers with no trade event in stale_seconds.

        new_stale  — first occurrence since last fresh trade; callers should log WARNING.
        known_stale — already reported and still stale; callers should log DEBUG only.

        Side effect: new_stale tickers are added to the internal _known_stale set so
        subsequent calls demote them to known_stale until a fresh trade clears them.
        """
        now = time.monotonic()
        with self._lock:
            symbols = set(self._symbols) | set(self._crypto_symbols)
        new_stale: list[str] = []
        known_stale: list[str] = []
        for ticker in symbols:
            last = self._last_trade_time.get(ticker)
            if last is None or (now - last) > stale_seconds:
                if ticker in self._known_stale:
                    known_stale.append(ticker)
                else:
                    new_stale.append(ticker)
                    self._known_stale.add(ticker)
        return new_stale, known_stale

    def inject_trade(self, ticker: str, price: float) -> None:
        """Fire _on_trade directly, bypassing WebSocket and throttle.

        Used by simulation to inject prices through the real stream code path.
        """
        self._on_trade(ticker, price)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _is_equity(ticker: str) -> bool:
        """Exclude crypto tickers (StockDataStream is equity-only)."""
        return not ticker.endswith("-USD")

    def _is_crypto(self, ticker: str) -> bool:
        """True if ticker is a known crypto asset."""
        cfg = ASSETS.get(ticker)
        return bool(cfg and cfg.asset_class == "crypto")

    @staticmethod
    def _to_alpaca_crypto_symbol(ticker: str) -> str:
        """Convert internal crypto key to Alpaca format: BTC → BTC/USD."""
        return ASSETS[ticker].ticker.replace("-", "/")

    def _get_feed(self) -> "DataFeed":
        if self._feed_name.lower() == "sip":
            return DataFeed.SIP
        return DataFeed.IEX

    async def _handle_trade(self, trade) -> None:
        """Called from StockDataStream's internal event loop per trade event."""
        ticker = str(trade.symbol)
        price = float(trade.price)
        now = time.monotonic()

        # Per-ticker throttle
        last = self._last_fired.get(ticker, 0.0)
        if now - last < self._throttle:
            return

        self._last_fired[ticker] = now
        self._last_trade_time[ticker] = now
        self._known_stale.discard(ticker)
        self._connected.set()

        try:
            self._on_trade(ticker, price)
        except Exception:
            logger.exception("on_trade callback raised for %s", ticker)

    async def _handle_crypto_trade(self, trade) -> None:
        """Called from CryptoDataStream's internal event loop per trade event.

        Normalizes Alpaca format (BTC/USD) to internal key (BTC).
        """
        # trade.symbol is "BTC/USD" — extract base to match internal key
        ticker = str(trade.symbol).split("/")[0]
        price = float(trade.price)
        now = time.monotonic()

        last = self._last_fired.get(ticker, 0.0)
        if now - last < self._throttle:
            return

        self._last_fired[ticker] = now
        self._last_trade_time[ticker] = now
        self._known_stale.discard(ticker)
        self._crypto_connected.set()

        try:
            self._on_trade(ticker, price)
        except Exception:
            logger.exception("on_trade callback raised for crypto %s", ticker)

    def _run_stream(self) -> None:
        """Create and run the StockDataStream (blocking). Called from supervisor."""
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]

        with self._lock:
            symbols = set(self._symbols)

        if not symbols:
            logger.info("Equity stream: no symbols to subscribe — idle")
            return

        stream = StockDataStream(api_key, secret_key, feed=self._get_feed())
        stream.subscribe_trades(self._handle_trade, *symbols)
        logger.info(
            "Equity stream: subscribing to %d symbols (feed=%s): %s",
            len(symbols), self._feed_name, sorted(symbols),
        )

        with self._lock:
            self._stream = stream

        logger.info("Equity stream: connecting (feed=%s)", self._feed_name)

        try:
            stream.run()
            logger.info("Equity stream: disconnected cleanly")
        finally:
            with self._lock:
                self._stream = None
            self._connected.clear()
            logger.info("Equity stream: connection closed")

    def _run_crypto_stream(self) -> None:
        """Create and run the CryptoDataStream (blocking). Called from crypto supervisor."""
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]

        with self._lock:
            symbols = set(self._crypto_symbols)

        if not symbols:
            logger.info("Crypto stream: no symbols to subscribe — idle")
            return

        alpaca_syms = [self._to_alpaca_crypto_symbol(s) for s in symbols]
        stream = CryptoDataStream(api_key, secret_key)
        stream.subscribe_trades(self._handle_crypto_trade, *alpaca_syms)
        logger.info(
            "Crypto stream: subscribing to %d symbols: %s",
            len(alpaca_syms), sorted(alpaca_syms),
        )

        with self._lock:
            self._crypto_stream = stream

        logger.info("Crypto stream: connecting")

        try:
            stream.run()
            logger.info("Crypto stream: disconnected cleanly")
        finally:
            with self._lock:
                self._crypto_stream = None
            self._crypto_connected.clear()
            logger.info("Crypto stream: connection closed")

    def _supervisor_loop(self) -> None:
        """Restart the equity stream on unexpected exit with exponential backoff."""
        backoff = 1.0
        max_backoff = 60.0
        attempt = 0

        while not self._stop_event.is_set():
            attempt += 1
            try:
                self._run_stream()
                if self._stop_event.is_set():
                    break
                # If there are no equity symbols, _run_stream returns immediately —
                # that's expected, not a crash. Wait briefly and retry without
                # incrementing backoff so the supervisor is ready when symbols appear.
                with self._lock:
                    has_symbols = bool(self._symbols)
                if not has_symbols:
                    logger.debug("Equity stream: no symbols, idle for 5s")
                    attempt -= 1
                    self._stop_event.wait(timeout=5.0)
                    continue
                logger.warning(
                    "Equity stream exited unexpectedly (attempt %d), restarting in %.0fs",
                    attempt, backoff,
                )
            except Exception as e:
                logger.warning(
                    "Equity stream error (attempt %d): %s — restarting in %.0fs",
                    attempt, e, backoff,
                )

            self._connected.clear()

            logger.info("Equity stream: reconnect attempt %d in %.0fs", attempt + 1, backoff)
            self._stop_event.wait(timeout=backoff)
            if self._stop_event.is_set():
                break

            backoff = min(backoff * 2, max_backoff)
            logger.info("Equity stream: reconnect attempt %d starting", attempt + 1)

    def _crypto_supervisor_loop(self) -> None:
        """Restart the crypto stream on unexpected exit with exponential backoff."""
        backoff = 1.0
        max_backoff = 60.0
        attempt = 0

        while not self._stop_event.is_set():
            attempt += 1
            try:
                self._run_crypto_stream()
                if self._stop_event.is_set():
                    break
                # If there are no crypto symbols, _run_crypto_stream returns immediately —
                # that's expected, not a crash. Wait briefly and retry without
                # incrementing backoff so the supervisor is ready when symbols appear.
                with self._lock:
                    has_symbols = bool(self._crypto_symbols)
                if not has_symbols:
                    logger.debug("Crypto stream: no symbols, idle for 5s")
                    attempt -= 1
                    self._stop_event.wait(timeout=5.0)
                    continue
                logger.warning(
                    "Crypto stream exited unexpectedly (attempt %d), restarting in %.0fs",
                    attempt, backoff,
                )
            except Exception as e:
                logger.warning(
                    "Crypto stream error (attempt %d): %s — restarting in %.0fs",
                    attempt, e, backoff,
                )

            self._crypto_connected.clear()

            logger.info("Crypto stream: reconnect attempt %d in %.0fs", attempt + 1, backoff)
            self._stop_event.wait(timeout=backoff)
            if self._stop_event.is_set():
                break

            backoff = min(backoff * 2, max_backoff)
            logger.info("Crypto stream: reconnect attempt %d starting", attempt + 1)
