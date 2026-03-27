"""
Alpaca real-time market data stream manager.

Runs StockDataStream in its own daemon thread with a dedicated event loop.
Public API is fully sync and thread-safe; the on_trade callback is called
from the stream thread so callers must handle thread-safety themselves.

No Discord imports. No asyncio dependency in the public API.
"""

import logging
import os
import threading
import time
from collections.abc import Callable

from config import PARAMS

logger = logging.getLogger(__name__)

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.live import StockDataStream
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
    """Manages an Alpaca WebSocket stream in a dedicated daemon thread.

    Usage::

        def on_trade(ticker: str, price: float) -> None:
            ...  # called from stream thread — be thread-safe

        mgr = AlpacaStreamManager(on_trade=on_trade)
        mgr.start(symbols={"AAPL", "MSFT"})
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

        self._symbols: set[str] = set()
        self._stream: "StockDataStream | None" = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Per-ticker last-fired timestamps for throttle
        self._last_fired: dict[str, float] = {}

        # is_connected is set True once the first trade event arrives
        self._connected = threading.Event()

        # Track last trade event time per ticker for stale detection
        self._last_trade_time: dict[str, float] = {}

        # Tickers already warned as stale; suppresses repeat WARNINGs until a fresh trade arrives
        self._known_stale: set[str] = set()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self, symbols: set[str]) -> None:
        """Launch the stream in a daemon thread. Thread-safe."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._symbols = {s for s in symbols if self._is_equity(s)}
            self._stop_event.clear()
            self._connected.clear()
            self._thread = threading.Thread(
                target=self._supervisor_loop, daemon=True, name="alpaca-stream",
            )
            self._thread.start()
            logger.info(
                "Stream started for %d equity symbols (feed=%s, throttle=%.0fs)",
                len(self._symbols), self._feed_name, self._throttle,
            )

    def stop(self) -> None:
        """Signal the stream to stop. Thread-safe."""
        self._stop_event.set()
        self._connected.clear()
        with self._lock:
            stream = self._stream
        if stream:
            try:
                stream.stop()
            except Exception as e:
                logger.debug("Stream stop error (expected): %s", e)
        logger.info("Stream stopped")

    def update_subscriptions(self, symbols: set[str]) -> None:
        """Diff the symbol set and subscribe/unsubscribe as needed. Thread-safe."""
        equity_symbols = {s for s in symbols if self._is_equity(s)}
        with self._lock:
            current = set(self._symbols)
            stream = self._stream

        to_add = equity_symbols - current
        to_remove = current - equity_symbols

        if not (to_add or to_remove):
            return

        logger.info(
            "Stream subscriptions: +%d -%d (total %d)",
            len(to_add), len(to_remove), len(equity_symbols),
        )

        with self._lock:
            self._symbols = equity_symbols

        if stream and self.is_connected:
            if to_add:
                try:
                    stream.subscribe_trades(self._handle_trade, *to_add)
                    logger.info("Subscribed to %s", sorted(to_add))
                except Exception as e:
                    logger.warning("Failed to subscribe to %s: %s", to_add, e)
            if to_remove:
                try:
                    stream.unsubscribe_trades(*to_remove)
                    logger.info("Unsubscribed from %s", sorted(to_remove))
                except Exception as e:
                    logger.warning("Failed to unsubscribe from %s: %s", to_remove, e)

    def get_stale_tickers(self, stale_seconds: float = 300.0) -> tuple[list[str], list[str]]:
        """Return (new_stale, known_stale) for tickers with no trade event in stale_seconds.

        new_stale  — first occurrence since last fresh trade; callers should log WARNING.
        known_stale — already reported and still stale; callers should log DEBUG only.

        Side effect: new_stale tickers are added to the internal _known_stale set so
        subsequent calls demote them to known_stale until a fresh trade clears them.
        """
        now = time.monotonic()
        with self._lock:
            symbols = set(self._symbols)
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _is_equity(ticker: str) -> bool:
        """Exclude crypto tickers (StockDataStream is equity-only)."""
        return not ticker.endswith("-USD")

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

    def _run_stream(self) -> None:
        """Create and run the StockDataStream (blocking). Called from supervisor."""
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]

        with self._lock:
            symbols = set(self._symbols)

        if not symbols:
            logger.info("Stream: no equity symbols to subscribe — idle")
            return

        stream = StockDataStream(api_key, secret_key, feed=self._get_feed())
        stream.subscribe_trades(self._handle_trade, *symbols)
        logger.info(
            "Stream: subscribing to %d symbols (feed=%s): %s",
            len(symbols), self._feed_name, sorted(symbols),
        )

        with self._lock:
            self._stream = stream

        logger.info("Stream: connecting (feed=%s)", self._feed_name)

        try:
            stream.run()  # blocks; creates its own event loop
            logger.info("Stream: disconnected cleanly")
        finally:
            with self._lock:
                self._stream = None
            self._connected.clear()
            logger.info("Stream: connection closed")

    def _supervisor_loop(self) -> None:
        """Restart the stream on unexpected exit with exponential backoff."""
        backoff = 1.0
        max_backoff = 60.0
        attempt = 0

        while not self._stop_event.is_set():
            attempt += 1
            try:
                self._run_stream()
                if self._stop_event.is_set():
                    break
                # Unexpected exit (not via stop()) — schedule restart
                logger.warning(
                    "Stream exited unexpectedly (attempt %d), restarting in %.0fs",
                    attempt, backoff,
                )
            except Exception as e:
                logger.warning(
                    "Stream error (attempt %d): %s — restarting in %.0fs",
                    attempt, e, backoff,
                )

            self._connected.clear()

            logger.info("Stream: reconnect attempt %d in %.0fs", attempt + 1, backoff)
            self._stop_event.wait(timeout=backoff)
            if self._stop_event.is_set():
                break

            backoff = min(backoff * 2, max_backoff)
            logger.info("Stream: reconnect attempt %d starting", attempt + 1)
