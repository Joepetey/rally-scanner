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
from collections.abc import Callable

from rally_ml.config import PARAMS

from .handlers import TradeThrottle
from .stale import detect_stale_tickers
from .subscriptions import apply_diff, crypto_symbol_mapper
from .supervisor import get_feed, run_stream_impl, run_supervisor
from .symbols import is_crypto, to_alpaca_crypto_symbol

logger = logging.getLogger(__name__)

try:
    from alpaca.data.live import CryptoDataStream, StockDataStream
except ImportError:
    pass


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

    def __init__(
        self,
        on_trade: Callable[[str, float], None],
        excluded: set[str] | None = None,
    ) -> None:
        self._excluded = excluded or set()
        self._feed_name = os.environ.get("ALPACA_DATA_FEED", PARAMS.stream_data_feed)

        throttle_seconds = float(
            os.environ.get(
                "STREAM_EVAL_THROTTLE_SECONDS",
                str(PARAMS.stream_eval_throttle_seconds),
            )
        )
        self._throttle_seconds = throttle_seconds
        self._handler = TradeThrottle(on_trade, throttle_seconds)

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

        # Equity connected: set True once first trade event arrives
        self._connected = threading.Event()
        # Crypto connected: set True once first trade event arrives
        self._crypto_connected = threading.Event()

    @property
    def _on_trade(self) -> Callable[[str, float], None]:
        return self._handler._on_trade

    @property
    def _known_stale(self) -> set[str]:
        return self._handler.known_stale

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() or self._crypto_connected.is_set()

    def start(self, symbols: set[str]) -> None:
        """Launch equity and/or crypto streams in daemon threads. Thread-safe."""
        symbols = symbols - self._excluded
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
                    len(self._symbols), self._feed_name, self._throttle_seconds,
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
                    len(self._crypto_symbols), self._throttle_seconds,
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
        symbols = symbols - self._excluded
        equity_symbols = {s for s in symbols if not self._is_crypto(s)}
        crypto_symbols = {s for s in symbols if self._is_crypto(s)}

        with self._lock:
            current_equity = set(self._symbols)
            current_crypto = set(self._crypto_symbols)
            equity_stream = self._stream
            crypto_stream = self._crypto_stream

        new_eq = apply_diff(
            "Equity", equity_symbols, current_equity,
            equity_stream, self._connected, self._handle_trade,
        )
        if new_eq is not None:
            with self._lock:
                self._symbols = new_eq

        new_cr = apply_diff(
            "Crypto", crypto_symbols, current_crypto,
            crypto_stream, self._crypto_connected, self._handle_crypto_trade,
            symbol_mapper=crypto_symbol_mapper,
        )
        if new_cr is not None:
            with self._lock:
                self._crypto_symbols = new_cr

    def get_stale_tickers(
        self, stale_seconds: float = 300.0,
    ) -> tuple[list[str], list[str], list[str]]:
        """Return (new_stale, known_stale, never_traded) for tickers with no trade event.

        Side effect: new_stale and never_traded tickers are added to the internal
        _known_stale set so subsequent calls demote them to known_stale.
        """
        with self._lock:
            symbols = set(self._symbols) | set(self._crypto_symbols)

        new_stale, known_stale, never_traded = detect_stale_tickers(
            symbols, self._handler.last_trade_time, self._handler.known_stale, stale_seconds,
        )
        # Maintain the side-effect contract: add newly detected to known_stale
        self._handler.known_stale.update(new_stale)
        self._handler.known_stale.update(never_traded)
        return new_stale, known_stale, never_traded

    def inject_trade(self, ticker: str, price: float) -> None:
        """Fire _on_trade directly, bypassing WebSocket and throttle.

        Used by simulation to inject prices through the real stream code path.
        """
        self._handler._on_trade(ticker, price)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _is_equity(ticker: str) -> bool:
        """Exclude crypto tickers (StockDataStream is equity-only)."""
        return not ticker.endswith("-USD")

    _is_crypto = staticmethod(is_crypto)
    _to_alpaca_crypto_symbol = staticmethod(to_alpaca_crypto_symbol)

    async def _handle_trade(self, trade) -> None:
        await self._handler.handle_equity_trade(trade, self._connected)

    async def _handle_crypto_trade(self, trade) -> None:
        await self._handler.handle_crypto_trade(trade, self._crypto_connected)

    def _run_stream(self) -> None:
        """Create and run the StockDataStream (blocking)."""
        with self._lock:
            symbols = set(self._symbols)
        run_stream_impl(
            symbols=symbols,
            stream_attr_setter=lambda s: setattr(self, '_stream', s) or None,
            connected_event=self._connected,
            label=f"Equity (feed={self._feed_name})",
            make_stream=lambda k, s: StockDataStream(k, s, feed=get_feed(self._feed_name)),
            make_symbols=lambda syms: list(syms),
            handler=self._handle_trade,
            lock=self._lock,
        )

    def _run_crypto_stream(self) -> None:
        """Create and run the CryptoDataStream (blocking)."""
        with self._lock:
            symbols = set(self._crypto_symbols)
        run_stream_impl(
            symbols=symbols,
            stream_attr_setter=lambda s: setattr(self, '_crypto_stream', s) or None,
            connected_event=self._crypto_connected,
            label="Crypto",
            make_stream=lambda k, s: CryptoDataStream(k, s),
            make_symbols=lambda syms: [to_alpaca_crypto_symbol(s) for s in syms],
            handler=self._handle_crypto_trade,
            lock=self._lock,
        )

    def _supervisor_loop(self) -> None:
        run_supervisor(
            self._run_stream,
            lambda: bool(self._symbols),
            self._connected,
            self._stop_event,
            "Equity",
        )

    def _crypto_supervisor_loop(self) -> None:
        run_supervisor(
            self._run_crypto_stream,
            lambda: bool(self._crypto_symbols),
            self._crypto_connected,
            self._stop_event,
            "Crypto",
        )
