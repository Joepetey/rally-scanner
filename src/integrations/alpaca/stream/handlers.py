"""Trade event handlers and throttling logic."""

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class TradeThrottle:
    """Per-ticker throttle that deduplicates and fires the on_trade callback."""

    def __init__(
        self,
        on_trade: Callable[[str, float], None],
        throttle_seconds: float,
    ) -> None:
        self._on_trade = on_trade
        self._throttle = throttle_seconds
        self._last_fired: dict[str, float] = {}
        self._last_trade_time: dict[str, float] = {}
        self._known_stale: set[str] = set()

    @property
    def last_trade_time(self) -> dict[str, float]:
        return self._last_trade_time

    @property
    def known_stale(self) -> set[str]:
        return self._known_stale

    def fire_trade(
        self, ticker: str, price: float, connected_event: threading.Event,
    ) -> None:
        """Throttle, dedup, and fire the on_trade callback for a ticker."""
        now = time.monotonic()
        last = self._last_fired.get(ticker, 0.0)
        if now - last < self._throttle:
            return
        self._last_fired[ticker] = now
        self._last_trade_time[ticker] = now
        self._known_stale.discard(ticker)
        connected_event.set()
        try:
            self._on_trade(ticker, price)
        except Exception:
            logger.exception("on_trade callback raised for %s", ticker)

    async def handle_equity_trade(
        self, trade, connected_event: threading.Event,
    ) -> None:
        """Called from StockDataStream's internal event loop per trade event."""
        self.fire_trade(str(trade.symbol), float(trade.price), connected_event)

    async def handle_crypto_trade(
        self, trade, connected_event: threading.Event,
    ) -> None:
        """Called from CryptoDataStream — normalizes BTC/USD to BTC."""
        ticker = str(trade.symbol).split("/")[0]
        self.fire_trade(ticker, float(trade.price), connected_event)
