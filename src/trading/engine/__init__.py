"""
Alert evaluation engine — decoupled from Discord.

Evaluates price thresholds, profit lock, and breach exits for open positions.
Zero Discord imports.
"""

import asyncio
import logging
import os
import zoneinfo
from datetime import datetime

from db.ops.events import log_price_alert
from db.trading.positions import save_position_meta
from trading.engine.housekeeping import (
    confirm_pending_fills,
    execute_breach,
    place_exit_orders_for_fills,
    run_housekeeping,
)
from trading.events import (  # noqa: F401 — re-export for backward compat
    AlertEvent,
    ExitResult,
    FillNotification,
    HousekeepingResult,
    RegimeEvent,
    RetrainResult,
    RiskActionEvent,
    ScanResult,
    StreamDegradedEvent,
    StreamRecoveredEvent,
    WatchlistEvent,
)
from trading.positions import update_position_for_price

_ET = zoneinfo.ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class AlertEngine:
    def __init__(self) -> None:
        self._proximity_pct = float(os.environ.get("ALERT_PROXIMITY_PCT", "1.5"))

    def is_market_open(self) -> bool:
        """True if Mon-Fri 9:30 AM - 4:00 PM ET."""
        now = datetime.now(_ET)
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    @staticmethod
    def evaluate_breach(
        ticker: str, price: float,
        effective_stop: float, target: float, entry: float, pnl_pct: float,
        stop: float, trailing: float,
    ) -> AlertEvent | None:
        """Pure evaluator: check hard stop/target breach. No DB calls."""
        if effective_stop > 0 and price <= effective_stop:
            level_name = "Trailing Stop" if trailing > stop else "Stop"
            return AlertEvent(
                ticker=ticker, alert_type="stop_breached", current_price=price,
                level_price=effective_stop, level_name=level_name,
                entry_price=entry, pnl_pct=pnl_pct,
            )
        if target > 0 and price >= target:
            return AlertEvent(
                ticker=ticker, alert_type="target_breached", current_price=price,
                level_price=target, level_name="Target",
                entry_price=entry, pnl_pct=pnl_pct,
            )
        return None

    def evaluate_proximity(
        self, ticker: str, price: float,
        effective_stop: float, target: float, entry: float, pnl_pct: float,
        stop: float, trailing: float,
    ) -> AlertEvent | None:
        """Pure evaluator: check near-stop/near-target proximity. No DB calls."""
        if self._proximity_pct <= 0:
            return None
        if effective_stop > 0:
            distance = (price / effective_stop - 1) * 100
            if 0 < distance <= self._proximity_pct:
                level_name = "Trailing Stop" if trailing > stop else "Stop"
                return AlertEvent(
                    ticker=ticker, alert_type="near_stop", current_price=price,
                    level_price=effective_stop, level_name=level_name,
                    entry_price=entry, pnl_pct=pnl_pct, distance_pct=round(distance, 2),
                )
        if target > 0:
            distance = (target / price - 1) * 100
            if 0 < distance <= self._proximity_pct:
                return AlertEvent(
                    ticker=ticker, alert_type="near_target", current_price=price,
                    level_price=target, level_name="Target",
                    entry_price=entry, pnl_pct=pnl_pct, distance_pct=round(distance, 2),
                )
        return None

    def _check_breach(
        self, ticker: str, price: float, pos: dict, today: str,
        effective_stop: float, target: float, entry: float, pnl_pct: float,
        stop: float, trailing: float,
    ) -> AlertEvent | None:
        """Evaluate breach + dedup via DB log."""
        event = self.evaluate_breach(
            ticker, price, effective_stop, target, entry, pnl_pct, stop, trailing,
        )
        if event and log_price_alert(
            today, ticker, event.alert_type, price, event.level_price, entry, pnl_pct,
        ):
            return event
        return None

    def _check_proximity(
        self, ticker: str, price: float, pos: dict, today: str,
        effective_stop: float, target: float, entry: float, pnl_pct: float,
        stop: float, trailing: float,
    ) -> AlertEvent | None:
        """Evaluate proximity + dedup via DB log."""
        event = self.evaluate_proximity(
            ticker, price, effective_stop, target, entry, pnl_pct, stop, trailing,
        )
        if event and log_price_alert(
            today, ticker, event.alert_type, price, event.level_price, entry, pnl_pct,
        ):
            return event
        return None

    def evaluate_single_ticker(
        self, ticker: str, price: float, pos: dict,
    ) -> AlertEvent | None:
        """Evaluate a single ticker for alert conditions."""
        today = datetime.now(_ET).strftime("%Y-%m-%d")
        entry = pos["entry_price"]
        stop = pos.get("stop_price", 0)
        target = pos.get("target_price", 0)
        trailing = pos.get("trailing_stop", 0)
        pnl_pct = round((price / entry - 1) * 100, 2) if entry else 0
        effective_stop = max(stop, trailing)

        return (
            self._check_breach(
                ticker, price, pos, today, effective_stop, target, entry, pnl_pct, stop, trailing,
            )
            or self._check_proximity(
                ticker, price, pos, today, effective_stop, target, entry, pnl_pct, stop, trailing,
            )
        )

    async def check_prices(
        self, positions: list[dict], quotes: dict[str, dict],
    ) -> list[AlertEvent]:
        """Evaluate all positions for alert conditions. Applies profit lock."""
        events, changed = self.evaluate_prices(positions, quotes)
        for pos in changed:
            await asyncio.to_thread(save_position_meta, pos)
        return events

    def evaluate_prices(
        self, positions: list[dict], quotes: dict[str, dict],
    ) -> tuple[list[AlertEvent], list[dict]]:
        """Pure evaluation: returns (events, positions_to_save). No DB calls."""
        events: list[AlertEvent] = []
        changed: list[dict] = []

        for pos in positions:
            ticker = pos["ticker"]
            quote = quotes.get(ticker)
            if not quote or "error" in quote:
                continue

            price = quote["price"]

            if update_position_for_price(pos, price):
                changed.append(pos)

            event = self.evaluate_single_ticker(ticker, price, pos)
            if event:
                events.append(event)

        return events, changed

    # Delegate to housekeeping module
    execute_breach = staticmethod(execute_breach)

    async def _confirm_pending_fills(
        self, positions: list[dict],
    ) -> list[FillNotification]:
        return await confirm_pending_fills(positions)

    async def _place_exit_orders_for_fills(self) -> list[dict]:
        return await place_exit_orders_for_fills()

    async def run_housekeeping(self, positions: list[dict]) -> HousekeepingResult:
        return await run_housekeeping(positions)
