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
from typing import Any, Protocol

from db.events import log_order, log_price_alert
from db.positions import save_position_meta
from integrations.alpaca.executor import (
    cancel_exit_orders,
    execute_exit,
    get_recent_sell_fills,
)
from trading.events import (  # noqa: F401 — re-export
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
from trading.positions import (
    async_close_position,
    update_position_for_price,
)

_ET = zoneinfo.ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Broker operations protocol for execute_breach dependency injection
# ---------------------------------------------------------------------------

class BrokerOps(Protocol):
    async def cancel_exit_orders(
        self, target_order_id: str | None, trail_order_id: str | None,
    ) -> None: ...

    async def execute_exit(self, ticker: str) -> Any: ...

    async def get_recent_sell_fills(
        self, tickers: list[str],
    ) -> dict[str, float]: ...

    async def close_position(
        self, ticker: str, fill_price: float, reason: str,
    ) -> dict | None: ...

    def log_order(
        self, ticker: str, side: str, order_type: str, qty: float | None,
        reason: str, order_id: str | None, status: str,
        fill_price: float | None, error: str | None,
    ) -> None: ...


class _DefaultBrokerOps:
    """Wraps existing executor/positions functions as BrokerOps."""

    async def cancel_exit_orders(
        self, target_order_id: str | None, trail_order_id: str | None,
    ) -> None:
        await cancel_exit_orders(target_order_id, trail_order_id)

    async def execute_exit(self, ticker: str) -> Any:
        return await execute_exit(ticker)

    async def get_recent_sell_fills(
        self, tickers: list[str],
    ) -> dict[str, float]:
        return await get_recent_sell_fills(tickers)

    async def close_position(
        self, ticker: str, fill_price: float, reason: str,
    ) -> dict | None:
        return await async_close_position(ticker, fill_price, reason)

    def log_order(
        self, ticker: str, side: str, order_type: str, qty: float | None,
        reason: str, order_id: str | None, status: str,
        fill_price: float | None, error: str | None,
    ) -> None:
        log_order(
            ticker, side, order_type, qty, reason,
            order_id, status, fill_price, error,
        )


# ---------------------------------------------------------------------------
# AlertEngine
# ---------------------------------------------------------------------------

class AlertEngine:
    def __init__(self, broker: BrokerOps | None = None) -> None:
        self._proximity_pct = float(os.environ.get("ALERT_PROXIMITY_PCT", "1.5"))
        self._broker: BrokerOps = broker or _DefaultBrokerOps()

    def is_market_open(self) -> bool:
        """True if Mon-Fri 9:30 AM - 4:00 PM ET."""
        now = datetime.now(_ET)
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    def _check_breach(
        self, ticker: str, price: float, pos: dict, today: str,
        effective_stop: float, target: float, entry: float, pnl_pct: float,
        stop: float, trailing: float,
    ) -> AlertEvent | None:
        """Check hard stop/target breach conditions. Returns event or None."""
        if effective_stop > 0 and price <= effective_stop:
            if log_price_alert(  # noqa: E501
                today, ticker, "stop_breached", price, effective_stop, entry, pnl_pct
            ):
                level_name = "Trailing Stop" if trailing > stop else "Stop"
                return AlertEvent(
                    ticker=ticker, alert_type="stop_breached", current_price=price,
                    level_price=effective_stop, level_name=level_name,
                    entry_price=entry, pnl_pct=pnl_pct,
                )
        elif target > 0 and price >= target:
            if log_price_alert(today, ticker, "target_breached", price, target, entry, pnl_pct):
                return AlertEvent(
                    ticker=ticker, alert_type="target_breached", current_price=price,
                    level_price=target, level_name="Target",
                    entry_price=entry, pnl_pct=pnl_pct,
                )
        return None

    def _check_proximity(
        self, ticker: str, price: float, pos: dict, today: str,
        effective_stop: float, target: float, entry: float, pnl_pct: float,
        stop: float, trailing: float,
    ) -> AlertEvent | None:
        """Check near-stop and near-target proximity conditions. Returns event or None."""
        if self._proximity_pct <= 0:
            return None
        if effective_stop > 0:
            distance = (price / effective_stop - 1) * 100
            if 0 < distance <= self._proximity_pct:
                if log_price_alert(  # noqa: E501
                    today, ticker, "near_stop", price, effective_stop, entry, pnl_pct
                ):
                    level_name = "Trailing Stop" if trailing > stop else "Stop"
                    return AlertEvent(
                        ticker=ticker, alert_type="near_stop", current_price=price,
                        level_price=effective_stop, level_name=level_name,
                        entry_price=entry, pnl_pct=pnl_pct, distance_pct=round(distance, 2),
                    )
        if target > 0:
            distance = (target / price - 1) * 100
            if 0 < distance <= self._proximity_pct:
                if log_price_alert(today, ticker, "near_target", price, target, entry, pnl_pct):
                    return AlertEvent(
                        ticker=ticker, alert_type="near_target", current_price=price,
                        level_price=target, level_name="Target",
                        entry_price=entry, pnl_pct=pnl_pct, distance_pct=round(distance, 2),
                    )
        return None

    def evaluate_single_ticker(
        self, ticker: str, price: float, pos: dict,
    ) -> AlertEvent | None:
        """Evaluate a single ticker for alert conditions.

        Sync and safe to call from any thread (DB uses ThreadedConnectionPool).
        Applies profit lock mutation to pos in-place; caller is responsible for
        persisting if needed (check_prices handles this for the polling path).
        """
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
        events: list[AlertEvent] = []

        for pos in positions:
            ticker = pos["ticker"]
            quote = quotes.get(ticker)
            if not quote or "error" in quote:
                continue

            price = quote["price"]

            if update_position_for_price(pos, price):
                await asyncio.to_thread(save_position_meta, pos)

            event = self.evaluate_single_ticker(ticker, price, pos)
            if event:
                events.append(event)

        return events

    async def execute_breach(
        self, ticker: str, pos: dict, price: float, reason: str,
    ) -> ExitResult | None:
        """Cancel OCO exit orders then execute a market sell via broker ops."""
        broker = self._broker
        await broker.cancel_exit_orders(
            pos.get("target_order_id"), pos.get("trail_order_id"),
        )
        try:
            result = await broker.execute_exit(ticker)
            if result.already_closed:
                # OCO/trailing stop already filled on the broker — fetch the actual fill
                # price so DB closes with the real execution price, not the current quote.
                broker_fills = await broker.get_recent_sell_fills([ticker])
                fill = broker_fills.get(ticker) or price
                reason = "oco_fill"
            else:
                fill = result.fill_price or price
            closed = await broker.close_position(ticker, fill, reason)
            broker.log_order(
                ticker, "sell", "market", result.qty,
                f"exit_{reason}", result.order_id,
                "filled" if result.success else "failed",
                fill, result.error,
            )
            return ExitResult(
                ticker=ticker,
                exit_reason=reason,
                fill_price=fill,
                order_id=result.order_id,
                realized_pnl_pct=closed.get("realized_pnl_pct") if closed else None,
                bars_held=closed.get("bars_held") if closed else None,
            )
        except Exception:
            logger.exception("Alpaca exit failed for %s", ticker)
            return None

    async def run_housekeeping(self, positions: list[dict]) -> HousekeepingResult:
        """Delegate to standalone housekeeping module."""
        from trading.housekeeping import run_housekeeping as _run
        return await _run(positions)
