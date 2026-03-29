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
from typing import Literal

from pydantic import BaseModel

from db.events import log_order, log_price_alert
from db.positions import load_positions, save_position_meta
from integrations.alpaca.executor import (
    cancel_exit_orders,
    check_pending_fills,
    execute_exit,
    is_enabled as alpaca_enabled,
    place_exit_orders,
)
from trading.positions import (
    async_close_position,
    async_save_positions,
    sync_positions_from_alpaca,
    update_fill_prices,
    update_position_for_price,
)

_ET = zoneinfo.ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scan / regime / retrain event types (MIC-26)
# ---------------------------------------------------------------------------

class ScanResult(BaseModel):
    signals: list[dict]
    exits: list[dict]
    orders: list[dict]
    positions_summary: dict
    scan_type: str = "daily"  # "daily", "morning", "midday", "cascade", "post_retrain"
    equity: float = 0.0  # account equity at scan time (for order embed dollar amounts)
    error: str | None = None


class WatchlistEvent(BaseModel):
    signals: list[dict]
    scan_type: str = "midday"


class RegimeEvent(BaseModel):
    transitions: list[dict]
    cascade_triggered: bool


class RetrainResult(BaseModel):
    tickers_retrained: list[str]
    duration_seconds: float
    manifest_size: int


class RiskActionEvent(BaseModel):
    actions: list[dict]


class StreamDegradedEvent(BaseModel):
    disconnected_minutes: int  # consecutive market-hours minutes stream has been down


class StreamRecoveredEvent(BaseModel):
    downtime_minutes: int  # how long stream was down before recovery


# ---------------------------------------------------------------------------
# Typed event models
# ---------------------------------------------------------------------------

class AlertEvent(BaseModel):
    ticker: str
    alert_type: Literal["stop_breached", "target_breached", "near_stop", "near_target"]
    current_price: float
    level_price: float
    level_name: str
    entry_price: float
    pnl_pct: float
    distance_pct: float = 0.0


class ExitResult(BaseModel):
    ticker: str
    exit_reason: str
    fill_price: float | None = None
    order_id: str | None = None
    realized_pnl_pct: float | None = None
    bars_held: int | None = None


class FillNotification(BaseModel):
    ticker: str
    fill_price: float
    qty: float | None = None
    stop_price: float = 0.0
    target_price: float = 0.0


class HousekeepingResult(BaseModel):
    fills_confirmed: list[FillNotification]
    orders_placed: list[dict]
    positions_synced: bool


# ---------------------------------------------------------------------------
# AlertEngine
# ---------------------------------------------------------------------------

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

        if effective_stop > 0 and price <= effective_stop:
            if log_price_alert(today, ticker, "stop_breached", price, effective_stop, entry, pnl_pct):
                level_name = "Trailing Stop" if trailing > stop else "Stop"
                return AlertEvent(
                    ticker=ticker,
                    alert_type="stop_breached",
                    current_price=price,
                    level_price=effective_stop,
                    level_name=level_name,
                    entry_price=entry,
                    pnl_pct=pnl_pct,
                )

        elif target > 0 and price >= target:
            if log_price_alert(today, ticker, "target_breached", price, target, entry, pnl_pct):
                return AlertEvent(
                    ticker=ticker,
                    alert_type="target_breached",
                    current_price=price,
                    level_price=target,
                    level_name="Target",
                    entry_price=entry,
                    pnl_pct=pnl_pct,
                )

        elif self._proximity_pct > 0 and effective_stop > 0:
            distance = (price / effective_stop - 1) * 100
            if 0 < distance <= self._proximity_pct:
                if log_price_alert(today, ticker, "near_stop", price, effective_stop, entry, pnl_pct):
                    level_name = "Trailing Stop" if trailing > stop else "Stop"
                    return AlertEvent(
                        ticker=ticker,
                        alert_type="near_stop",
                        current_price=price,
                        level_price=effective_stop,
                        level_name=level_name,
                        entry_price=entry,
                        pnl_pct=pnl_pct,
                        distance_pct=round(distance, 2),
                    )
            # Not near stop — still check near_target
            if target > 0:
                distance = (target / price - 1) * 100
                if 0 < distance <= self._proximity_pct:
                    if log_price_alert(today, ticker, "near_target", price, target, entry, pnl_pct):
                        return AlertEvent(
                            ticker=ticker,
                            alert_type="near_target",
                            current_price=price,
                            level_price=target,
                            level_name="Target",
                            entry_price=entry,
                            pnl_pct=pnl_pct,
                            distance_pct=round(distance, 2),
                        )

        elif self._proximity_pct > 0 and target > 0:
            distance = (target / price - 1) * 100
            if 0 < distance <= self._proximity_pct:
                if log_price_alert(today, ticker, "near_target", price, target, entry, pnl_pct):
                    return AlertEvent(
                        ticker=ticker,
                        alert_type="near_target",
                        current_price=price,
                        level_price=target,
                        level_name="Target",
                        entry_price=entry,
                        pnl_pct=pnl_pct,
                        distance_pct=round(distance, 2),
                    )

        return None

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
        """Cancel OCO exit orders then execute a market sell. Delegates to executor.py."""
        await cancel_exit_orders(pos.get("target_order_id"), pos.get("trail_order_id"))
        try:
            result = await execute_exit(ticker)
            fill = result.fill_price or price
            closed = await async_close_position(ticker, fill, reason)
            log_order(
                ticker, "sell", "market", result.qty,
                f"exit_{reason}", result.order_id,
                "filled" if result.success else "failed",
                result.fill_price, result.error,
            )
            return ExitResult(
                ticker=ticker,
                exit_reason=reason,
                fill_price=result.fill_price,
                order_id=result.order_id,
                realized_pnl_pct=closed.get("realized_pnl_pct") if closed else None,
                bars_held=closed.get("bars_held") if closed else None,
            )
        except Exception:
            logger.exception("Alpaca exit failed for %s", ticker)
            return None

    async def run_housekeeping(self, positions: list[dict]) -> HousekeepingResult:
        """Sync positions, check pending fills, place exit orders for new fills."""
        fills_confirmed: list[FillNotification] = []
        orders_placed: list[dict] = []

        if alpaca_enabled():
            await sync_positions_from_alpaca()

        if alpaca_enabled():
            pending_ids = [p.get("order_id") for p in positions if p.get("order_id")]
            if pending_ids:
                fills = await check_pending_fills(pending_ids)
                if fills:
                    from db.events import update_order_fill
                    for oid, fp in fills.items():
                        update_order_fill(oid, fp)
                    n_filled = await update_fill_prices(fills)
                    if n_filled:
                        logger.info("Updated %d fill prices from Alpaca", n_filled)
                        fills_confirmed = [
                            FillNotification(
                                ticker=p["ticker"],
                                fill_price=fills[p["order_id"]],
                                qty=p.get("qty"),
                                stop_price=p.get("stop_price", 0.0),
                                target_price=p.get("target_price", 0.0),
                            )
                            for p in positions
                            if p.get("order_id") in fills
                        ]

            # Place exit orders for newly filled entries (no order_id → fill confirmed)
            fresh_state = load_positions()
            for pos in fresh_state.get("positions", []):
                if (not pos.get("order_id")
                        and not pos.get("target_order_id")
                        and not pos.get("trail_order_id")
                        and pos.get("target_price") and pos.get("stop_price")
                        and pos.get("qty")):
                    effective_stop = max(
                        pos.get("stop_price", 0), pos.get("trailing_stop", 0),
                    )
                    t_oid, s_oid = await place_exit_orders(
                        pos["ticker"], pos["qty"],
                        pos["target_price"], effective_stop,
                    )
                    if t_oid or s_oid:
                        pos["target_order_id"] = t_oid
                        pos["trail_order_id"] = s_oid
                        if t_oid:
                            log_order(pos["ticker"], "sell", "limit",
                                      pos["qty"], "exit_target", t_oid, "pending")
                        if s_oid:
                            log_order(pos["ticker"], "sell", "stop",
                                      pos["qty"], "exit_stop", s_oid, "pending")
                        logger.info(
                            "Exit orders placed for %s: target=%s stop=%s",
                            pos["ticker"], t_oid, s_oid,
                        )
                        orders_placed.append({
                            "ticker": pos["ticker"],
                            "target_order_id": t_oid,
                            "stop_order_id": s_oid,
                        })
            if orders_placed:
                await async_save_positions(fresh_state)

        return HousekeepingResult(
            fills_confirmed=fills_confirmed,
            orders_placed=orders_placed,
            positions_synced=alpaca_enabled(),
        )
