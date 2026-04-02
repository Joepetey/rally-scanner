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

import rally_ml.config as config

from db.events import log_order, log_price_alert
from db.positions import load_positions, save_position_meta
from integrations.alpaca.exits import cancel_exit_orders, execute_exit, place_exit_orders
from integrations.alpaca.fills import check_pending_fills, get_recent_sell_fills
from integrations.alpaca.models import is_enabled as alpaca_enabled
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
        """Evaluate all positions for alert conditions. Applies profit lock.

        Persists changed positions inline. For callers that need to control
        persistence, use evaluate_prices() instead.
        """
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

    async def execute_breach(
        self, ticker: str, pos: dict, price: float, reason: str,
    ) -> ExitResult | None:
        """Cancel OCO exit orders then execute a market sell. Delegates to executor.py."""
        await cancel_exit_orders(pos.get("target_order_id"), pos.get("trail_order_id"))
        try:
            result = await execute_exit(ticker)
            if result.already_closed:
                # OCO/trailing stop already filled on the broker — fetch the actual fill
                # price so DB closes with the real execution price, not the current quote.
                broker_fills = await get_recent_sell_fills([ticker])
                fill = broker_fills.get(ticker) or price
                reason = "oco_fill"
            else:
                fill = result.fill_price or price
            closed = await async_close_position(ticker, fill, reason)
            log_order(
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

    async def _confirm_pending_fills(
        self, positions: list[dict],
    ) -> list[FillNotification]:
        """Check pending entry orders and update fill prices. Returns confirmed fills."""
        pending_ids = [p.get("order_id") for p in positions if p.get("order_id")]
        if not pending_ids:
            return []
        fills = await check_pending_fills(pending_ids)
        if not fills:
            return []

        from db.events import update_order_fill
        for oid, fp in fills.items():
            update_order_fill(oid, fp)
        n_filled = await update_fill_prices(fills)
        if n_filled:
            logger.info("Updated %d fill prices from Alpaca", n_filled)
        return [
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

    async def _place_exit_orders_for_fills(self) -> list[dict]:
        """Place OCO exit orders for newly filled entries. Returns orders placed."""
        # Crypto positions excluded: Alpaca doesn't support OCO for crypto.
        orders_placed: list[dict] = []
        fresh_state = load_positions()
        for pos in fresh_state.get("positions", []):
            ticker = pos["ticker"]
            is_crypto = (
                ticker in config.ASSETS
                and config.ASSETS[ticker].asset_class == "crypto"
            )
            if (not is_crypto
                    and not pos.get("order_id")
                    and not pos.get("target_order_id")
                    and not pos.get("trail_order_id")
                    and pos.get("target_price") and pos.get("stop_price")
                    and pos.get("qty")):
                effective_stop = max(pos.get("stop_price", 0), pos.get("trailing_stop", 0))
                t_oid, s_oid = await place_exit_orders(
                    ticker, pos["qty"], pos["target_price"], effective_stop,
                )
                if t_oid or s_oid:
                    pos["target_order_id"] = t_oid
                    pos["trail_order_id"] = s_oid
                    if t_oid:
                        log_order(ticker, "sell", "limit", pos["qty"], "exit_target", t_oid, "pending")  # noqa: E501
                    if s_oid:
                        log_order(ticker, "sell", "stop", pos["qty"], "exit_stop", s_oid, "pending")
                    logger.info("Exit orders placed for %s: target=%s stop=%s", ticker, t_oid, s_oid)  # noqa: E501
                    orders_placed.append({
                        "ticker": ticker,
                        "target_order_id": t_oid,
                        "stop_order_id": s_oid,
                    })
        if orders_placed:
            await async_save_positions(fresh_state)
        return orders_placed

    async def run_housekeeping(self, positions: list[dict]) -> HousekeepingResult:
        """Sync positions, check pending fills, place exit orders for new fills."""
        fills_confirmed: list[FillNotification] = []
        orders_placed: list[dict] = []

        if alpaca_enabled():
            try:
                await sync_positions_from_alpaca()
            except Exception:
                logger.exception("Housekeeping: Alpaca sync failed")

            fills_confirmed = await self._confirm_pending_fills(positions)
            orders_placed = await self._place_exit_orders_for_fills()

        return HousekeepingResult(
            fills_confirmed=fills_confirmed,
            orders_placed=orders_placed,
            positions_synced=alpaca_enabled(),
        )
