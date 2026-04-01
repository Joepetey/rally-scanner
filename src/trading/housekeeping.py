"""Housekeeping tasks — fill confirmation, exit order placement, position sync."""

import logging

import config
from db.events import log_order, update_order_fill
from db.positions import load_positions
from integrations.alpaca.executor import (
    check_pending_fills,
    place_exit_orders,
)
from integrations.alpaca.executor import (
    is_enabled as alpaca_enabled,
)
from trading.engine import FillNotification, HousekeepingResult
from trading.positions import (
    async_save_positions,
    sync_positions_from_alpaca,
    update_fill_prices,
)

logger = logging.getLogger(__name__)


async def _confirm_pending_fills(
    positions: list[dict],
) -> list[FillNotification]:
    """Check pending entry orders and update fill prices. Returns confirmed fills."""
    pending_ids = [p.get("order_id") for p in positions if p.get("order_id")]
    if not pending_ids:
        return []
    fills = await check_pending_fills(pending_ids)
    if not fills:
        return []

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


async def _place_exit_orders_for_fills() -> list[dict]:
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
                    log_order(ticker, "sell", "limit", pos["qty"], "exit_target", t_oid, "pending")
                if s_oid:
                    log_order(ticker, "sell", "stop", pos["qty"], "exit_stop", s_oid, "pending")
                logger.info("Exit orders placed for %s: target=%s stop=%s", ticker, t_oid, s_oid)
                orders_placed.append({
                    "ticker": ticker,
                    "target_order_id": t_oid,
                    "stop_order_id": s_oid,
                })
    if orders_placed:
        await async_save_positions(fresh_state)
    return orders_placed


async def run_housekeeping(positions: list[dict]) -> HousekeepingResult:
    """Sync positions, check pending fills, place exit orders for new fills."""
    fills_confirmed: list[FillNotification] = []
    orders_placed: list[dict] = []

    if alpaca_enabled():
        try:
            await sync_positions_from_alpaca()
        except Exception:
            logger.exception("Housekeeping: Alpaca sync failed")

        fills_confirmed = await _confirm_pending_fills(positions)
        orders_placed = await _place_exit_orders_for_fills()

    return HousekeepingResult(
        fills_confirmed=fills_confirmed,
        orders_placed=orders_placed,
        positions_synced=alpaca_enabled(),
    )
