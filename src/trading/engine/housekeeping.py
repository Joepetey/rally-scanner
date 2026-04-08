"""Housekeeping — Alpaca sync, fill confirmation, OCO exit order placement.

Extracted from engine.py to keep AlertEngine focused on price evaluation.
"""

import logging

import rally_ml.config as config

from db.ops.events import log_order, update_order_fill
from db.trading.positions import load_positions, save_position_meta
from integrations.alpaca.broker import is_enabled as alpaca_enabled
from integrations.alpaca.exits import (
    cancel_exit_orders,
    execute_exit,
    place_exit_orders,
    place_trailing_stop,
)
from integrations.alpaca.fills import (
    check_pending_fills,
    get_recent_sell_fills,
)
from trading.events import ExitResult, FillNotification, HousekeepingResult, LetItRideEvent
from trading.positions import (
    async_close_position,
    async_save_positions,
    sync_positions_from_alpaca,
    update_fill_prices,
)

logger = logging.getLogger(__name__)


async def execute_breach(
    ticker: str, pos: dict, price: float, reason: str,
) -> ExitResult | None:
    """Cancel OCO exit orders then execute a market sell."""
    await cancel_exit_orders(pos.get("target_order_id"), pos.get("trail_order_id"))
    try:
        result = await execute_exit(ticker)
        if result.already_closed:
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


async def _check_broker_position(ticker: str) -> float | None:
    """Return broker qty for ticker, or None if no position held."""
    from integrations.alpaca.broker import get_all_positions
    broker_positions = await get_all_positions()
    for bp in broker_positions:
        if bp["ticker"] == ticker:
            return bp["qty"]
    return None


async def convert_to_trailing_stop(
    ticker: str, pos: dict, price: float,
) -> LetItRideEvent | ExitResult | None:
    """Cancel OCO, place Alpaca trailing stop, let position ride.

    If the OCO target already filled at the broker (no position held),
    closes the position in DB using the broker fill price and returns
    an ExitResult instead.
    """
    from rally_ml.config import PARAMS

    is_crypto = (
        ticker in config.ASSETS
        and config.ASSETS[ticker].asset_class == "crypto"
    )
    if is_crypto:
        logger.info("Skipping let-it-ride for crypto %s — selling at target", ticker)
        return None

    await cancel_exit_orders(pos.get("target_order_id"), pos.get("trail_order_id"))

    # Guard: verify broker still holds the position. The OCO target may have
    # already filled before our polling detected the breach.
    broker_qty = await _check_broker_position(ticker)
    if not broker_qty:
        logger.warning(
            "Let-it-ride aborted for %s — no broker position "
            "(OCO target likely already filled)", ticker,
        )
        from integrations.alpaca.fills import get_recent_sell_fills
        fills = await get_recent_sell_fills([ticker])
        fill_price = fills.get(ticker, price)
        closed = await async_close_position(ticker, fill_price, "oco_fill")
        if closed:
            return ExitResult(
                ticker=ticker,
                exit_reason="oco_fill",
                fill_price=fill_price,
                order_id=pos.get("target_order_id"),
                realized_pnl_pct=closed.get("realized_pnl_pct"),
                bars_held=closed.get("bars_held"),
            )
        return None

    trail_pct = PARAMS.let_it_ride_trail_pct
    qty = pos.get("qty", 0)
    trail_order_id = await place_trailing_stop(ticker, qty, trail_pct) if qty else None

    entry = pos["entry_price"]
    floor = round(entry * (1 + PARAMS.profit_lock_floor_pct), 4)
    if pos.get("stop_price", 0) < floor:
        pos["stop_price"] = floor

    original_target = pos.get("target_price", 0)
    pos["let_it_ride"] = True
    pos["target_order_id"] = None
    pos["trail_order_id"] = trail_order_id

    save_position_meta(pos)

    pnl_pct = round((price / entry - 1) * 100, 2) if entry else 0

    log_order(
        ticker, "sell", "trailing_stop", qty,
        "let_it_ride", trail_order_id,
        "pending" if trail_order_id else "failed",
        None, None,
    )

    logger.info(
        "Let-it-ride: %s target $%.2f hit at $%.2f (+%.1f%%), "
        "trailing stop %.1f%% placed (order %s)",
        ticker, original_target, price, pnl_pct, trail_pct, trail_order_id,
    )

    return LetItRideEvent(
        ticker=ticker,
        entry_price=entry,
        target_price=original_target,
        current_price=price,
        trail_pct=trail_pct,
        trail_order_id=trail_order_id,
        pnl_pct=pnl_pct,
    )


async def confirm_pending_fills(
    positions: list[dict],
) -> list[FillNotification]:
    """Check pending entry orders and update fill prices."""
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


async def place_exit_orders_for_fills() -> list[dict]:
    """Place OCO exit orders for newly filled entries."""
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


async def run_housekeeping(positions: list[dict]) -> HousekeepingResult:
    """Sync positions, check pending fills, place exit orders for new fills."""
    fills_confirmed: list[FillNotification] = []
    orders_placed: list[dict] = []

    if alpaca_enabled():
        try:
            await sync_positions_from_alpaca()
        except Exception:
            logger.exception("Housekeeping: Alpaca sync failed")

        fills_confirmed = await confirm_pending_fills(positions)
        orders_placed = await place_exit_orders_for_fills()

    return HousekeepingResult(
        fills_confirmed=fills_confirmed,
        orders_placed=orders_placed,
        positions_synced=alpaca_enabled(),
    )
