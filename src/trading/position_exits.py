"""Position exit/exposure operations — intraday close, fill tracking, exposure."""

import logging
from datetime import datetime

from config import PARAMS, TICKER_TO_GROUP
from db.positions import (
    delete_position_meta,
    load_all_position_meta,
    load_position_meta,
    record_closed_position,
    save_position_meta,
)
from trading.position_sync import _positions_lock

logger = logging.getLogger(__name__)


def close_position_intraday(ticker: str, price: float, reason: str) -> dict | None:
    """Close a single position during market hours. Returns closed position or None."""
    pos = load_position_meta(ticker)
    if pos is None:
        return None

    pos["exit_reason"] = reason
    pos["exit_price"] = price
    pos["exit_date"] = datetime.now().strftime("%Y-%m-%d")
    pos["realized_pnl_pct"] = round((price / pos["entry_price"] - 1) * 100, 2)
    pos["status"] = "closed"

    delete_position_meta(ticker)
    record_closed_position(pos)
    return pos


async def async_close_position(
    ticker: str, price: float, reason: str,
) -> dict | None:
    """Thread-safe close via asyncio.Lock."""
    async with _positions_lock:
        return close_position_intraday(ticker, price, reason)


async def update_fill_prices(fills: dict[str, float]) -> int:
    """Update positions that have pending order_ids with actual fill prices."""
    async with _positions_lock:
        positions = load_all_position_meta()
        updated = 0
        for pos in positions:
            oid = pos.get("order_id")
            if oid and oid in fills:
                fill_price = fills[oid]
                pos["entry_price"] = fill_price
                pos["order_id"] = None

                # Recalibrate trailing_stop and highest_close to fill price.
                # stop_price is left alone — it's set to range_low (ML support level) and is
                # still valid regardless of fill slippage.
                p = PARAMS
                atr_val = pos.get("atr", fill_price * p.default_atr_pct)
                pos["trailing_stop"] = round(fill_price - p.trailing_stop_atr_mult * atr_val, 4)
                pos["highest_close"] = fill_price

                save_position_meta(pos)
                updated += 1
        return updated


def get_trail_order_ids() -> dict[str, str]:
    """Return {ticker: trail_order_id} for positions with trailing stops."""
    positions = load_all_position_meta()
    return {
        pos["ticker"]: pos["trail_order_id"]
        for pos in positions
        if pos.get("trail_order_id")
    }


def close_position_by_trail_fill(ticker: str, fill_price: float) -> dict | None:
    """Close a position whose trailing stop filled at the broker."""
    return close_position_intraday(ticker, fill_price, "trail_stop_filled")


def get_total_exposure() -> float:
    """Return sum of size (fraction of equity) across all open positions."""
    return sum(p.get("size", 0) for p in load_all_position_meta())


def get_group_exposure(group: str) -> tuple[int, float]:
    """Return (count, total_exposure) for positions in the given asset group."""
    matched = [p for p in load_all_position_meta() if TICKER_TO_GROUP.get(p["ticker"]) == group]
    return len(matched), sum(p.get("size", 0) for p in matched)
