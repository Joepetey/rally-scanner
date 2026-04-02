"""Alpaca broker reconciliation — sync DB positions with broker state."""

import asyncio
import logging
from datetime import datetime

from rally_ml.config import PARAMS

from db.positions import (
    delete_position_meta,
    load_all_position_meta,
    load_positions,
    record_closed_position,
    save_position_meta,
)
from integrations.alpaca.broker import get_all_positions

logger = logging.getLogger(__name__)

# Async lock for concurrent position writes (scan + price alerts)
_positions_lock = asyncio.Lock()


def build_untracked_position(
    ticker: str, broker_pos: dict, equity: float,
) -> dict:
    """Pure builder: create a position dict from broker data + PARAMS-derived stops."""
    p = PARAMS
    entry = broker_pos["avg_entry_price"]
    atr_val = round(entry * p.default_atr_pct, 4)
    size = round(broker_pos["market_value"] / equity, 4) if equity > 0 else 0.0
    current = (
        round(broker_pos["market_value"] / broker_pos["qty"], 4)
        if broker_pos["qty"] else entry
    )
    pnl = round((current / entry - 1) * 100, 2) if entry else 0.0
    return {
        "ticker": ticker,
        "qty": broker_pos["qty"],
        "entry_price": entry,
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "stop_price": round(entry * (1 - p.fallback_stop_pct), 4),
        "target_price": round(entry + p.profit_atr_mult * atr_val, 4),
        "trailing_stop": round(entry - p.trailing_stop_atr_mult * atr_val, 4),
        "highest_close": max(entry, current),
        "atr": atr_val,
        "bars_held": 0,
        "size": size,
        "current_price": current,
        "unrealized_pnl_pct": pnl,
        "p_rally": 0.0,
        "status": "open",
    }


def compute_sync_updates(
    pos: dict, broker_pos: dict, equity: float,
) -> dict:
    """Pure builder: compute field updates for a matched position from broker data."""
    current = (
        round(broker_pos["market_value"] / broker_pos["qty"], 4)
        if broker_pos["qty"] else pos["entry_price"]
    )
    entry = pos["entry_price"]
    updates = {
        "qty": broker_pos["qty"],
        "entry_price": broker_pos["avg_entry_price"],
        "current_price": current,
        "unrealized_pnl_pct": round(
            (current / entry - 1) * 100, 2
        ) if entry else 0.0,
    }
    if equity > 0:
        updates["size"] = round(broker_pos["market_value"] / equity, 4)
    return updates


async def _sync_close_broker_exits(
    closed_tickers: set[str], meta_map: dict,
) -> None:
    """Case 3: in DB but gone from Alpaca — broker closed the position."""
    from integrations.alpaca.fills import get_recent_sell_fills

    fills = await get_recent_sell_fills(list(closed_tickers)) if closed_tickers else {}
    for ticker in closed_tickers:
        pos = meta_map[ticker]
        fill_price = fills.get(ticker, 0.0)
        pnl = (
            round((fill_price / pos["entry_price"] - 1) * 100, 2)
            if fill_price and pos.get("entry_price") else 0.0
        )
        pos["exit_price"] = fill_price
        pos["exit_date"] = datetime.now().strftime("%Y-%m-%d")
        pos["exit_reason"] = "broker_closed"
        pos["realized_pnl_pct"] = pnl
        pos["status"] = "closed"
        delete_position_meta(ticker)
        record_closed_position(pos)
        logger.info("sync: closed %s (broker_closed), fill=%.4f", ticker, fill_price)


def _sync_insert_untracked(
    inserted_tickers: set[str], broker_map: dict, equity: float,
) -> None:
    """Case 2: in Alpaca but not in DB — insert with PARAMS-derived stops."""
    for ticker in inserted_tickers:
        new_pos = build_untracked_position(ticker, broker_map[ticker], equity)
        save_position_meta(new_pos)
        logger.warning(
            "sync: inserted untracked position %s (size=%.1f%%)",
            ticker, new_pos["size"] * 100,
        )


def _sync_update_matched(
    matched_tickers: set[str], broker_map: dict, meta_map: dict, equity: float,
) -> None:
    """Case 1: in both — update qty/entry/current_price from broker."""
    for ticker in matched_tickers:
        pos = meta_map[ticker]
        updates = compute_sync_updates(pos, broker_map[ticker], equity)
        pos.update(updates)
        save_position_meta(pos)


async def sync_positions_from_alpaca(equity: float = 0.0) -> dict:
    """Reconcile DB with Alpaca broker state.

    Three cases:
    1. In Alpaca + in DB  → update qty/entry_price/current_price from broker
    2. In Alpaca, not DB  → insert with broker values and PARAMS-derived stops
    3. In DB, not Alpaca  → broker closed it; recover fill price, delete + record closed
    """
    async with _positions_lock:
        broker_list = await get_all_positions()
        broker_map = {p["ticker"]: p for p in broker_list}
        meta_list = load_all_position_meta()
        meta_map = {p["ticker"]: p for p in meta_list}

        broker_tickers = set(broker_map)
        db_tickers = set(meta_map)

        closed_tickers = db_tickers - broker_tickers
        inserted_tickers = broker_tickers - db_tickers
        matched_tickers = broker_tickers & db_tickers

        await _sync_close_broker_exits(closed_tickers, meta_map)
        _sync_insert_untracked(inserted_tickers, broker_map, equity)
        _sync_update_matched(matched_tickers, broker_map, meta_map, equity)

        return {
            "synced": len(matched_tickers),
            "closed": len(closed_tickers),
            "inserted": len(inserted_tickers),
        }


async def get_merged_positions(equity: float = 0.0) -> dict:
    """Sync with Alpaca broker state then return DB positions."""
    await sync_positions_from_alpaca(equity=equity)
    return load_positions()


def get_merged_positions_sync() -> dict:
    """DB-only read; sync is handled proactively by the scheduler."""
    return load_positions()
