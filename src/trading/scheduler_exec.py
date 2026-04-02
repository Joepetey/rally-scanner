"""Execution helpers for TradingScheduler.

Broker entry/exit execution, order ID persistence, and watchlist saving.
Extracted from scheduler.py.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date as _date
from typing import TYPE_CHECKING

from db.events import log_order
from db.portfolio import record_closed_trades
from db.positions import (
    delete_position_meta as _del_meta,
)
from db.positions import load_positions
from db.positions import (
    record_closed_position as _rec_closed,
)
from db.positions import (
    save_watchlist as _db_save_watchlist,
)
from integrations.alpaca.broker import get_all_positions
from integrations.alpaca.executor import (
    execute_entries,
    execute_exits,
)
from trading.positions import (
    add_signal_positions,
    async_save_positions,
    process_signal_queue,
)

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

logger = logging.getLogger(__name__)


async def execute_and_log_entries(
    sched: TradingScheduler,
    signals: list[dict],
    equity: float,
    orders: list,
    *,
    update_stream: bool = True,
) -> None:
    """Execute entry orders, log results, persist positions + order IDs."""
    entry_results = await execute_entries(signals, equity=equity)
    for r in entry_results:
        log_order(r.ticker, "buy", "market", r.qty, "entry",
                  r.order_id, "filled" if r.success else "failed",
                  r.fill_price, r.error)
    ok = [r for r in entry_results if r.success]
    if ok:
        size_map = {
            r.ticker: r.actual_size for r in ok if r.actual_size is not None
        }
        filled_tickers = {r.ticker for r in ok}
        filled_signals = [
            {**s, "size": size_map.get(s["ticker"], s["size"])}
            for s in signals if s["ticker"] in filled_tickers
        ]

        fresh_positions = load_positions()
        add_signal_positions(fresh_positions, filled_signals)

        await store_order_ids(entry_results)
        if update_stream and sched._stream:
            all_pos = load_positions().get("positions", [])
            sched._stream.update_subscriptions({p["ticker"] for p in all_pos})
    orders.extend([r.model_dump() for r in entry_results])


async def execute_and_log_exits(
    sched: TradingScheduler,
    closed: list[dict],
    orders: list,
) -> list[dict]:
    """Execute exit orders, confirm with broker, clean up DB records.

    Returns the list of confirmed exit positions.
    """
    exit_results = await execute_exits(closed)
    for r in exit_results:
        log_order(r.ticker, "sell", "market", r.qty, "exit_scan",
                  r.order_id, "filled" if r.success else "failed",
                  r.fill_price, r.error)
    confirmed_exits: list[dict] = []
    ok_exit = [r for r in exit_results if r.success]
    if ok_exit:
        ok_tickers = {r.ticker for r in ok_exit}
        broker_positions = await get_all_positions()
        still_open = {p["ticker"] for p in broker_positions}
        ghost_tickers = ok_tickers & still_open
        if ghost_tickers:
            logger.warning(
                "Exit API success but position still open at broker "
                "— skipping DB delete to avoid ghost position: %s",
                ghost_tickers,
            )
        confirmed_exits = [
            p for p in closed
            if p["ticker"] in ok_tickers and p["ticker"] not in still_open
        ]
        for pos in confirmed_exits:
            _del_meta(pos["ticker"])
            _rec_closed(pos)
        record_closed_trades(confirmed_exits)

    orders.extend([r.model_dump() for r in exit_results])
    return confirmed_exits


async def store_order_ids(results: list) -> None:
    """Persist Alpaca order IDs to position records after entry fills."""
    state = load_positions()
    pos_by_ticker = {pos["ticker"]: pos for pos in state["positions"]}
    for result in results:
        if result.success and result.order_id:
            pos = pos_by_ticker.get(result.ticker)
            if pos:
                if not pos.get("order_id"):
                    pos["order_id"] = result.order_id
                if result.qty:
                    pos["qty"] = result.qty
                if result.trail_order_id:
                    pos["trail_order_id"] = result.trail_order_id
    await async_save_positions(state)


async def retry_queued_signals(
    sched: TradingScheduler, equity: float, orders: list,
) -> None:
    """Re-attempt queued signals freed by today's closes."""
    queued = await asyncio.to_thread(process_signal_queue)
    if queued:
        await execute_and_log_entries(
            sched, queued, equity, orders, update_stream=False,
        )


def save_watchlist(results: list, positions: dict) -> None:
    """Persist scan results as watchlist for agent queries."""
    open_tickers = {p["ticker"] for p in positions.get("positions", [])}
    watchlist = []
    for r in sorted(results, key=lambda x: x.get("p_rally", 0), reverse=True):
        if r.get("status") != "ok":
            continue
        if r["ticker"] in open_tickers:
            continue
        watchlist.append({
            "ticker": r["ticker"],
            "p_rally": round(r.get("p_rally", 0) * 100, 1),
            "p_rally_raw": r.get("p_rally_raw", 0),
            "comp_score": r.get("comp_score", 0),
            "fail_dn": r.get("fail_dn", 0),
            "trend": r.get("trend", 0),
            "golden_cross": r.get("golden_cross", 0),
            "hmm_compressed": r.get("hmm_compressed", 0),
            "rv_pctile": r.get("rv_pctile", 0),
            "atr_pct": r.get("atr_pct", 0),
            "macd_hist": r.get("macd_hist", 0),
            "vol_ratio": r.get("vol_ratio", 1),
            "vix_pctile": r.get("vix_pctile", 0),
            "rsi": r.get("rsi", 0),
            "close": r.get("close", 0),
            "size": r.get("size", 0),
            "signal": bool(r.get("signal")),
        })
    _db_save_watchlist(watchlist, scan_date=_date.today())
