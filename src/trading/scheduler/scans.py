"""Scan operations for TradingScheduler.

Daily, pre-market, market-open, and midday scan entry points.
Shared pipeline logic lives in scheduler_prepare.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from datetime import date as _date
from typing import TYPE_CHECKING

from rally_ml.config import PARAMS

from db.ops.events import finish_scheduler_event, log_scheduler_event
from db.trading.positions import load_current_signals, load_positions
from integrations.alpaca.account import get_account_equity
from integrations.alpaca.broker import is_enabled as alpaca_enabled
from pipeline.scanner import scan_watchlist
from trading.events import ScanResult, WatchlistEvent
from trading.positions import sync_positions_from_alpaca, update_skipped_outcomes
from trading.scheduler.exec import (
    execute_and_log_entries,
    execute_and_log_exits,
    retry_queued_signals,
)
from trading.scheduler.prepare import run_scan_core

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

logger = logging.getLogger(__name__)


async def run_daily_scan(
    sched: TradingScheduler,
    scan_type: str = "daily",
    tickers: list[str] | None = None,
) -> None:
    """Run the full scan pipeline and emit a ScanResult event."""
    from trading.scheduler.actions import run_risk_evaluation

    core = await run_scan_core(sched, scan_type, tickers)
    if not core:
        return

    orders: list[dict] = []
    confirmed_exits: list[dict] = list(core.closed) if not alpaca_enabled() else []
    scan_equity: float = 0.0

    if alpaca_enabled():
        try:
            equity = await get_account_equity()
            scan_equity = equity
            if core.signals:
                await execute_and_log_entries(sched, core.signals, equity, orders)
            if core.closed:
                confirmed_exits = await execute_and_log_exits(sched, core.closed, orders)
            await retry_queued_signals(sched, equity, orders)
        except Exception:
            logger.exception("Alpaca execution failed during %s scan", scan_type)

    await asyncio.to_thread(update_skipped_outcomes, core.results)

    duration = round(_time.time() - core.t0, 1)
    finish_scheduler_event(core.event_id, "success", n_signals=len(core.signals),
                           n_exits=len(confirmed_exits), duration_s=duration)
    logger.info(
        "Scan complete — %d signals, %d exits (%.0fs)",
        len(core.signals), len(confirmed_exits), duration,
    )

    await sched._on_event(ScanResult(
        signals=core.signals,
        exits=confirmed_exits,
        orders=orders,
        positions_summary={"positions": core.positions_for_embed},
        scan_type=scan_type,
        equity=scan_equity,
    ))

    await run_risk_evaluation(sched)


async def run_premarket_scan(sched: TradingScheduler) -> None:
    """Run full-universe scan at 9:00 AM ET and store signals for market-open execution."""
    core = await run_scan_core(sched, "premarket")
    if not core:
        return

    # Store for market-open execution
    sched.state.pending_signals = core.signals
    sched.state.pending_exits = list(core.closed) if alpaca_enabled() else []
    sched.state.pending_scan_results = core.results
    sched.state.pending_positions_embed = core.positions_for_embed

    duration = round(_time.time() - core.t0, 1)
    confirmed_exits = list(core.closed) if not alpaca_enabled() else []
    finish_scheduler_event(core.event_id, "success", n_signals=len(core.signals),
                           n_exits=len(confirmed_exits), duration_s=duration)
    logger.info(
        "Pre-market scan complete — %d signals, %d pending exits (%.0fs)",
        len(core.signals), len(sched.state.pending_exits), duration,
    )

    await sched._on_event(ScanResult(
        signals=core.signals,
        exits=confirmed_exits,
        orders=[],
        positions_summary={"positions": core.positions_for_embed},
        scan_type="premarket",
    ))


async def run_market_open_execute(sched: TradingScheduler) -> None:
    """Execute pending signals and exits at market open (9:30 AM ET)."""
    from trading.scheduler.actions import run_risk_evaluation

    # Restart fallback: reload from DB if in-memory state was lost
    if not sched.state.pending_signals and not sched.state.pending_exits:
        if sched.state.ran_morning_scan == _date.today().isoformat():
            db_signals = load_current_signals()
            if db_signals:
                logger.info(
                    "Restart fallback: loaded %d signals from current_signals table",
                    len(db_signals),
                )
                sched.state.pending_signals = db_signals

    if not sched.state.pending_signals and not sched.state.pending_exits:
        logger.info("Market-open execute: nothing pending, skipping")
        return

    logger.info(
        "Scheduler: executing %d entries + %d exits at market open",
        len(sched.state.pending_signals), len(sched.state.pending_exits),
    )
    _event_id = log_scheduler_event("execute")
    t0 = _time.time()

    signals = sched.state.pending_signals
    closed = sched.state.pending_exits
    results = sched.state.pending_scan_results
    positions_for_embed = sched.state.pending_positions_embed

    orders: list[dict] = []
    confirmed_exits: list[dict] = list(closed) if not alpaca_enabled() else []
    scan_equity: float = 0.0

    if alpaca_enabled():
        try:
            equity = await get_account_equity()
            scan_equity = equity
            await sync_positions_from_alpaca(equity=equity)

            open_tickers = {
                p["ticker"] for p in load_positions().get("positions", [])
            }
            signals = [s for s in signals if s["ticker"] not in open_tickers]

            if signals:
                await execute_and_log_entries(sched, signals, equity, orders)
            if closed:
                confirmed_exits = await execute_and_log_exits(sched, closed, orders)
            await retry_queued_signals(sched, equity, orders)
        except Exception:
            logger.exception("Alpaca execution failed during market-open execute")

    if results:
        await asyncio.to_thread(update_skipped_outcomes, results)

    duration = round(_time.time() - t0, 1)
    finish_scheduler_event(_event_id, "success", n_signals=len(signals),
                           n_exits=len(confirmed_exits), duration_s=duration)
    logger.info(
        "Market-open execute complete — %d signals, %d exits (%.0fs)",
        len(signals), len(confirmed_exits), duration,
    )

    await sched._on_event(ScanResult(
        signals=signals,
        exits=confirmed_exits,
        orders=orders,
        positions_summary={"positions": positions_for_embed},
        scan_type="morning",
        equity=scan_equity,
    ))

    # Clear pending state
    sched.state.pending_signals = []
    sched.state.pending_exits = []
    sched.state.pending_scan_results = []
    sched.state.pending_positions_embed = []

    await run_risk_evaluation(sched)


async def run_midday_scan(sched: TradingScheduler) -> None:
    """Run a lightweight scan on watchlist tickers only."""
    if not PARAMS.midday_scans_enabled or not sched._engine.is_market_open():
        return
    if not sched.state.watchlist_tickers:
        logger.debug("Mid-day scan: empty watchlist, skipping")
        return

    logger.info("Mid-day scan: checking %d watchlist tickers", len(sched.state.watchlist_tickers))
    results = await asyncio.to_thread(scan_watchlist, sched.state.watchlist_tickers)
    all_signals = [r for r in results if r.get("signal")]

    open_tickers = {p["ticker"] for p in load_positions().get("positions", [])}
    signals = [s for s in all_signals if s["ticker"] not in open_tickers]

    await sched._on_event(WatchlistEvent(signals=signals, scan_type="midday"))
