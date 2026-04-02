"""Scan operations and public actions for TradingScheduler.

Extracted from scheduler.py to keep the coordinator focused on
lifecycle and wiring.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from datetime import date as _date
from typing import TYPE_CHECKING

from rally_ml.config import PARAMS
from rally_ml.core.persistence import load_manifest
from rally_ml.pipeline.retrain import retrain_all

from db.events import finish_scheduler_event, log_scheduler_event
from db.portfolio import record_closed_trades, update_daily_snapshot
from db.positions import (
    get_recently_closed_tickers,
    load_current_signals,
    load_positions,
)
from db.positions import (
    save_latest_scan as _save_latest_scan,
)
from integrations.alpaca.executor import (
    get_account_equity,
)
from integrations.alpaca.executor import (
    is_enabled as alpaca_enabled,
)
from pipeline.scanner import scan_all, scan_watchlist
from trading.events import (
    RetrainResult,
    RiskActionEvent,
    ScanResult,
    WatchlistEvent,
)
from trading.positions import (
    sync_positions_from_alpaca,
    update_existing_positions,
    update_skipped_outcomes,
)
from trading.regime_monitor import check_regime_shifts, get_regime_states, is_cascade
from trading.risk_manager import evaluate, execute_actions
from trading.scheduler_exec import (
    execute_and_log_entries,
    execute_and_log_exits,
    retry_queued_signals,
    save_watchlist,
)

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared scan post-processing
# ------------------------------------------------------------------

async def prepare_scan_results(
    sched: TradingScheduler, results: list[dict],
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """Shared scan post-processing: sync, detect exits, filter signals.

    Returns (signals, positions, closed, positions_for_embed).
    """
    all_signals = [r for r in results if r.get("signal")]

    if alpaca_enabled():
        try:
            _equity = await get_account_equity()
            await sync_positions_from_alpaca(equity=_equity)
        except Exception:
            logger.exception("Alpaca sync failed — using cached positions")
    positions = load_positions()

    # Detect exits; defer DB commit if Alpaca will confirm them
    positions = await asyncio.to_thread(
        update_existing_positions, positions, results, not alpaca_enabled(),
    )
    closed = positions.get("closed_today", [])

    if closed and not alpaca_enabled():
        record_closed_trades(closed)

    update_daily_snapshot(positions, results)
    save_watchlist(results, positions)
    await asyncio.to_thread(_save_latest_scan, results, positions)

    # Refresh watchlist for mid-day scans
    manifest = load_manifest()
    if manifest:
        sched._watchlist_tickers = sorted(manifest.keys())

    open_tickers = {p["ticker"] for p in positions.get("positions", [])}
    cooldown_tickers: set[str] = set()
    if PARAMS.cooldown_days > 0:
        cooldown_tickers = get_recently_closed_tickers(PARAMS.cooldown_days)
        cooled = [s["ticker"] for s in all_signals
                  if s["ticker"] not in open_tickers and s["ticker"] in cooldown_tickers]
        if cooled:
            logger.info("Cooldown filter (%dd): skipping %s", PARAMS.cooldown_days, cooled)
    signals = [s for s in all_signals
               if s["ticker"] not in open_tickers and s["ticker"] not in cooldown_tickers]

    positions_for_embed = list(positions.get("positions", []))
    return signals, positions, closed, positions_for_embed


# ------------------------------------------------------------------
# Scan operations
# ------------------------------------------------------------------

async def run_daily_scan(
    sched: TradingScheduler,
    scan_type: str = "daily",
    tickers: list[str] | None = None,
) -> None:
    """Run the full scan pipeline and emit a ScanResult event."""
    logger.info("Scheduler: starting %s scan", scan_type)
    _event_id = log_scheduler_event("scan")
    t0 = _time.time()

    manifest = load_manifest()
    if not manifest:
        finish_scheduler_event(_event_id, "error", n_signals=0, n_exits=0, duration_s=0)
        await sched._on_event(ScanResult(
            signals=[], exits=[], orders=[],
            positions_summary={},
            scan_type=scan_type,
            error="No trained models found",
        ))
        return

    results = await asyncio.to_thread(scan_all, tickers, "conservative")
    if not results:
        finish_scheduler_event(_event_id, "success", n_signals=0, n_exits=0,
                               duration_s=round(_time.time() - t0, 1))
        return

    signals, positions, closed, positions_for_embed = await prepare_scan_results(
        sched, results,
    )

    orders: list[dict] = []
    confirmed_exits: list[dict] = list(closed) if not alpaca_enabled() else []
    scan_equity: float = 0.0

    if alpaca_enabled():
        try:
            equity = await get_account_equity()
            scan_equity = equity

            if signals:
                await execute_and_log_entries(sched, signals, equity, orders)

            if closed:
                confirmed_exits = await execute_and_log_exits(sched, closed, orders)

            await retry_queued_signals(sched, equity, orders)

        except Exception:
            logger.exception("Alpaca execution failed during %s scan", scan_type)

    await asyncio.to_thread(update_skipped_outcomes, results)

    duration = round(_time.time() - t0, 1)
    finish_scheduler_event(_event_id, "success", n_signals=len(signals),
                           n_exits=len(confirmed_exits), duration_s=duration)
    logger.info(
        "Scan complete — %d signals, %d exits (%.0fs)",
        len(signals), len(confirmed_exits), duration,
    )

    await sched._on_event(ScanResult(
        signals=signals,
        exits=confirmed_exits,
        orders=orders,
        positions_summary={"positions": positions_for_embed},
        scan_type=scan_type,
        equity=scan_equity,
    ))

    # Post-scan risk evaluation picks up any regime changes
    await run_risk_evaluation(sched)


async def run_premarket_scan(sched: TradingScheduler) -> None:
    """Run full-universe scan at 9:00 AM ET and store signals for market-open execution."""
    logger.info("Scheduler: starting pre-market scan")
    _event_id = log_scheduler_event("scan")
    t0 = _time.time()

    manifest = load_manifest()
    if not manifest:
        finish_scheduler_event(_event_id, "error", n_signals=0, n_exits=0, duration_s=0)
        await sched._on_event(ScanResult(
            signals=[], exits=[], orders=[],
            positions_summary={},
            scan_type="premarket",
            error="No trained models found",
        ))
        return

    results = await asyncio.to_thread(scan_all, None, "conservative")
    if not results:
        finish_scheduler_event(_event_id, "success", n_signals=0, n_exits=0,
                               duration_s=round(_time.time() - t0, 1))
        return

    signals, positions, closed, positions_for_embed = await prepare_scan_results(
        sched, results,
    )

    # Store for market-open execution
    sched._pending_signals = signals
    sched._pending_exits = list(closed) if alpaca_enabled() else []
    sched._pending_scan_results = results
    sched._pending_positions_embed = positions_for_embed

    duration = round(_time.time() - t0, 1)
    confirmed_exits = list(closed) if not alpaca_enabled() else []
    finish_scheduler_event(_event_id, "success", n_signals=len(signals),
                           n_exits=len(confirmed_exits), duration_s=duration)
    logger.info(
        "Pre-market scan complete — %d signals, %d pending exits (%.0fs)",
        len(signals), len(sched._pending_exits), duration,
    )

    await sched._on_event(ScanResult(
        signals=signals,
        exits=confirmed_exits,
        orders=[],
        positions_summary={"positions": positions_for_embed},
        scan_type="premarket",
    ))


async def run_market_open_execute(sched: TradingScheduler) -> None:
    """Execute pending signals and exits at market open (9:30 AM ET)."""
    # Restart fallback: reload from DB if in-memory state was lost
    if not sched._pending_signals and not sched._pending_exits:
        if sched._ran_morning_scan == _date.today().isoformat():
            db_signals = load_current_signals()
            if db_signals:
                logger.info(
                    "Restart fallback: loaded %d signals from current_signals table",
                    len(db_signals),
                )
                sched._pending_signals = db_signals

    if not sched._pending_signals and not sched._pending_exits:
        logger.info("Market-open execute: nothing pending, skipping")
        return

    logger.info(
        "Scheduler: executing %d entries + %d exits at market open",
        len(sched._pending_signals), len(sched._pending_exits),
    )
    _event_id = log_scheduler_event("execute")
    t0 = _time.time()

    signals = sched._pending_signals
    closed = sched._pending_exits
    results = sched._pending_scan_results
    positions_for_embed = sched._pending_positions_embed

    orders: list[dict] = []
    confirmed_exits: list[dict] = list(closed) if not alpaca_enabled() else []
    scan_equity: float = 0.0

    if alpaca_enabled():
        try:
            # Re-sync in case of pre-market changes
            equity = await get_account_equity()
            scan_equity = equity
            await sync_positions_from_alpaca(equity=equity)

            # Re-filter against current open positions (safety)
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
    sched._pending_signals = []
    sched._pending_exits = []
    sched._pending_scan_results = []
    sched._pending_positions_embed = []

    await run_risk_evaluation(sched)


async def run_midday_scan(sched: TradingScheduler) -> None:
    """Run a lightweight scan on watchlist tickers only."""
    if not PARAMS.midday_scans_enabled or not sched._engine.is_market_open():
        return
    if not sched._watchlist_tickers:
        logger.debug("Mid-day scan: empty watchlist, skipping")
        return

    logger.info("Mid-day scan: checking %d watchlist tickers", len(sched._watchlist_tickers))
    results = await asyncio.to_thread(scan_watchlist, sched._watchlist_tickers)
    all_signals = [r for r in results if r.get("signal")]

    open_tickers = {p["ticker"] for p in load_positions().get("positions", [])}
    signals = [s for s in all_signals if s["ticker"] not in open_tickers]

    await sched._on_event(WatchlistEvent(signals=signals, scan_type="midday"))


# ------------------------------------------------------------------
# Regime / retrain / risk
# ------------------------------------------------------------------

async def run_regime_check(sched: TradingScheduler) -> None:
    """Check HMM regime states and emit a RegimeEvent on transitions."""
    from trading.events import RegimeEvent

    if not PARAMS.regime_check_enabled or not sched._engine.is_market_open():
        return

    transitions = await asyncio.to_thread(check_regime_shifts)
    if not transitions:
        return

    sched._regime_states = get_regime_states()
    cascade = is_cascade(transitions)

    await sched._on_event(RegimeEvent(transitions=transitions, cascade_triggered=cascade))

    if cascade:
        logger.info(
            "Regime cascade detected (%d shifts) — triggering early scan",
            len(transitions),
        )
        await run_daily_scan(sched, "cascade")


async def run_retrain(sched: TradingScheduler) -> None:
    """Run model retraining and emit a RetrainResult event, then scan."""
    logger.info("Scheduler: starting weekly retrain")
    _event_id = log_scheduler_event("retrain")
    t0 = _time.time()
    await asyncio.to_thread(retrain_all)
    elapsed = _time.time() - t0

    manifest = load_manifest()
    finish_scheduler_event(_event_id, "success", duration_s=round(elapsed, 1))
    logger.info("Scheduler: retrain complete (%.0fs)", elapsed)

    await sched._on_event(RetrainResult(
        tickers_retrained=sorted(manifest.keys()) if manifest else [],
        duration_seconds=round(elapsed, 1),
        manifest_size=len(manifest) if manifest else 0,
    ))

    # Scan so Monday signals are ready before market open
    await run_daily_scan(sched, "post_retrain")


async def run_risk_evaluation(sched: TradingScheduler) -> None:
    """Run proactive risk evaluation on current positions."""
    if not PARAMS.proactive_risk_enabled:
        return

    state = load_positions()
    positions = state.get("positions", [])
    if not positions:
        return

    try:
        equity = await get_account_equity() if alpaca_enabled() else 100_000
    except Exception:
        logger.warning("Equity fetch failed, defaulting to $100k", exc_info=True)
        equity = 100_000

    actions = await asyncio.to_thread(evaluate, equity, positions, sched._regime_states)
    if not actions:
        return

    results = await execute_actions(actions, positions)
    meaningful = [r for r in results if not r.get("skipped")]
    if meaningful:
        await sched._on_event(RiskActionEvent(actions=meaningful))
        logger.info(
            "Risk evaluation: %d actions (%s)",
            len(meaningful),
            ", ".join(f"{r['ticker']}:{r['action']}" for r in meaningful),
        )
