"""Non-scan scheduler actions: regime checks, retraining, risk evaluation."""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import TYPE_CHECKING

from rally_ml.config import PARAMS
from rally_ml.core.persistence import load_manifest
from rally_ml.pipeline.retrain import retrain_all

from db.events import finish_scheduler_event, log_scheduler_event
from db.positions import load_positions
from integrations.alpaca.account import get_account_equity
from integrations.alpaca.broker import is_enabled as alpaca_enabled
from trading.events import RegimeEvent, RetrainResult, RiskActionEvent
from trading.regime_monitor import check_regime_shifts, get_regime_states, is_cascade
from trading.risk_manager import evaluate, execute_actions

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

logger = logging.getLogger(__name__)


async def run_regime_check(sched: TradingScheduler) -> None:
    """Check HMM regime states and emit a RegimeEvent on transitions."""
    if not PARAMS.regime_check_enabled or not sched._engine.is_market_open():
        return

    transitions = await asyncio.to_thread(check_regime_shifts)
    if not transitions:
        return

    sched.state.regime_states = get_regime_states()
    cascade = is_cascade(transitions)

    await sched._on_event(RegimeEvent(transitions=transitions, cascade_triggered=cascade))

    if cascade:
        logger.info(
            "Regime cascade detected (%d shifts) — triggering early scan",
            len(transitions),
        )
        from trading.scheduler_scans import run_daily_scan

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
    from trading.scheduler_scans import run_daily_scan

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

    actions = await asyncio.to_thread(evaluate, equity, positions, sched.state.regime_states)
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
