"""Shared scan pipeline: sync, exit detection, filtering, and core scan orchestration."""

from __future__ import annotations

import asyncio
import logging
import time as _time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rally_ml.config import PARAMS
from rally_ml.core.persistence import load_manifest

from db.ops.events import finish_scheduler_event, log_scheduler_event
from db.trading.portfolio import record_closed_trades, update_daily_snapshot
from db.trading.positions import get_recently_closed_tickers, load_positions
from db.trading.positions import save_latest_scan as _save_latest_scan
from integrations.alpaca.account import get_account_equity
from integrations.alpaca.broker import is_enabled as alpaca_enabled
from pipeline.scanner import scan_all
from trading.events import ScanResult
from trading.positions import sync_positions_from_alpaca, update_existing_positions
from trading.scheduler.exec import save_watchlist

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

logger = logging.getLogger(__name__)


@dataclass
class ScanCore:
    """Intermediate result from the shared scan pipeline."""

    event_id: int
    t0: float
    results: list[dict]
    signals: list[dict]
    positions: dict
    closed: list[dict]
    positions_for_embed: list[dict]


async def sync_and_detect_exits(
    results: list[dict],
) -> tuple[dict, list[dict]]:
    """Alpaca sync + exit detection. Returns (positions, closed)."""
    if alpaca_enabled():
        try:
            _equity = await get_account_equity()
            await sync_positions_from_alpaca(equity=_equity)
        except Exception:
            logger.exception("Alpaca sync failed — using cached positions")
    positions = load_positions()

    positions = await asyncio.to_thread(
        update_existing_positions, positions, results, not alpaca_enabled(),
    )
    closed = positions.get("closed_today", [])

    if closed and not alpaca_enabled():
        record_closed_trades(closed)

    update_daily_snapshot(positions, results)
    return positions, closed


async def persist_and_filter(
    sched: TradingScheduler,
    results: list[dict],
    positions: dict,
) -> tuple[list[dict], list[dict]]:
    """Save watchlist/scan, refresh manifest, filter signals by cooldown.

    Returns (signals, positions_for_embed).
    """
    save_watchlist(results, positions)
    await asyncio.to_thread(_save_latest_scan, results, positions)

    manifest = load_manifest()
    if manifest:
        sched.state.watchlist_tickers = sorted(manifest.keys())

    all_signals = [r for r in results if r.get("signal")]
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
    return signals, positions_for_embed


async def run_scan_core(
    sched: TradingScheduler,
    scan_type: str,
    tickers: list[str] | None = None,
) -> ScanCore | None:
    """Shared scan pipeline: manifest check -> scan -> sync -> filter.

    Returns None if there's nothing to process (no manifest or no results).
    """
    logger.info("Scheduler: starting %s scan", scan_type)
    event_id = log_scheduler_event("scan")
    t0 = _time.time()

    manifest = load_manifest()
    if not manifest:
        finish_scheduler_event(event_id, "error", n_signals=0, n_exits=0, duration_s=0)
        await sched._on_event(ScanResult(
            signals=[], exits=[], orders=[],
            positions_summary={},
            scan_type=scan_type,
            error="No trained models found",
        ))
        return None

    results = await asyncio.to_thread(scan_all, tickers, "conservative")
    if not results:
        finish_scheduler_event(event_id, "success", n_signals=0, n_exits=0,
                               duration_s=round(_time.time() - t0, 1))
        return None

    positions, closed = await sync_and_detect_exits(results)
    signals, positions_for_embed = await persist_and_filter(sched, results, positions)

    return ScanCore(event_id, t0, results, signals, positions, closed, positions_for_embed)
