"""Timer-based loops: scan scheduling, regime checks, weekly retrain."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from rally_ml.config import PARAMS
from rally_ml.core.persistence import load_manifest

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

_ET = __import__("zoneinfo").ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


async def regime_loop(sched: TradingScheduler) -> None:
    """30-minute timer: HMM regime check."""
    while True:
        await asyncio.sleep(1800)
        try:
            await sched.run_regime_check()
        except Exception:
            logger.exception("Regime loop error")


async def scan_loop(sched: TradingScheduler) -> None:
    """Time-based scan loop.

    Weekdays: pre-market scan (9:00 AM), market-open execute (9:30 AM),
              midday watchlist scans (11 AM, 1 PM) ET.
    Weekends: crypto-only scan every 6 hours (0, 6, 12, 18 ET).
    """
    while True:
        await asyncio.sleep(30)
        now = datetime.now(_ET)
        today = now.date().isoformat()

        if now.weekday() >= 5:
            # Weekend: crypto-only scan every 6 hours
            try:
                await maybe_run_weekend_crypto_scan(sched, now, today)
            except Exception:
                logger.exception("Weekend crypto scan error")
            continue

        # Weekday scans
        try:
            # 9:00 AM ET — pre-market scan (full universe)
            if PARAMS.morning_scan_enabled and now.hour == 9 and now.minute < 5:
                if (
                    sched.state.ran_morning_scan != today
                    and not sched.state.scan_in_progress.get("morning")
                ):
                    sched.state.ran_morning_scan = today
                    sched.state.scan_in_progress["morning"] = True
                    try:
                        async with asyncio.timeout(600):
                            await sched.run_premarket_scan()
                    finally:
                        sched.state.scan_in_progress["morning"] = False
            # 9:30 AM ET — execute entries/exits at market open
            elif PARAMS.morning_scan_enabled and now.hour == 9 and 30 <= now.minute < 35:
                if (
                    sched.state.ran_morning_execute != today
                    and not sched.state.scan_in_progress.get("execute")
                ):
                    sched.state.ran_morning_execute = today
                    sched.state.scan_in_progress["execute"] = True
                    try:
                        async with asyncio.timeout(300):
                            await sched.run_market_open_execute()
                    finally:
                        sched.state.scan_in_progress["execute"] = False
            elif PARAMS.midday_scans_enabled and now.hour == 11 and now.minute < 5:
                if (
                    sched.state.ran_midday_1 != today
                    and not sched.state.scan_in_progress.get("midday_1")
                ):
                    sched.state.ran_midday_1 = today
                    sched.state.scan_in_progress["midday_1"] = True
                    try:
                        async with asyncio.timeout(600):
                            await sched.run_midday_scan()
                    finally:
                        sched.state.scan_in_progress["midday_1"] = False
            elif PARAMS.midday_scans_enabled and now.hour == 13 and now.minute < 5:
                if (
                    sched.state.ran_midday_2 != today
                    and not sched.state.scan_in_progress.get("midday_2")
                ):
                    sched.state.ran_midday_2 = today
                    sched.state.scan_in_progress["midday_2"] = True
                    try:
                        async with asyncio.timeout(600):
                            await sched.run_midday_scan()
                    finally:
                        sched.state.scan_in_progress["midday_2"] = False
        except Exception:
            logger.exception("Scan loop error")


async def maybe_run_weekend_crypto_scan(
    sched: TradingScheduler, now: datetime, today: str,
) -> None:
    """Run a crypto-only scan every 6 hours on weekends."""
    from rally_ml.config import ASSETS

    # Fire at hours 0, 6, 12, 18 ET within a 5-minute window
    if now.hour % 6 != 0 or now.minute >= 5:
        return

    run_key = f"{today}_{now.hour}"
    if sched.state.ran_weekend_scan == run_key:
        return

    crypto_tickers = [t for t, cfg in ASSETS.items() if cfg.asset_class == "crypto"]
    if not crypto_tickers:
        return

    # Only run if at least one crypto ticker has a trained model
    manifest = load_manifest()
    if not manifest or not any(t in manifest for t in crypto_tickers):
        logger.debug("Weekend crypto scan: no trained crypto models found, skipping")
        return

    if sched.state.scan_in_progress.get("weekend_crypto"):
        return
    sched.state.ran_weekend_scan = run_key
    logger.info(
        "Weekend crypto scan: %s (tickers=%s)",
        run_key, sorted(crypto_tickers),
    )
    sched.state.scan_in_progress["weekend_crypto"] = True
    try:
        async with asyncio.timeout(600):
            await sched.run_daily_scan("weekend_crypto", tickers=crypto_tickers)
    finally:
        sched.state.scan_in_progress["weekend_crypto"] = False


async def retrain_loop(sched: TradingScheduler) -> None:
    """Time-based: weekly retrain Sunday >= 6 PM ET."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now(_ET)
        if now.weekday() != 6:  # Sunday only
            continue
        today = now.date().isoformat()
        if now.hour >= 18 and sched.state.ran_retrain != today:
            try:
                sched.state.ran_retrain = today
                await sched.run_retrain()
            except Exception:
                logger.exception("Retrain loop error")
