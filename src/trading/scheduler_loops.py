"""Background loops for TradingScheduler.

Each loop is a method on the scheduler class, extracted here to keep
scheduler.py focused on lifecycle and coordination.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from rally_ml.config import PARAMS
from rally_ml.core.data import fetch_quotes
from rally_ml.core.persistence import load_manifest

from db.portfolio import record_closed_trades
from db.positions import (
    load_position_meta as _load_position_meta,
)
from db.positions import load_positions
from integrations.alpaca.account import get_snapshots
from integrations.alpaca.fills import check_exit_fills
from integrations.alpaca.models import is_enabled as alpaca_enabled
from trading.events import (
    ExitResult,
    StreamDegradedEvent,
    StreamRecoveredEvent,
)
from trading.positions import (
    async_close_position,
)

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

_ET = __import__("zoneinfo").ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Housekeeping loop
# ------------------------------------------------------------------

async def housekeeping_loop(sched: TradingScheduler) -> None:
    """1-minute timer: fill checks, OCO placement, position sync."""
    while True:
        await asyncio.sleep(60)
        if not sched._engine.is_market_open() and not has_open_crypto_positions():
            continue
        try:
            sched._housekeeping_cycles += 1
            state = load_positions()
            positions = state.get("positions", [])
            result = await sched._engine.run_housekeeping(positions)

            if result.fills_confirmed or result.orders_placed:
                await sched._on_event(result)
                if sched._stream:
                    all_pos = load_positions().get("positions", [])
                    sched._stream.update_subscriptions({p["ticker"] for p in all_pos})

            await check_stale_tickers(sched)
            await check_stream_health(sched)

        except Exception:
            logger.exception("Housekeeping loop error")


async def check_stale_tickers(sched: TradingScheduler) -> None:
    """Every 2 cycles: warn about stale stream tickers, run IEX fallback."""
    if not (
        sched._stream and sched._stream.is_connected
        and sched._housekeeping_cycles % 2 == 0
    ):
        return

    new_stale, known_stale, never_traded = sched._stream.get_stale_tickers(
        stale_seconds=300.0,
    )
    stale = new_stale + known_stale + never_traded
    if new_stale:
        logger.warning(
            "No stream trade in 5 min for %d ticker(s): %s — "
            "possible subscription issue",
            len(new_stale), sorted(new_stale),
        )
    if never_traded:
        logger.info(
            "No stream trade yet for %d ticker(s): %s — "
            "likely low-volume",
            len(never_traded), sorted(never_traded),
        )
    if known_stale:
        logger.debug(
            "Still stale (low-volume expected): %s",
            sorted(known_stale),
        )
    # IEX fallback: evaluate stale tickers via REST snapshot
    if stale and alpaca_enabled():
        try:
            async with sched._snapshot_lock:
                snapshots = await get_snapshots(stale)
            fresh_state = load_positions()
            fresh_positions = fresh_state.get("positions", [])
            events = await sched._engine.check_prices(
                [p for p in fresh_positions if p["ticker"] in snapshots],
                snapshots,
            )
            for event in events:
                await sched._on_event(event)
                if event.alert_type in ("stop_breached", "target_breached"):
                    pos = next(
                        (p for p in fresh_positions if p["ticker"] == event.ticker),
                        None,
                    )
                    if pos:
                        await sched._execute_breach_guarded(
                            event.ticker, pos,
                            event.current_price, event.alert_type,
                        )
        except Exception:
            logger.exception("IEX fallback evaluation failed")


async def check_stream_health(sched: TradingScheduler) -> None:
    """Alert after 5 consecutive disconnected cycles (~5 min)."""
    _DEGRADE_THRESHOLD = 5
    if not sched._stream:
        return

    if not sched._stream.is_connected:
        sched._stream_degraded_cycles += 1
        if (
            sched._stream_degraded_cycles >= _DEGRADE_THRESHOLD
            and not sched._stream_alert_sent
        ):
            sched._stream_alert_sent = True
            logger.warning(
                "Stream has been disconnected for %d min"
                " — emitting degradation alert",
                sched._stream_degraded_cycles,
            )
            await sched._on_event(StreamDegradedEvent(
                disconnected_minutes=sched._stream_degraded_cycles,
            ))
    else:
        if sched._stream_alert_sent:
            logger.info(
                "Stream reconnected after %d min — emitting recovery alert",
                sched._stream_degraded_cycles,
            )
            await sched._on_event(StreamRecoveredEvent(
                downtime_minutes=sched._stream_degraded_cycles,
            ))
            sched._stream_alert_sent = False
        sched._stream_degraded_cycles = 0


# ------------------------------------------------------------------
# Polling loop
# ------------------------------------------------------------------

async def polling_loop(sched: TradingScheduler) -> None:
    """Price alert polling — active when stream is disconnected or disabled."""
    while True:
        await asyncio.sleep(60)
        if not sched._engine.is_market_open() and not has_open_crypto_positions():
            continue
        # Skip polling while stream is connected
        if sched._stream and sched._stream.is_connected:
            continue
        try:
            now = datetime.now(_ET)
            elapsed = (now - sched._last_alert_check).total_seconds() / 60
            if elapsed < sched._current_alert_interval:
                continue
            sched._last_alert_check = now

            state = load_positions()
            positions = state.get("positions", [])
            if not positions:
                continue

            tickers = [p["ticker"] for p in positions]
            if alpaca_enabled():
                async with sched._snapshot_lock:
                    quotes = await get_snapshots(tickers)
            else:
                quotes = await asyncio.to_thread(fetch_quotes, tickers)

            events = await sched._engine.check_prices(positions, quotes)
            for event in events:
                await sched._on_event(event)
                if (
                    event.alert_type in ("stop_breached", "target_breached")
                    and alpaca_enabled()
                ):
                    # Reload from DB so execute_breach sees current
                    # exit order IDs (MIC-102).
                    pos = _load_position_meta(event.ticker)
                    if not pos:
                        pos = next(
                            (p for p in positions if p["ticker"] == event.ticker), None,
                        )
                    if pos:
                        await sched._execute_breach_guarded(
                            event.ticker, pos, event.current_price, event.alert_type,
                        )

            # Update adaptive interval
            if PARAMS.adaptive_alerts_enabled:
                fast = await should_use_fast_alerts(sched, positions, quotes)
                sched._current_alert_interval = (
                    PARAMS.fast_alert_interval if fast else PARAMS.base_alert_interval
                )
        except Exception:
            logger.exception("Polling loop error")


# ------------------------------------------------------------------
# Reconciliation loop
# ------------------------------------------------------------------

async def reconcile_loop(sched: TradingScheduler) -> None:
    """15-minute timer: check broker-side OCO fills and sync DB."""
    while True:
        await asyncio.sleep(900)
        if not alpaca_enabled() or not sched._engine.is_market_open():
            continue
        try:
            state = load_positions()
            positions = state.get("positions", [])
            filled = await check_exit_fills(positions)
            for fill in filled:
                ticker = fill["ticker"]
                fill_price = fill["fill_price"]
                exit_reason = fill["exit_reason"]
                logger.info(
                    "Broker exit filled for %s at %.2f (%s) — syncing DB",
                    ticker, fill_price, exit_reason,
                )
                pos = await async_close_position(ticker, fill_price, exit_reason)
                if pos:

                    record_closed_trades([pos])
                    await sched._on_event(ExitResult(
                        ticker=ticker,
                        exit_reason=exit_reason,
                        fill_price=fill_price,
                        realized_pnl_pct=pos.get("realized_pnl_pct"),
                        bars_held=pos.get("bars_held"),
                    ))
                    if sched._stream:
                        all_pos = load_positions().get("positions", [])
                        sched._stream.update_subscriptions(
                            {p["ticker"] for p in all_pos},
                        )
        except Exception:
            logger.exception("Reconciliation loop error")


# ------------------------------------------------------------------
# Timer loops
# ------------------------------------------------------------------

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
                    sched._ran_morning_scan != today
                    and not sched._scan_in_progress.get("morning")
                ):
                    sched._ran_morning_scan = today
                    sched._scan_in_progress["morning"] = True
                    try:
                        async with asyncio.timeout(600):
                            await sched.run_premarket_scan()
                    finally:
                        sched._scan_in_progress["morning"] = False
            # 9:30 AM ET — execute entries/exits at market open
            elif PARAMS.morning_scan_enabled and now.hour == 9 and 30 <= now.minute < 35:
                if (
                    sched._ran_morning_execute != today
                    and not sched._scan_in_progress.get("execute")
                ):
                    sched._ran_morning_execute = today
                    sched._scan_in_progress["execute"] = True
                    try:
                        async with asyncio.timeout(300):
                            await sched.run_market_open_execute()
                    finally:
                        sched._scan_in_progress["execute"] = False
            elif PARAMS.midday_scans_enabled and now.hour == 11 and now.minute < 5:
                if (
                    sched._ran_midday_1 != today
                    and not sched._scan_in_progress.get("midday_1")
                ):
                    sched._ran_midday_1 = today
                    sched._scan_in_progress["midday_1"] = True
                    try:
                        async with asyncio.timeout(600):
                            await sched.run_midday_scan()
                    finally:
                        sched._scan_in_progress["midday_1"] = False
            elif PARAMS.midday_scans_enabled and now.hour == 13 and now.minute < 5:
                if (
                    sched._ran_midday_2 != today
                    and not sched._scan_in_progress.get("midday_2")
                ):
                    sched._ran_midday_2 = today
                    sched._scan_in_progress["midday_2"] = True
                    try:
                        async with asyncio.timeout(600):
                            await sched.run_midday_scan()
                    finally:
                        sched._scan_in_progress["midday_2"] = False
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
    if sched._ran_weekend_scan == run_key:
        return

    crypto_tickers = [t for t, cfg in ASSETS.items() if cfg.asset_class == "crypto"]
    if not crypto_tickers:
        return

    # Only run if at least one crypto ticker has a trained model
    manifest = load_manifest()
    if not manifest or not any(t in manifest for t in crypto_tickers):
        logger.debug("Weekend crypto scan: no trained crypto models found, skipping")
        return

    if sched._scan_in_progress.get("weekend_crypto"):
        return
    sched._ran_weekend_scan = run_key
    logger.info(
        "Weekend crypto scan: %s (tickers=%s)",
        run_key, sorted(crypto_tickers),
    )
    sched._scan_in_progress["weekend_crypto"] = True
    try:
        async with asyncio.timeout(600):
            await sched.run_daily_scan("weekend_crypto", tickers=crypto_tickers)
    finally:
        sched._scan_in_progress["weekend_crypto"] = False


async def retrain_loop(sched: TradingScheduler) -> None:
    """Time-based: weekly retrain Sunday >= 6 PM ET."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now(_ET)
        if now.weekday() != 6:  # Sunday only
            continue
        today = now.date().isoformat()
        if now.hour >= 18 and sched._ran_retrain != today:
            try:
                sched._ran_retrain = today
                await sched.run_retrain()
            except Exception:
                logger.exception("Retrain loop error")


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

async def should_use_fast_alerts(
    sched: TradingScheduler,
    positions: list[dict],
    quotes: dict[str, dict],
) -> bool:
    """True if any position is near its stop or portfolio is drawing down."""
    from trading.risk_manager import compute_drawdown

    for pos in positions:
        ticker = pos["ticker"]
        price = quotes.get(ticker, {}).get("price", 0)
        stop = max(pos.get("stop_price", 0), pos.get("trailing_stop", 0))
        if stop > 0 and price > 0:
            distance_pct = (price / stop - 1) * 100
            if 0 < distance_pct <= PARAMS.stop_proximity_pct:
                return True
    try:
        dd = compute_drawdown(100_000)
        if dd >= PARAMS.risk_tier1_dd:
            return True
    except Exception:
        logger.warning("Drawdown computation failed in _should_use_fast_alerts", exc_info=True)
    return False


def has_open_crypto_positions() -> bool:
    """True if any open position is a crypto asset."""
    from rally_ml.config import ASSETS
    positions = load_positions().get("positions", [])
    return any(
        ASSETS.get(p["ticker"]) and ASSETS[p["ticker"]].asset_class == "crypto"
        for p in positions
    )
