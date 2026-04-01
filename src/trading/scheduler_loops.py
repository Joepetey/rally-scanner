"""Background loop functions extracted from TradingScheduler."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from config import ASSETS, PARAMS
from core.data import fetch_quotes
from core.persistence import load_manifest
from db.portfolio import record_closed_trades
from db.positions import (
    load_position_meta as _load_position_meta,
)
from db.positions import (
    load_positions,
)
from integrations.alpaca.executor import (
    check_exit_fills,
    get_snapshots,
)
from integrations.alpaca.executor import (
    is_enabled as alpaca_enabled,
)
from trading.engine import (
    ExitResult,
    StreamDegradedEvent,
    StreamRecoveredEvent,
)
from trading.positions import async_close_position
from trading.scheduler_stream import (
    execute_breach_guarded,
    has_open_crypto_positions,
    should_use_fast_alerts,
)

_ET = __import__("zoneinfo").ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


async def housekeeping_loop(scheduler) -> None:
    """1-minute timer: fill checks, OCO placement, position sync.

    Also handles stale-price warnings and IEX coverage fallback every
    2 cycles (~2 min) when the stream is connected but a ticker has had
    no trade event.
    """
    while True:
        await asyncio.sleep(60)
        if not scheduler._engine.is_market_open() and not has_open_crypto_positions():
            continue
        try:
            scheduler._housekeeping_cycles += 1
            state = load_positions()
            positions = state.get("positions", [])
            result = await scheduler._engine.run_housekeeping(positions)

            if result.fills_confirmed or result.orders_placed:
                await scheduler._on_event(result)
                if scheduler._stream:
                    all_pos = load_positions().get("positions", [])
                    scheduler._stream.update_subscriptions({p["ticker"] for p in all_pos})

            # Stale price check + IEX REST fallback every 2 housekeeping cycles
            if (
                scheduler._stream and scheduler._stream.is_connected
                and scheduler._housekeeping_cycles % 2 == 0
            ):
                new_stale, known_stale, never_traded = scheduler._stream.get_stale_tickers(
                    stale_seconds=300.0,
                )
                stale = new_stale + known_stale + never_traded
                if new_stale:
                    logger.warning(
                        "No stream trade in 5 min for %d ticker(s): %s \u2014 "
                        "possible subscription issue",
                        len(new_stale), sorted(new_stale),
                    )
                if never_traded:
                    logger.info(
                        "No stream trade yet for %d ticker(s): %s \u2014 "
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
                        async with scheduler._snapshot_lock:
                            snapshots = await get_snapshots(stale)
                        fresh_state = load_positions()
                        fresh_positions = fresh_state.get("positions", [])
                        events = await scheduler._engine.check_prices(
                            [p for p in fresh_positions if p["ticker"] in snapshots],
                            snapshots,
                        )
                        for event in events:
                            await scheduler._on_event(event)
                            if event.alert_type in ("stop_breached", "target_breached"):
                                pos = next(
                                    (p for p in fresh_positions if p["ticker"] == event.ticker),
                                    None,
                                )
                                if pos:
                                    await execute_breach_guarded(
                                        scheduler, event.ticker, pos,
                                        event.current_price, event.alert_type,
                                    )
                    except Exception:
                        logger.exception("IEX fallback evaluation failed")

            # Stream degradation monitoring
            _DEGRADE_THRESHOLD = 5
            if scheduler._stream:
                if not scheduler._stream.is_connected:
                    scheduler._stream_degraded_cycles += 1
                    if (
                        scheduler._stream_degraded_cycles >= _DEGRADE_THRESHOLD
                        and not scheduler._stream_alert_sent
                    ):
                        scheduler._stream_alert_sent = True
                        logger.warning(
                            "Stream has been disconnected for %d min"
                            " \u2014 emitting degradation alert",
                            scheduler._stream_degraded_cycles,
                        )
                        await scheduler._on_event(StreamDegradedEvent(
                            disconnected_minutes=scheduler._stream_degraded_cycles,
                        ))
                else:
                    if scheduler._stream_alert_sent:
                        logger.info(
                            "Stream reconnected after %d min \u2014 emitting recovery alert",
                            scheduler._stream_degraded_cycles,
                        )
                        await scheduler._on_event(StreamRecoveredEvent(
                            downtime_minutes=scheduler._stream_degraded_cycles,
                        ))
                        scheduler._stream_alert_sent = False
                    scheduler._stream_degraded_cycles = 0

        except Exception:
            logger.exception("Housekeeping loop error")


async def polling_loop(scheduler) -> None:
    """Price alert polling \u2014 active when stream is disconnected or disabled."""
    while True:
        await asyncio.sleep(60)
        if not scheduler._engine.is_market_open() and not has_open_crypto_positions():
            continue
        # Skip polling while stream is connected
        if scheduler._stream and scheduler._stream.is_connected:
            continue
        try:
            now = datetime.now(_ET)
            elapsed = (now - scheduler._last_alert_check).total_seconds() / 60
            if elapsed < scheduler._current_alert_interval:
                continue
            scheduler._last_alert_check = now

            state = load_positions()
            positions = state.get("positions", [])
            if not positions:
                continue

            tickers = [p["ticker"] for p in positions]
            if alpaca_enabled():
                async with scheduler._snapshot_lock:
                    quotes = await get_snapshots(tickers)
            else:
                quotes = await asyncio.to_thread(fetch_quotes, tickers)

            events = await scheduler._engine.check_prices(positions, quotes)
            for event in events:
                await scheduler._on_event(event)
                if (
                    event.alert_type in ("stop_breached", "target_breached")
                    and alpaca_enabled()
                ):
                    pos = _load_position_meta(event.ticker)
                    if not pos:
                        pos = next((p for p in positions if p["ticker"] == event.ticker), None)
                    if pos:
                        await execute_breach_guarded(
                            scheduler, event.ticker, pos, event.current_price, event.alert_type,
                        )

            # Update adaptive interval
            if PARAMS.adaptive_alerts_enabled:
                fast = await should_use_fast_alerts(positions, quotes)
                scheduler._current_alert_interval = (
                    PARAMS.fast_alert_interval if fast else PARAMS.base_alert_interval
                )
        except Exception:
            logger.exception("Polling loop error")


async def reconcile_loop(scheduler) -> None:
    """15-minute timer: check broker-side OCO fills and sync DB."""
    while True:
        await asyncio.sleep(900)
        if not alpaca_enabled() or not scheduler._engine.is_market_open():
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
                    "Broker exit filled for %s at %.2f (%s) \u2014 syncing DB",
                    ticker, fill_price, exit_reason,
                )
                pos = await async_close_position(ticker, fill_price, exit_reason)
                if pos:
                    record_closed_trades([pos])
                    await scheduler._on_event(ExitResult(
                        ticker=ticker,
                        exit_reason=exit_reason,
                        fill_price=fill_price,
                        realized_pnl_pct=pos.get("realized_pnl_pct"),
                        bars_held=pos.get("bars_held"),
                    ))
                    if scheduler._stream:
                        all_pos = load_positions().get("positions", [])
                        scheduler._stream.update_subscriptions({p["ticker"] for p in all_pos})
        except Exception:
            logger.exception("Reconciliation loop error")


async def regime_loop(scheduler) -> None:
    """30-minute timer: HMM regime check."""
    while True:
        await asyncio.sleep(1800)
        try:
            await scheduler.run_regime_check()
        except Exception:
            logger.exception("Regime loop error")


async def scan_loop(scheduler) -> None:
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
                await maybe_run_weekend_crypto_scan(scheduler, now, today)
            except Exception:
                logger.exception("Weekend crypto scan error")
            continue

        # Weekday scans
        try:
            # 9:00 AM ET — pre-market scan (full universe)
            if PARAMS.morning_scan_enabled and now.hour == 9 and now.minute < 5:
                if (
                    scheduler._ran_morning_scan != today
                    and not scheduler._scan_in_progress.get("morning")
                ):
                    scheduler._ran_morning_scan = today
                    scheduler._scan_in_progress["morning"] = True
                    try:
                        async with asyncio.timeout(600):
                            await scheduler.run_premarket_scan()
                    finally:
                        scheduler._scan_in_progress["morning"] = False
            # 9:30 AM ET — execute entries/exits at market open
            elif PARAMS.morning_scan_enabled and now.hour == 9 and 30 <= now.minute < 35:
                if (
                    scheduler._ran_morning_execute != today
                    and not scheduler._scan_in_progress.get("execute")
                ):
                    scheduler._ran_morning_execute = today
                    scheduler._scan_in_progress["execute"] = True
                    try:
                        async with asyncio.timeout(300):
                            await scheduler.run_market_open_execute()
                    finally:
                        scheduler._scan_in_progress["execute"] = False
            elif PARAMS.midday_scans_enabled and now.hour == 11 and now.minute < 5:
                if (
                    scheduler._ran_midday_1 != today
                    and not scheduler._scan_in_progress.get("midday_1")
                ):
                    scheduler._ran_midday_1 = today
                    scheduler._scan_in_progress["midday_1"] = True
                    try:
                        async with asyncio.timeout(600):
                            await scheduler.run_midday_scan()
                    finally:
                        scheduler._scan_in_progress["midday_1"] = False
            elif PARAMS.midday_scans_enabled and now.hour == 13 and now.minute < 5:
                if (
                    scheduler._ran_midday_2 != today
                    and not scheduler._scan_in_progress.get("midday_2")
                ):
                    scheduler._ran_midday_2 = today
                    scheduler._scan_in_progress["midday_2"] = True
                    try:
                        async with asyncio.timeout(600):
                            await scheduler.run_midday_scan()
                    finally:
                        scheduler._scan_in_progress["midday_2"] = False
        except Exception:
            logger.exception("Scan loop error")


async def maybe_run_weekend_crypto_scan(
    scheduler, now: datetime, today: str,
) -> None:
    """Run a crypto-only scan every 6 hours on weekends."""
    # Fire at hours 0, 6, 12, 18 ET within a 5-minute window
    if now.hour % 6 != 0 or now.minute >= 5:
        return

    run_key = f"{today}_{now.hour}"
    if scheduler._ran_weekend_scan == run_key:
        return

    crypto_tickers = [t for t, cfg in ASSETS.items() if cfg.asset_class == "crypto"]
    if not crypto_tickers:
        return

    # Only run if at least one crypto ticker has a trained model
    manifest = load_manifest()
    if not manifest or not any(t in manifest for t in crypto_tickers):
        logger.debug("Weekend crypto scan: no trained crypto models found, skipping")
        return

    if scheduler._scan_in_progress.get("weekend_crypto"):
        return
    scheduler._ran_weekend_scan = run_key
    logger.info(
        "Weekend crypto scan: %s (tickers=%s)",
        run_key, sorted(crypto_tickers),
    )
    scheduler._scan_in_progress["weekend_crypto"] = True
    try:
        async with asyncio.timeout(600):
            await scheduler.run_daily_scan("weekend_crypto", tickers=crypto_tickers)
    finally:
        scheduler._scan_in_progress["weekend_crypto"] = False


async def retrain_loop(scheduler) -> None:
    """Time-based: weekly retrain Sunday >= 6 PM ET."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now(_ET)
        if now.weekday() != 6:  # Sunday only
            continue
        today = now.date().isoformat()
        if now.hour >= 18 and scheduler._ran_retrain != today:
            try:
                scheduler._ran_retrain = today
                await scheduler.run_retrain()
            except Exception:
                logger.exception("Retrain loop error")
