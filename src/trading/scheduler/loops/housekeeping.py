"""Housekeeping loop: fill checks, OCO placement, stale ticker coverage, stream health."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from db.trading.positions import load_positions

from integrations.alpaca.account import get_snapshots
from integrations.alpaca.broker import is_enabled as alpaca_enabled
from trading.events import StreamDegradedEvent, StreamRecoveredEvent
from trading.scheduler.loops._crypto import has_open_crypto_positions

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

logger = logging.getLogger(__name__)


async def housekeeping_loop(sched: TradingScheduler) -> None:
    """1-minute timer: fill checks, OCO placement, position sync."""
    while True:
        await asyncio.sleep(60)
        if not sched._engine.is_market_open() and not has_open_crypto_positions():
            continue
        try:
            sched.state.housekeeping_cycles += 1
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
        and sched.state.housekeeping_cycles % 2 == 0
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
            async with sched.state.snapshot_lock:
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
        sched.state.stream_degraded_cycles += 1
        if (
            sched.state.stream_degraded_cycles >= _DEGRADE_THRESHOLD
            and not sched.state.stream_alert_sent
        ):
            sched.state.stream_alert_sent = True
            logger.warning(
                "Stream has been disconnected for %d min"
                " — emitting degradation alert",
                sched.state.stream_degraded_cycles,
            )
            await sched._on_event(StreamDegradedEvent(
                disconnected_minutes=sched.state.stream_degraded_cycles,
            ))
    else:
        if sched.state.stream_alert_sent:
            logger.info(
                "Stream reconnected after %d min — emitting recovery alert",
                sched.state.stream_degraded_cycles,
            )
            await sched._on_event(StreamRecoveredEvent(
                downtime_minutes=sched.state.stream_degraded_cycles,
            ))
            sched.state.stream_alert_sent = False
        sched.state.stream_degraded_cycles = 0
