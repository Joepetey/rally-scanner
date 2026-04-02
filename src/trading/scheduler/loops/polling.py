"""Polling loop: price alert polling when stream is disconnected or disabled."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from rally_ml.config import PARAMS
from rally_ml.core.data import fetch_quotes

from db.trading.positions import load_position_meta as _load_position_meta
from db.trading.positions import load_positions
from integrations.alpaca.account import get_snapshots
from integrations.alpaca.broker import is_enabled as alpaca_enabled
from trading.scheduler.loops._crypto import has_open_crypto_positions

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

_ET = __import__("zoneinfo").ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


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
            elapsed = (now - sched.state.last_alert_check).total_seconds() / 60
            if elapsed < sched.state.current_alert_interval:
                continue
            sched.state.last_alert_check = now

            state = load_positions()
            positions = state.get("positions", [])
            if not positions:
                continue

            tickers = [p["ticker"] for p in positions]
            if alpaca_enabled():
                async with sched.state.snapshot_lock:
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
                sched.state.current_alert_interval = (
                    PARAMS.fast_alert_interval if fast else PARAMS.base_alert_interval
                )
        except Exception:
            logger.exception("Polling loop error")


async def should_use_fast_alerts(
    sched: TradingScheduler,
    positions: list[dict],
    quotes: dict[str, dict],
) -> bool:
    """True if any position is near its stop or portfolio is drawing down."""
    from trading.risk import compute_drawdown

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
