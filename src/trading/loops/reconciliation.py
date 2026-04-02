"""Reconciliation loop: check broker-side OCO fills and sync DB."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from db.portfolio import record_closed_trades
from db.positions import load_positions
from integrations.alpaca.broker import is_enabled as alpaca_enabled
from integrations.alpaca.fills import check_exit_fills
from trading.events import ExitResult
from trading.positions import async_close_position

if TYPE_CHECKING:
    from trading.scheduler import TradingScheduler

logger = logging.getLogger(__name__)


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
