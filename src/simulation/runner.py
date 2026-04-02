"""SimulationRunner — orchestrates BTC-USD paper trading simulations end-to-end.

Flow for stream-based scenarios (target, stop, trail):
  1. Fetch live BTC price from Alpaca snapshot
  2. execute_entries() → real paper order on Alpaca
  3. Poll check_pending_fills() until confirmed
  4. place_exit_orders() → OCO on Alpaca
  5. inject_trade() to force the intended price condition
  6. Scheduler detects breach via _on_stream_trade → execute_breach → exit
  7. Poll until position disappears from DB

Flow for time scenario:
  1-4. Same as above (entry, fill, OCO)
  5. Advance bars_held past time_stop_bars in DB
  6. execute_exit() to close Alpaca position + cancel OCO
  7. close_position_intraday() to close DB record

Discord embeds are emitted at each step via the send_embed callback.
The breach alert and exit embeds for stream scenarios are emitted by the
scheduler's normal event path — the runner does not duplicate them.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from pydantic import BaseModel
from rally_ml.config import PARAMS

from db.ops.events import clear_price_alerts
from db.trading.positions import load_position_meta, save_position_meta
from integrations.alpaca.account import get_snapshots
from integrations.alpaca.entries import execute_entries
from integrations.alpaca.exits import execute_exit, place_exit_orders
from integrations.alpaca.fills import check_pending_fills
from integrations.discord.notify import (
    _exit_embed,
    _fill_confirmation_embed,
    _order_embed,
)
from simulation.scenarios import (
    TICKER,
    get_inject_prices,
    get_signal_for_scenario,
    setup_position_for_scenario,
)
from trading.positions import close_position_intraday, update_fill_prices

logger = logging.getLogger(__name__)

_FILL_POLL_INTERVAL = 5     # seconds between fill poll attempts
_FILL_POLL_ATTEMPTS = 24    # 2 minutes total before timeout
_EXIT_POLL_INTERVAL = 2     # seconds between exit poll attempts
_EXIT_POLL_ATTEMPTS = 20    # 40 seconds total before timeout

SendEmbed = Callable[[dict], Awaitable[None]]


class SimulationResult(BaseModel):
    scenario: str
    ticker: str = TICKER
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_price: float = 0.0
    exit_reason: str | None = None
    exit_price: float | None = None
    realized_pnl_pct: float | None = None
    success: bool
    error: str | None = None


class SimulationRunner:
    """Run a single BTC paper trading simulation scenario end-to-end.

    Args:
        inject_fn: AlpacaStreamManager.inject_trade (required for stream scenarios).
    """

    def __init__(
        self,
        inject_fn: Callable[[str, float], None] | None = None,
    ) -> None:
        self._inject_fn = inject_fn

    async def run(
        self,
        scenario: str,
        equity: float,
        send_embed: SendEmbed,
    ) -> SimulationResult:
        """Run the full simulation. Emits embeds via send_embed at each milestone."""
        valid = {"target", "stop", "trail", "time"}
        if scenario not in valid:
            return SimulationResult(
                scenario=scenario, success=False,
                error=f"Unknown scenario: {scenario!r}. Must be one of {sorted(valid)}",
            )

        if scenario != "time" and self._inject_fn is None:
            return SimulationResult(
                scenario=scenario, success=False,
                error=(
                    "Stream not running — set ALPACA_STREAM_ENABLED=1 "
                    "and ALPACA_AUTO_EXECUTE=1 to use stream scenarios"
                ),
            )

        try:
            return await self._run(scenario, equity, send_embed)
        except Exception as e:
            logger.exception("Simulation failed (scenario=%s)", scenario)
            return SimulationResult(scenario=scenario, success=False, error=str(e))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(
        self, scenario: str, equity: float, send_embed: SendEmbed,
    ) -> SimulationResult:
        # Clear per-day alert dedup for the simulation ticker so consecutive
        # scenarios (e.g. stop then trail) can each fire stop_breached.
        today = datetime.now().strftime("%Y-%m-%d")
        clear_price_alerts(TICKER, today)

        # 1. Live BTC price
        snapshots = await get_snapshots([TICKER])
        snap = snapshots.get(TICKER)
        if not snap:
            return SimulationResult(
                scenario=scenario, success=False, error="Could not fetch BTC price from Alpaca",
            )
        entry_price = snap["price"]
        logger.info("sim[%s]: BTC price=%.2f equity=%.2f", scenario, entry_price, equity)

        # 2. Place paper entry order
        signal = get_signal_for_scenario(scenario, entry_price)
        results = await execute_entries([signal], equity)
        result = results[0]
        if not result.success:
            return SimulationResult(
                scenario=scenario, success=False,
                error=f"Entry order failed: {result.error}",
            )

        order_id = result.order_id
        qty = result.qty
        logger.info("sim[%s]: entry order_id=%s qty=%.6f", scenario, order_id, qty)

        # 3. Persist position to DB with order_id so fill-tracking can update it
        pos = setup_position_for_scenario(scenario, entry_price)
        pos["order_id"] = order_id
        pos["qty"] = qty
        save_position_meta(pos)

        # Emit entry embed
        await send_embed(_order_embed([result.model_dump()], equity))

        # 4. Poll for fill confirmation (BTC paper orders fill within seconds)
        fill_price = entry_price
        fill_confirmed = False
        for attempt in range(_FILL_POLL_ATTEMPTS):
            await asyncio.sleep(_FILL_POLL_INTERVAL)
            fills = await check_pending_fills([order_id])
            if order_id in fills:
                fill_price = fills[order_id]
                await update_fill_prices(fills)
                fresh = load_position_meta(TICKER)
                if fresh is None:
                    return SimulationResult(
                        scenario=scenario, success=False,
                        error="Position disappeared from DB after fill update",
                    )
                pos = fresh
                fill_confirmed = True
                logger.info(
                    "sim[%s]: fill confirmed %.4f (attempt %d)", scenario, fill_price, attempt + 1
                )
                break

        if not fill_confirmed:
            return SimulationResult(
                scenario=scenario, success=False,
                error=f"Fill not confirmed after {_FILL_POLL_ATTEMPTS * _FILL_POLL_INTERVAL}s",
            )

        # Emit fill-confirmed embed
        await send_embed(_fill_confirmation_embed([{
            "ticker": TICKER,
            "fill_price": fill_price,
            "qty": qty,
            "stop_price": pos["stop_price"],
            "target_price": pos["target_price"],
        }]))

        # 5. Place OCO exit orders
        target_oid, stop_oid = await place_exit_orders(
            TICKER, qty, pos["target_price"], pos["stop_price"],
        )
        pos = load_position_meta(TICKER) or pos  # re-read; fallback to local if missing
        pos["target_order_id"] = target_oid
        pos["trail_order_id"] = stop_oid
        save_position_meta(pos)
        logger.info("sim[%s]: OCO placed target=%s stop=%s", scenario, target_oid, stop_oid)

        # 6. Trigger exit
        if scenario == "time":
            exit_reason, exit_price = await self._run_time_exit(pos, fill_price)
        else:
            exit_reason, exit_price = await self._run_stream_exit(scenario, pos)

        if exit_reason is None:
            return SimulationResult(
                scenario=scenario, success=False,
                error=(
                    f"Exit did not complete within timeout "
                    f"({_EXIT_POLL_ATTEMPTS * _EXIT_POLL_INTERVAL}s)"
                ),
            )

        pnl_pct = round((exit_price / fill_price - 1) * 100, 2) if fill_price else None

        # Emit exit embed for time scenario (stream scenarios have it emitted by the scheduler)
        if scenario == "time":
            await send_embed(_exit_embed([{
                "ticker": TICKER,
                "exit_reason": exit_reason,
                "realized_pnl_pct": pnl_pct,
                "bars_held": pos.get("bars_held", 0),
            }]))

        return SimulationResult(
            scenario=scenario,
            ticker=TICKER,
            entry_price=fill_price,
            target_price=pos["target_price"],
            stop_price=pos["stop_price"],
            exit_reason=exit_reason,
            exit_price=exit_price,
            realized_pnl_pct=pnl_pct,
            success=True,
        )

    async def _run_stream_exit(
        self, scenario: str, pos: dict,
    ) -> tuple[str | None, float]:
        """Inject price(s) and wait for scheduler to execute the breach exit.

        Returns (exit_reason, exit_price) or (None, 0.0) on timeout.
        """
        inject_prices = get_inject_prices(scenario, pos)

        if scenario == "trail":
            # Phase 1: inject high price to tighten trailing stop.
            # _on_stream_trade calls update_position_for_price + _save_position_meta
            # synchronously, so the DB is updated before inject_trade returns.
            high_price = inject_prices[0]
            self._inject_fn(TICKER, high_price)
            logger.info("sim[trail]: Phase 1 injected high=%.2f", high_price)

            # Re-read pos to get the tightened trailing_stop
            fresh = load_position_meta(TICKER)
            if fresh is None:
                # Already closed (shouldn't happen on a high inject, but handle gracefully)
                return "trail_stop", high_price
            new_trail = fresh.get("trailing_stop", pos.get("trailing_stop", 0))

            # Phase 2: inject just below the new trailing stop to trigger stop_breached
            low_price = round(new_trail * 0.996, 2)
            logger.info(
                "sim[trail]: trailing_stop now=%.4f → Phase 2 low=%.2f", new_trail, low_price,
            )
            self._inject_fn(TICKER, low_price)
            exit_reason_hint = "trail_stop"
            exit_price_hint = low_price
        else:
            price = inject_prices[0]
            self._inject_fn(TICKER, price)
            logger.info("sim[%s]: injected exit price=%.2f", scenario, price)
            exit_reason_hint = "profit_target" if scenario == "target" else "stop"
            exit_price_hint = price

        # Wait for scheduler's _execute_breach_guarded to close the position
        for _ in range(_EXIT_POLL_ATTEMPTS):
            await asyncio.sleep(_EXIT_POLL_INTERVAL)
            if load_position_meta(TICKER) is None:
                logger.info("sim[%s]: position closed by scheduler", scenario)
                return exit_reason_hint, exit_price_hint

        return None, 0.0

    async def _run_time_exit(
        self, pos: dict, current_price: float,
    ) -> tuple[str | None, float]:
        """Advance bars_held and close both the Alpaca position and DB record."""
        pos["bars_held"] = PARAMS.time_stop_bars + 1
        save_position_meta(pos)

        # Close on Alpaca (cancels OCO + market sell)
        result = await execute_exit(
            TICKER,
            trail_order_id=pos.get("trail_order_id"),
        )
        exit_price = result.fill_price or current_price

        # Close in DB
        close_position_intraday(TICKER, exit_price, "time_stop")
        logger.info("sim[time]: closed at %.4f", exit_price)

        return "time_stop", exit_price
