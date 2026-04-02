#!/usr/bin/env python
"""Run BTC-USD paper trading simulations locally without the Discord bot.

Boots TradingScheduler (stream + breach detection), runs all four scenarios
sequentially, and prints a summary table.

Usage:
    python scripts/simulate.py                  # run all scenarios
    python scripts/simulate.py target           # run one scenario
    python scripts/simulate.py --equity 5000    # override equity

Requires: ALPACA_SIMULATION_API_KEY, ALPACA_SIMULATION_SECRET_KEY, DATABASE_URL in .env (or env).
Stream scenarios (target, stop, trail) also require ALPACA_STREAM_ENABLED=1.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os

_sim_key = os.environ.get("ALPACA_SIMULATION_API_KEY")
_sim_secret = os.environ.get("ALPACA_SIMULATION_SECRET_KEY")
if not _sim_key or not _sim_secret:
    print("ERROR: ALPACA_SIMULATION_API_KEY and ALPACA_SIMULATION_SECRET_KEY must be set")
    sys.exit(1)
os.environ["ALPACA_API_KEY"] = _sim_key
os.environ["ALPACA_SECRET_KEY"] = _sim_secret

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db import close_pool, init_pool, init_schema
from db.core.pool import get_conn
from db.trading.positions import delete_position_meta
from integrations.alpaca.account import get_account_equity
from integrations.alpaca.exits import execute_exit
from log import setup_logging
from simulation.runner import SimulationResult, SimulationRunner
from simulation.scenarios import TICKER
from trading.scheduler import TradingScheduler

setup_logging()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("alpaca").setLevel(logging.WARNING)
logging.getLogger("alpaca.data.live.websocket").setLevel(logging.CRITICAL)
logging.getLogger("alpaca.trading.stream").setLevel(logging.CRITICAL)
logger = logging.getLogger("simulate")

SCENARIOS = ("target", "stop", "trail", "time")


def _print_embed(embed_dict: dict) -> None:
    title = embed_dict.get("title", "")
    fields = embed_dict.get("fields", [])
    desc = embed_dict.get("description", "")
    print(f"    [{title}]", end="")
    if desc:
        print(f" {desc}", end="")
    for f in fields:
        print(f"  {f['name']}: {f['value']}", end="")
    print()


async def _send_embed(embed_dict: dict) -> None:
    _print_embed(embed_dict)


def _divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _print_summary(results: list[SimulationResult]) -> None:
    _divider("Summary")
    col = 10
    header = f"  {'scenario':<{col}}  {'result':<8}  {'entry':>10}  {'exit':>10}  {'reason':<14}  pnl"
    print(header)
    print("  " + "-" * (len(header) - 2))
    all_passed = True
    for r in results:
        if r.success:
            sign = "+" if (r.realized_pnl_pct or 0) >= 0 else ""
            print(
                f"  {r.scenario:<{col}}  {'PASSED':<8}  "
                f"${r.entry_price:>9,.2f}  ${r.exit_price:>9,.2f}  "
                f"{r.exit_reason:<14}  {sign}{r.realized_pnl_pct:.2f}%"
            )
        else:
            all_passed = False
            print(f"  {r.scenario:<{col}}  {'FAILED':<8}  {r.error}")
    print()
    if all_passed:
        print("  ALL SCENARIOS PASSED\n")
    else:
        print("  ONE OR MORE SCENARIOS FAILED\n")


async def _reset_sim_state() -> None:
    """Clear any leftover BTC position (DB + Alpaca) and alert-log dedup entries."""
    # Close any open BTC position on Alpaca (best-effort; ignore 404)
    try:
        await execute_exit(TICKER)
    except Exception:
        pass

    # Remove any open BTC position from the DB
    delete_position_meta(TICKER)

    # Clear ALL price_alert_log rows for BTC so dedup doesn't suppress breach events.
    # Must not filter by CURRENT_DATE — engine logs alerts in ET while PG uses UTC,
    # so late-night UTC runs (where ET date != UTC date) would miss stale records.
    with get_conn() as conn:
        conn.cursor().execute(
            "DELETE FROM price_alert_log WHERE ticker = %s",
            (TICKER,),
        )


async def main(scenarios: tuple[str, ...], equity_override: float) -> None:
    init_pool()
    init_schema()
    await _reset_sim_state()

    scheduler = TradingScheduler(on_event=lambda _: asyncio.sleep(0))
    await scheduler.start()

    # Cancel all background loops — simulation only needs the stream + breach path.
    # Housekeeping/polling/reconcile loops interfere by syncing/modifying positions.
    for task in scheduler._tasks:
        task.cancel()
    await asyncio.gather(*scheduler._tasks, return_exceptions=True)
    scheduler._tasks.clear()

    try:
        if equity_override > 0:
            equity = equity_override
        else:
            equity = await get_account_equity()
            if equity <= 0:
                print("  [FAIL] equity is zero — check API keys / paper account")
                sys.exit(1)

        print(f"\n  equity=${equity:,.0f}  scenarios={list(scenarios)}")

        runner = SimulationRunner(
            inject_fn=scheduler._stream.inject_trade if scheduler._stream else None,
        )

        results: list[SimulationResult] = []
        for scenario in scenarios:
            await _reset_sim_state()
            _divider(f"Running: {scenario}")
            result = await runner.run(scenario, equity, _send_embed)
            results.append(result)
            if result.success:
                sign = "+" if (result.realized_pnl_pct or 0) >= 0 else ""
                print(f"  -> PASSED  exit={result.exit_reason}  pnl={sign}{result.realized_pnl_pct:.2f}%")
            else:
                print(f"  -> FAILED  {result.error}")

        _print_summary(results)

        if not all(r.success for r in results):
            sys.exit(1)

    finally:
        await scheduler.stop()
        close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BTC paper trading simulations")
    parser.add_argument(
        "scenario", nargs="?", choices=SCENARIOS,
        help="Single scenario to run (default: all)",
    )
    parser.add_argument("--equity", type=float, default=0.0,
                        help="Equity override in USD (default: fetch from Alpaca account)")
    args = parser.parse_args()

    scenarios = (args.scenario,) if args.scenario else SCENARIOS
    asyncio.run(main(scenarios, args.equity))
