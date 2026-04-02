#!/usr/bin/env python
"""
Exit condition test suite — exercises take-profit, stop-loss, and all EOD
conditional exits through the real production code path.

Two groups of tests:
  1. Intraday exits (profit_target, stop_loss) — places a real paper trade,
     rigs stop/target to trigger against live price, runs the exact same
     check+exit logic as _check_price_alerts in the bot.
  2. EOD condition exits (trail_stop, time_stop, vol_exhaustion, stop) —
     inserts fake positions into DB, calls update_existing_positions with
     crafted price data, verifies the right exit reason fires and DB is clean.

Usage:
    .venv/bin/python scripts/test_exits.py
"""

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rally_ml.config import PARAMS
from test_helpers import TestRunner, cleanup, fake_position, price_result

from db import close_pool, init_pool, init_schema
from db.positions import load_position_meta, load_positions, save_position_meta
from integrations.alpaca.account import get_account_equity, get_snapshots
from integrations.alpaca.entries import execute_entries
from integrations.alpaca.exits import execute_exit
from trading.positions import add_signal_positions, async_close_position, update_existing_positions

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_TICKER = "AAPL"


async def intraday_exit_test(
    t: TestRunner, equity: float, live_price: float, exit_type: str,
) -> None:
    """Buy 1 share, rig stop/target to trigger, verify exit + DB cleanup."""
    t.divider(f"Intraday: {exit_type}", char="-")
    cleanup(_TICKER)

    signal = {
        "ticker": _TICKER,
        "close": live_price,
        "entry_price": live_price,
        "size": 0.02,
        "p_rally": 0.75,
        "comp_score": 0.5,
        "atr_pct": PARAMS.default_atr_pct,
        "range_low": round(live_price * 0.97, 2),
        "date": str(date.today()),
    }
    buy_results = await execute_entries([signal], equity=equity)
    if not buy_results or not buy_results[0].success:
        t.fail(f"buy failed: {buy_results[0].error if buy_results else 'no result'}")
        return
    t.ok(f"bought {buy_results[0].qty} shares at ~${live_price:.2f}")

    state = load_positions()
    state = add_signal_positions(state, [signal])

    pos = load_position_meta(_TICKER)
    if pos is None:
        t.fail("position not found in DB after buy")
        cleanup(_TICKER)
        return

    if exit_type == "profit_target":
        pos["target_price"] = round(live_price * 0.50, 2)
        pos["stop_price"] = round(live_price * 0.50, 2)
    else:
        pos["stop_price"] = round(live_price * 2.0, 2)
        pos["trailing_stop"] = 0.0
    save_position_meta(pos)

    snapshots = await get_snapshots([_TICKER])
    price = snapshots[_TICKER]["price"]

    stop = pos.get("stop_price", 0)
    target = pos.get("target_price", 0)
    trailing = pos.get("trailing_stop", 0)
    effective_stop = max(stop, trailing)

    triggered = (effective_stop > 0 and price <= effective_stop) or \
                (target > 0 and price >= target)
    reason = "profit_target" if target > 0 and price >= target else "stop"

    t.check(triggered,
            f"exit condition triggered (price=${price:.2f}, "
            f"stop=${effective_stop:.2f}, target=${target:.2f})")

    if not triggered:
        await execute_exit(_TICKER)
        cleanup(_TICKER)
        return

    result = await execute_exit(_TICKER, trail_order_id=pos.get("trail_order_id"))
    t.check(result.success or result.already_closed,
            f"execute_exit succeeded (fill=${result.fill_price})")

    fill = result.fill_price or price
    await async_close_position(_TICKER, fill, reason)
    t.check(load_position_meta(_TICKER) is None, "position removed from DB")

    if not result.success and not result.already_closed:
        cleanup(_TICKER)


def eod_condition_test(
    t: TestRunner,
    condition: str,
    price_override: float | None = None,
    pos_overrides: dict | None = None,
    price_extras: dict | None = None,
    ticker_prefix: str = "_TEST",
) -> None:
    """Insert fake position, trigger exit condition, verify reason + DB cleanup."""
    t.divider(f"EOD condition: {condition} [{ticker_prefix}]", char="-")

    test_ticker = f"{ticker_prefix}_{condition.upper()}"
    cleanup(test_ticker)

    base_price = 100.0
    pos = fake_position(test_ticker, base_price, **(pos_overrides or {}))
    close = price_override if price_override is not None else base_price

    state = {"positions": [dict(pos)], "closed_today": [], "last_updated": ""}
    result = price_result(test_ticker, close, **(price_extras or {}))
    updated = update_existing_positions(state, [result])

    closed = updated.get("closed_today", [])
    actual_reason = next(
        (p.get("exit_reason") for p in closed if p["ticker"] == test_ticker), None
    )

    t.check(any(p["ticker"] == test_ticker for p in closed),
            f"position closed (exit_reason={actual_reason!r})")
    t.check(actual_reason == condition,
            f"exit_reason == {condition!r} (got {actual_reason!r})")
    t.check(not any(p["ticker"] == test_ticker for p in updated.get("positions", [])),
            "position not in still_open")
    t.check(load_position_meta(test_ticker) is None, "position removed from DB")
    cleanup(test_ticker)


def run_eod_suite(t: TestRunner, ticker_prefix: str = "_TEST") -> None:
    """Run all 5 EOD exit condition tests."""
    p = PARAMS

    eod_condition_test(t, "stop", price_override=95.0, ticker_prefix=ticker_prefix)
    eod_condition_test(t, "profit_target", price_override=105.0, ticker_prefix=ticker_prefix)
    eod_condition_test(
        t, "trail_stop", price_override=96.5,
        pos_overrides={"bars_held": 2, "stop_price": 90.0, "trailing_stop": 97.0,
                       "target_price": 120.0},
        ticker_prefix=ticker_prefix,
    )
    eod_condition_test(
        t, "time_stop", price_override=100.5,
        pos_overrides={"bars_held": p.time_stop_bars, "stop_price": 90.0,
                       "target_price": 120.0, "trailing_stop": 88.0},
        ticker_prefix=ticker_prefix,
    )
    eod_condition_test(
        t, "vol_exhaustion", price_override=100.5,
        pos_overrides={"stop_price": 90.0, "target_price": 120.0, "trailing_stop": 88.0},
        price_extras={"rv_pctile": p.rv_exit_pct + 0.05},
        ticker_prefix=ticker_prefix,
    )


async def main() -> None:
    t = TestRunner()
    init_pool()
    init_schema()

    try:
        t.divider("Preflight: account + live price")
        equity = await get_account_equity()
        print(f"  equity=${equity:,.2f}")

        snapshots = await get_snapshots([_TICKER])
        if _TICKER not in snapshots:
            t.fail(f"no snapshot for {_TICKER}")
            return
        live_price = snapshots[_TICKER]["price"]
        print(f"  {_TICKER} live price=${live_price:.2f}")

        cleanup(_TICKER)
        await intraday_exit_test(t, equity, live_price, "profit_target")
        await intraday_exit_test(t, equity, live_price, "stop")
        run_eod_suite(t)
    finally:
        cleanup(_TICKER)
        close_pool()

    if not t.summary():
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
