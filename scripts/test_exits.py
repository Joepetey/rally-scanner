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

from config import PARAMS
from db import close_pool, init_pool, init_schema
from db.positions import (
    delete_position_meta,
    load_all_position_meta,
    load_position_meta,
    load_positions,
    record_closed_position,
    save_position_meta,
)
from integrations.alpaca.executor import (
    execute_entries,
    execute_exit,
    get_account_equity,
    get_snapshots,
)
from trading.positions import add_signal_positions, async_close_position, update_existing_positions

logging.basicConfig(
    level=logging.WARNING,  # suppress noisy library logs
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_exits")

_passed = 0
_failed = 0
_TICKER = "AAPL"


def _divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def _ok(msg: str = "") -> None:
    global _passed
    _passed += 1
    print(f"  [PASS] {msg}")


def _fail(msg: str = "") -> None:
    global _failed
    _failed += 1
    print(f"  [FAIL] {msg}")


def _assert(cond: bool, msg: str) -> bool:
    if cond:
        _ok(msg)
    else:
        _fail(msg)
    return cond


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup(ticker: str) -> None:
    """Remove any test artifacts from DB — safe to call even if nothing exists."""
    delete_position_meta(ticker)


def _fake_position(ticker: str, price: float, **overrides) -> dict:
    """Build a minimal position dict and save it to DB."""
    atr_val = price * PARAMS.default_atr_pct
    pos = {
        "ticker": ticker,
        "entry_price": price,
        "entry_date": str(date.today()),
        "stop_price": round(price * 0.97, 2),
        "target_price": round(price + PARAMS.profit_atr_mult * atr_val, 2),
        "trailing_stop": round(price - PARAMS.trailing_stop_atr_mult * atr_val, 2),
        "highest_close": price,
        "atr": round(atr_val, 4),
        "bars_held": 0,
        "size": 0.02,
        "qty": 1,
        "order_id": None,
        "trail_order_id": None,
        "p_rally": 0.65,
        "status": "open",
    }
    pos.update(overrides)
    save_position_meta(pos)
    return pos


def _price_result(ticker: str, close: float, **extras) -> dict:
    """Minimal scanner result dict for update_existing_positions."""
    return {
        "ticker": ticker,
        "status": "ok",
        "close": close,
        "date": str(date.today()),
        "atr": close * PARAMS.default_atr_pct,
        "rv_pctile": 0.0,
        **extras,
    }


# ---------------------------------------------------------------------------
# Intraday exit tests (real Alpaca paper trades)
# ---------------------------------------------------------------------------

async def _intraday_exit_test(
    equity: float,
    live_price: float,
    exit_type: str,  # "profit_target" | "stop"
) -> None:
    """
    Buy 1 share, rig the stop or target to trigger immediately against live
    price, run the exact bot check logic, verify Alpaca exit + DB cleanup.
    """
    _divider(f"Intraday: {exit_type}")
    _cleanup(_TICKER)

    # --- Buy ---
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
        _fail(f"buy failed: {buy_results[0].error if buy_results else 'no result'}")
        return
    _ok(f"bought {buy_results[0].qty} shares at ~${live_price:.2f}")

    # Save position with rigged stop/target
    state = load_positions()
    state = add_signal_positions(state, [signal])

    pos = load_position_meta(_TICKER)
    if pos is None:
        _fail("position not found in DB after buy")
        _cleanup(_TICKER)
        return

    if exit_type == "profit_target":
        # Set target well below current price → price >= target triggers immediately
        pos["target_price"] = round(live_price * 0.50, 2)
        pos["stop_price"] = round(live_price * 0.50, 2)  # keep stop far below target
    else:  # stop
        # Set stop well above current price → price <= stop triggers immediately
        pos["stop_price"] = round(live_price * 2.0, 2)
        pos["trailing_stop"] = 0.0
    save_position_meta(pos)

    # Fetch live price (exactly as the bot does)
    snapshots = await get_snapshots([_TICKER])
    price = snapshots[_TICKER]["price"]

    stop = pos.get("stop_price", 0)
    target = pos.get("target_price", 0)
    trailing = pos.get("trailing_stop", 0)
    effective_stop = max(stop, trailing)

    triggered = False
    reason = None
    if effective_stop > 0 and price <= effective_stop:
        triggered = True
        reason = "stop"
    elif target > 0 and price >= target:
        triggered = True
        reason = "profit_target"

    _assert(triggered, f"exit condition triggered (price=${price:.2f}, "
            f"stop=${effective_stop:.2f}, target=${target:.2f})")

    if not triggered:
        await execute_exit(_TICKER)
        _cleanup(_TICKER)
        return

    # --- Execute exit (exact bot path) ---
    result = await execute_exit(_TICKER, trail_order_id=pos.get("trail_order_id"))
    _assert(result.success or result.already_closed,
            f"execute_exit succeeded (qty={result.qty}, fill={result.fill_price})")

    fill = result.fill_price or price
    await async_close_position(_TICKER, fill, reason)

    pos_after = load_position_meta(_TICKER)
    _assert(pos_after is None, "position removed from system_positions")

    _assert(result.success, f"Alpaca order result.success={result.success}")
    if not result.success and not result.already_closed:
        _cleanup(_TICKER)


# ---------------------------------------------------------------------------
# EOD condition tests (pure DB + logic, no Alpaca orders)
# ---------------------------------------------------------------------------

def _eod_condition_test(
    condition: str,
    price_override: float | None = None,
    pos_overrides: dict | None = None,
    price_extras: dict | None = None,
) -> None:
    """
    Insert a fake position, craft price data to trigger the given condition,
    run update_existing_positions, and verify the exit reason + DB cleanup.
    """
    _divider(f"EOD condition: {condition}")

    test_ticker = f"_TEST_{condition.upper()}"
    _cleanup(test_ticker)

    base_price = 100.0
    pos = _fake_position(test_ticker, base_price, **(pos_overrides or {}))
    close = price_override if price_override is not None else base_price

    state = {
        "positions": [dict(pos)],
        "closed_today": [],
        "last_updated": "",
    }
    result = _price_result(test_ticker, close, **(price_extras or {}))
    updated = update_existing_positions(state, [result])

    closed = updated.get("closed_today", [])
    still_open = updated.get("positions", [])

    triggered = any(p["ticker"] == test_ticker for p in closed)
    actual_reason = next(
        (p.get("exit_reason") for p in closed if p["ticker"] == test_ticker), None
    )

    _assert(triggered, f"position closed (exit_reason={actual_reason!r})")
    _assert(actual_reason == condition,
            f"exit_reason == {condition!r} (got {actual_reason!r})")
    _assert(
        not any(p["ticker"] == test_ticker for p in still_open),
        "position not in still_open",
    )

    # DB should already be cleaned up by update_existing_positions
    db_pos = load_position_meta(test_ticker)
    _assert(db_pos is None, "position removed from DB")

    _cleanup(test_ticker)  # safety net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    init_pool()
    init_schema()

    try:
        # Fetch shared context for intraday tests
        _divider("Preflight: account + live price")
        equity = await get_account_equity()
        print(f"  equity=${equity:,.2f}")

        snapshots = await get_snapshots([_TICKER])
        if _TICKER not in snapshots:
            print(f"  [FAIL] no snapshot for {_TICKER}")
            return
        live_price = snapshots[_TICKER]["price"]
        print(f"  {_TICKER} live price=${live_price:.2f}")

        # Ensure clean state
        _cleanup(_TICKER)

        # ── Intraday exits ──────────────────────────────────────────────────
        await _intraday_exit_test(equity, live_price, "profit_target")
        await _intraday_exit_test(equity, live_price, "stop")

        # ── EOD condition exits ─────────────────────────────────────────────
        p = PARAMS

        # stop: close <= stop_price
        _eod_condition_test(
            "stop",
            price_override=95.0,           # below default stop_price (97.0)
        )

        # profit_target: close >= target_price
        # default target = 100 + 1.0 * (100 * 0.02) = 102.0
        _eod_condition_test(
            "profit_target",
            price_override=105.0,
        )

        # trail_stop: bars_held >= 2 and close < trailing_stop
        # stop_price must be below close so it doesn't fire first
        _eod_condition_test(
            "trail_stop",
            price_override=96.5,
            pos_overrides={
                "bars_held": 2,
                "stop_price": 90.0,     # below close → doesn't trigger stop
                "trailing_stop": 97.0,  # above close → triggers trail_stop
                "target_price": 120.0,
            },
        )

        # time_stop: bars_held >= time_stop_bars (3)
        # use a price that doesn't trigger any other condition
        _eod_condition_test(
            "time_stop",
            price_override=100.5,          # no stop/target/trail breach
            pos_overrides={
                "bars_held": p.time_stop_bars,
                "stop_price": 90.0,
                "target_price": 120.0,
                "trailing_stop": 88.0,
            },
        )

        # vol_exhaustion: rv_pctile > rv_exit_pct (0.80)
        _eod_condition_test(
            "vol_exhaustion",
            price_override=100.5,
            pos_overrides={
                "stop_price": 90.0,
                "target_price": 120.0,
                "trailing_stop": 88.0,
            },
            price_extras={"rv_pctile": p.rv_exit_pct + 0.05},
        )

    finally:
        _cleanup(_TICKER)
        close_pool()

    # ── Summary ─────────────────────────────────────────────────────────────
    total = _passed + _failed
    print(f"\n{'=' * 60}")
    print(f"  Results: {_passed}/{total} passed", end="")
    if _failed:
        print(f"  ({_failed} FAILED)")
    else:
        print("  — all exit conditions work correctly")
    print("=" * 60)

    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
