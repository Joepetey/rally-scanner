#!/usr/bin/env python
"""
Combined test suite — runs every manual test in sequence.

Sections:
  1. Preflight: account connectivity + live price
  2. Scanner: dry-run on AAPL/MSFT/SPY
  3. Full trade lifecycle: buy → save → sell
  4. Intraday exits: profit_target, stop (real paper trades)
  5. EOD exit conditions: stop, profit_target, trail_stop, time_stop, vol_exhaustion

Usage:
    .venv/bin/python scripts/test_all.py
    .venv/bin/python scripts/test_all.py --ticker MSFT
    .venv/bin/python scripts/test_all.py --skip-scanner

Requires: ALPACA_API_KEY, ALPACA_SECRET_KEY, DATABASE_URL in .env (or env).
"""

import argparse
import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import rally_ml.config as config
from rally_ml.config import PARAMS

from db import close_pool, init_pool, init_schema
from db.positions import (
    delete_position_meta,
    load_position_meta,
    load_positions,
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
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_passed = 0
_failed = 0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _divider(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print("═" * 64)


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
# Shared helpers
# ---------------------------------------------------------------------------

def _cleanup(ticker: str) -> None:
    delete_position_meta(ticker)


def _fake_position(ticker: str, price: float, **overrides) -> dict:
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
# Section 1 + 2: Preflight & Scanner
# ---------------------------------------------------------------------------

async def section_preflight(ticker: str) -> tuple[float, float] | None:
    """Returns (equity, live_price) or None if preflight fails."""
    _divider("1. Preflight — account + live price")

    equity = await get_account_equity()
    if not _assert(equity > 0, f"account equity ${equity:,.2f}"):
        return None

    snapshots = await get_snapshots([ticker])
    if not _assert(ticker in snapshots, f"got snapshot for {ticker}"):
        return None

    price = snapshots[ticker]["price"]
    if not _assert(price > 0, f"{ticker} live price ${price:.2f}"):
        return None

    is_crypto = (
        ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto"
    )
    if is_crypto:
        # Crypto supports fractional shares — just need qty > 0
        qty_expected = equity * 0.02 / price
        if not _assert(qty_expected > 0,
                       f"enough equity for fractional order (equity=${equity:,.0f}, price=${price:,.2f})"):
            return None
    else:
        qty_expected = int(equity * 0.02 / price)
        if not _assert(qty_expected >= 1,
                       f"enough equity for 1 share (equity=${equity:,.0f}, price=${price:.2f})"):
            return None

    return equity, price


def section_scanner(ticker: str, skip: bool) -> None:
    is_crypto = ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto"
    scan_tickers = [ticker] if is_crypto else ["AAPL", "MSFT", "SPY"]
    _divider(f"2. Scanner — dry-run on {' / '.join(scan_tickers)}")
    if skip:
        print("  [SKIP] --skip-scanner flag set")
        return

    from pipeline.scanner import scan_all
    try:
        results = scan_all(tickers=scan_tickers, config_name="conservative")
        signals = [r for r in results if r.get("signal")]
        ok_count = sum(1 for r in results if r.get("status") == "ok")
        _ok(f"scan completed — {ok_count}/{len(results)} ok, {len(signals)} signal(s)")
    except Exception as exc:
        _fail(f"scan raised exception: {exc}")


# ---------------------------------------------------------------------------
# Section 3: Full trade lifecycle
# ---------------------------------------------------------------------------

async def section_trade_lifecycle(ticker: str, equity: float, price: float) -> None:
    _divider("3. Full trade lifecycle — buy → save → sell")
    _cleanup(ticker)

    state = load_positions()
    if any(p["ticker"] == ticker for p in state.get("positions", [])):
        _fail(f"{ticker} already in open positions — cannot run lifecycle test")
        return

    signal = {
        "ticker": ticker,
        "close": price,
        "entry_price": price,
        "size": 0.02,
        "p_rally": 0.75,
        "comp_score": 0.5,
        "atr_pct": PARAMS.default_atr_pct,
        "range_low": round(price * 0.97, 2),
        "date": str(date.today()),
    }

    # Buy
    buy_results = await execute_entries([signal], equity=equity)
    if not _assert(bool(buy_results) and buy_results[0].success,
                   f"execute_entries succeeded (qty={buy_results[0].qty if buy_results else 0})"):
        _cleanup(ticker)
        return
    result = buy_results[0]

    # Save to DB
    signal["order_id"] = result.order_id
    state = load_positions()
    state = add_signal_positions(state, [signal])
    _assert(any(p["ticker"] == ticker for p in state.get("positions", [])),
            "position saved to DB")

    # Sell
    exit_result = await execute_exit(ticker, trail_order_id=None)
    _assert(exit_result.success or exit_result.already_closed,
            f"execute_exit succeeded (qty={exit_result.qty}, fill=${exit_result.fill_price})")

    # Cleanup
    delete_position_meta(ticker)
    state_after = load_positions()
    _assert(not any(p["ticker"] == ticker for p in state_after.get("positions", [])),
            "position removed from DB after cleanup")


# ---------------------------------------------------------------------------
# Section 4: Intraday exits
# ---------------------------------------------------------------------------

async def _intraday_exit(equity: float, live_price: float, exit_type: str, ticker: str) -> None:
    _divider(f"4. Intraday exit — {exit_type}")
    _cleanup(ticker)

    signal = {
        "ticker": ticker,
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
    if not _assert(bool(buy_results) and buy_results[0].success,
                   f"bought {buy_results[0].qty if buy_results else 0} share(s) at ~${live_price:.2f}"):
        _cleanup(ticker)
        return

    state = load_positions()
    state = add_signal_positions(state, [signal])

    pos = load_position_meta(ticker)
    if not _assert(pos is not None, "position saved to DB"):
        _cleanup(ticker)
        return

    if exit_type == "profit_target":
        pos["target_price"] = round(live_price * 0.50, 2)
        pos["stop_price"] = round(live_price * 0.50, 2)
    else:  # stop
        pos["stop_price"] = round(live_price * 2.0, 2)
        pos["trailing_stop"] = 0.0
    save_position_meta(pos)

    snapshots = await get_snapshots([ticker])
    current_price = snapshots[ticker]["price"]

    stop = pos.get("stop_price", 0)
    target = pos.get("target_price", 0)
    trailing = pos.get("trailing_stop", 0)
    effective_stop = max(stop, trailing)

    triggered = (effective_stop > 0 and current_price <= effective_stop) or \
                (target > 0 and current_price >= target)
    reason = "profit_target" if target > 0 and current_price >= target else "stop"

    _assert(triggered,
            f"exit condition triggered (price=${current_price:.2f}, "
            f"stop=${effective_stop:.2f}, target=${target:.2f})")

    if not triggered:
        await execute_exit(ticker)
        _cleanup(ticker)
        return

    exit_result = await execute_exit(ticker, trail_order_id=pos.get("trail_order_id"))
    _assert(exit_result.success or exit_result.already_closed,
            f"execute_exit succeeded (fill=${exit_result.fill_price})")

    fill = exit_result.fill_price or current_price
    await async_close_position(ticker, fill, reason)

    _assert(load_position_meta(ticker) is None, "position removed from DB")

    if not exit_result.success and not exit_result.already_closed:
        _cleanup(ticker)


async def section_intraday_exits(ticker: str, equity: float, price: float) -> None:
    await _intraday_exit(equity, price, "profit_target", ticker)
    await _intraday_exit(equity, price, "stop", ticker)


# ---------------------------------------------------------------------------
# Section 5: EOD exit conditions
# ---------------------------------------------------------------------------

def _eod_condition_test(
    condition: str,
    price_override: float | None = None,
    pos_overrides: dict | None = None,
    price_extras: dict | None = None,
    ticker_prefix: str = "_TEST",
) -> None:
    _divider(f"5. EOD condition — {condition} [{ticker_prefix}]")

    test_ticker = f"{ticker_prefix}_{condition.upper()}"
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
    actual_reason = next(
        (p.get("exit_reason") for p in closed if p["ticker"] == test_ticker), None
    )

    _assert(any(p["ticker"] == test_ticker for p in closed),
            f"position closed (exit_reason={actual_reason!r})")
    _assert(actual_reason == condition,
            f"exit_reason == {condition!r} (got {actual_reason!r})")
    _assert(
        not any(p["ticker"] == test_ticker for p in updated.get("positions", [])),
        "position not in still_open",
    )
    _assert(load_position_meta(test_ticker) is None, "position removed from DB")

    _cleanup(test_ticker)


def section_eod_exits() -> None:
    p = PARAMS

    _eod_condition_test("stop", price_override=95.0)

    _eod_condition_test("profit_target", price_override=105.0)

    _eod_condition_test(
        "trail_stop",
        price_override=96.5,
        pos_overrides={
            "bars_held": 2,
            "stop_price": 90.0,
            "trailing_stop": 97.0,
            "target_price": 120.0,
        },
    )

    _eod_condition_test(
        "time_stop",
        price_override=100.5,
        pos_overrides={
            "bars_held": p.time_stop_bars,
            "stop_price": 90.0,
            "target_price": 120.0,
            "trailing_stop": 88.0,
        },
    )

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


def section_eod_exits_btc() -> None:
    """Regression: all 5 EOD exit conditions with BTC-keyed fake positions.

    Verifies no false exits from missing/zero fields (rv_pctile=0 must NOT
    trigger vol_exhaustion) and that bars_held logic works identically.
    """
    p = PARAMS
    prefix = "_TEST_BTC"

    _eod_condition_test("stop", price_override=95.0, ticker_prefix=prefix)

    _eod_condition_test("profit_target", price_override=105.0, ticker_prefix=prefix)

    _eod_condition_test(
        "trail_stop",
        price_override=96.5,
        pos_overrides={
            "bars_held": 2,
            "stop_price": 90.0,
            "trailing_stop": 97.0,
            "target_price": 120.0,
        },
        ticker_prefix=prefix,
    )

    _eod_condition_test(
        "time_stop",
        price_override=100.5,
        pos_overrides={
            "bars_held": p.time_stop_bars,
            "stop_price": 90.0,
            "target_price": 120.0,
            "trailing_stop": 88.0,
        },
        ticker_prefix=prefix,
    )

    _eod_condition_test(
        "vol_exhaustion",
        price_override=100.5,
        pos_overrides={
            "stop_price": 90.0,
            "target_price": 120.0,
            "trailing_stop": 88.0,
        },
        price_extras={"rv_pctile": p.rv_exit_pct + 0.05},
        ticker_prefix=prefix,
    )

    # Regression: rv_pctile=0 (missing in crypto scan) must NOT trigger vol_exhaustion
    _divider("5b. Crypto regression — rv_pctile=0 does not trigger vol_exhaustion")
    safe_ticker = "_TEST_BTC_VOL_SAFE"
    _cleanup(safe_ticker)
    base_price = 100.0
    pos = _fake_position(safe_ticker, base_price, stop_price=90.0, target_price=120.0)
    state = {"positions": [dict(pos)], "closed_today": [], "last_updated": ""}
    result = _price_result(safe_ticker, 100.5, rv_pctile=0.0)
    updated = update_existing_positions(state, [result])
    closed = updated.get("closed_today", [])
    _assert(
        not any(p["ticker"] == safe_ticker for p in closed),
        "rv_pctile=0.0 does NOT trigger vol_exhaustion (no false exit)",
    )
    _cleanup(safe_ticker)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(ticker: str, skip_scanner: bool) -> None:
    init_pool()
    init_schema()

    try:
        preflight = await section_preflight(ticker)
        section_scanner(ticker=ticker, skip=skip_scanner)

        if preflight is None:
            _fail("preflight failed — skipping live tests")
        else:
            equity, price = preflight
            await section_trade_lifecycle(ticker, equity, price)
            await section_intraday_exits(ticker, equity, price)

        section_eod_exits()
        section_eod_exits_btc()

    finally:
        close_pool()

    total = _passed + _failed
    print(f"\n{'═' * 64}")
    if _failed == 0:
        print(f"  ALL TESTS PASSED — {_passed}/{total}")
    else:
        print(f"  {_failed} FAILED — {_passed}/{total} passed")
    print("═" * 64)

    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined trading system test suite")
    parser.add_argument("--ticker", default="AAPL",
                        help="Ticker for live/intraday tests (default: AAPL)")
    parser.add_argument("--skip-scanner", action="store_true",
                        help="Skip scanner dry-run (use if models/ is empty)")
    args = parser.parse_args()
    asyncio.run(main(args.ticker, args.skip_scanner))
