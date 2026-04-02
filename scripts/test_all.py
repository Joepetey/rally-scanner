#!/usr/bin/env python
"""
Combined test suite — runs every manual test in sequence.

Sections:
  1. Preflight: account connectivity + live price
  2. Scanner: dry-run on AAPL/MSFT/SPY
  3. Full trade lifecycle: buy -> save -> sell
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
from test_exits import intraday_exit_test, run_eod_suite
from test_helpers import TestRunner, cleanup, fake_position, price_result

from db import close_pool, init_pool, init_schema
from db.positions import delete_position_meta, load_positions
from integrations.alpaca.account import get_account_equity, get_snapshots
from integrations.alpaca.entries import execute_entries
from integrations.alpaca.exits import execute_exit
from trading.positions import add_signal_positions, update_existing_positions

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def section_preflight(t: TestRunner, ticker: str) -> tuple[float, float] | None:
    """Returns (equity, live_price) or None if preflight fails."""
    t.divider("1. Preflight -- account + live price")

    equity = await get_account_equity()
    if not t.check(equity > 0, f"account equity ${equity:,.2f}"):
        return None

    snapshots = await get_snapshots([ticker])
    if not t.check(ticker in snapshots, f"got snapshot for {ticker}"):
        return None

    price = snapshots[ticker]["price"]
    if not t.check(price > 0, f"{ticker} live price ${price:.2f}"):
        return None

    is_crypto = ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto"
    if is_crypto:
        qty_expected = equity * 0.02 / price
        msg = f"enough equity for fractional order (equity=${equity:,.0f}, price=${price:,.2f})"
        if not t.check(qty_expected > 0, msg):
            return None
    else:
        qty_expected = int(equity * 0.02 / price)
        if not t.check(qty_expected >= 1,
                       f"enough equity for 1 share (equity=${equity:,.0f}, price=${price:.2f})"):
            return None

    return equity, price


def section_scanner(t: TestRunner, ticker: str, skip: bool) -> None:
    is_crypto = ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto"
    scan_tickers = [ticker] if is_crypto else ["AAPL", "MSFT", "SPY"]
    t.divider(f"2. Scanner -- dry-run on {' / '.join(scan_tickers)}")
    if skip:
        print("  [SKIP] --skip-scanner flag set")
        return

    from pipeline.scanner import scan_all
    try:
        results = scan_all(tickers=scan_tickers, config_name="conservative")
        signals = [r for r in results if r.get("signal")]
        ok_count = sum(1 for r in results if r.get("status") == "ok")
        t.ok(f"scan completed -- {ok_count}/{len(results)} ok, {len(signals)} signal(s)")
    except Exception as exc:
        t.fail(f"scan raised exception: {exc}")


async def section_trade_lifecycle(t: TestRunner, ticker: str, equity: float, price: float) -> None:
    t.divider("3. Full trade lifecycle -- buy -> save -> sell")
    cleanup(ticker)

    state = load_positions()
    if any(p["ticker"] == ticker for p in state.get("positions", [])):
        t.fail(f"{ticker} already in open positions -- cannot run lifecycle test")
        return

    signal = {
        "ticker": ticker, "close": price, "entry_price": price, "size": 0.02,
        "p_rally": 0.75, "comp_score": 0.5, "atr_pct": PARAMS.default_atr_pct,
        "range_low": round(price * 0.97, 2), "date": str(date.today()),
    }

    buy_results = await execute_entries([signal], equity=equity)
    if not t.check(bool(buy_results) and buy_results[0].success,
                   f"execute_entries succeeded (qty={buy_results[0].qty if buy_results else 0})"):
        cleanup(ticker)
        return
    signal["order_id"] = buy_results[0].order_id

    state = load_positions()
    state = add_signal_positions(state, [signal])
    t.check(any(p["ticker"] == ticker for p in state.get("positions", [])),
            "position saved to DB")

    exit_result = await execute_exit(ticker, trail_order_id=None)
    t.check(exit_result.success or exit_result.already_closed,
            f"execute_exit succeeded (qty={exit_result.qty}, fill=${exit_result.fill_price})")

    delete_position_meta(ticker)
    state_after = load_positions()
    t.check(not any(p["ticker"] == ticker for p in state_after.get("positions", [])),
            "position removed from DB after cleanup")


def section_eod_btc_regression(t: TestRunner) -> None:
    """Regression: rv_pctile=0 (missing in crypto scan) must NOT trigger vol_exhaustion."""
    t.divider("5b. Crypto regression -- rv_pctile=0 does not trigger vol_exhaustion", char="-")
    safe_ticker = "_TEST_BTC_VOL_SAFE"
    cleanup(safe_ticker)
    fake_position(safe_ticker, 100.0, stop_price=90.0, target_price=120.0)
    state = {"positions": [fake_position(safe_ticker, 100.0, stop_price=90.0, target_price=120.0)],
             "closed_today": [], "last_updated": ""}
    result = price_result(safe_ticker, 100.5, rv_pctile=0.0)
    updated = update_existing_positions(state, [result])
    closed = updated.get("closed_today", [])
    t.check(not any(p["ticker"] == safe_ticker for p in closed),
            "rv_pctile=0.0 does NOT trigger vol_exhaustion (no false exit)")
    cleanup(safe_ticker)


async def main(ticker: str, skip_scanner: bool) -> None:
    t = TestRunner()
    init_pool()
    init_schema()

    try:
        preflight = await section_preflight(t, ticker)
        section_scanner(t, ticker=ticker, skip=skip_scanner)

        if preflight is None:
            t.fail("preflight failed -- skipping live tests")
        else:
            equity, price = preflight
            await section_trade_lifecycle(t, ticker, equity, price)
            await intraday_exit_test(t, equity, price, "profit_target")
            await intraday_exit_test(t, equity, price, "stop")

        run_eod_suite(t)
        run_eod_suite(t, ticker_prefix="_TEST_BTC")
        section_eod_btc_regression(t)
    finally:
        close_pool()

    if not t.summary():
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined trading system test suite")
    parser.add_argument("--ticker", default="AAPL",
                        help="Ticker for live/intraday tests (default: AAPL)")
    parser.add_argument("--skip-scanner", action="store_true",
                        help="Skip scanner dry-run (use if models/ is empty)")
    args = parser.parse_args()
    asyncio.run(main(args.ticker, args.skip_scanner))
