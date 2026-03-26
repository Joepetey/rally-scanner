#!/usr/bin/env python
"""
End-to-end paper trade test — exercises the full production code path.

Runs a buy + immediate sell through execute_entries / execute_exit, including
all DB reads (exposure check, position state) and DB writes (position save,
queue cleanup). Uses real Alpaca paper trading credentials.

Usage:
    python scripts/test_trade.py [--ticker AAPL]

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

from db import close_pool, init_pool, init_schema
from db.positions import delete_position_meta, load_positions, save_position_meta
from integrations.alpaca.executor import (
    execute_entries,
    execute_exit,
    get_account_equity,
    get_snapshots,
)
from trading.positions import add_signal_positions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_trade")

STEP_OK = "  [OK]"
STEP_FAIL = "  [FAIL]"


def _divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


async def run(ticker: str) -> bool:
    """Full production cycle: buy → save position → sell → cleanup.

    Returns True if all steps succeeded.
    """
    _divider("1. Account connectivity")
    equity = await get_account_equity()
    print(f"  Account equity: ${equity:,.2f}")
    if equity <= 0:
        print(STEP_FAIL, "equity is zero — check API keys / paper account")
        return False
    print(STEP_OK)

    _divider(f"2. Fetch live price for {ticker}")
    snapshots = await get_snapshots([ticker])
    if ticker not in snapshots:
        print(STEP_FAIL, f"no snapshot returned — is {ticker!r} in config.ASSETS?")
        return False
    price = snapshots[ticker]["price"]
    print(f"  {ticker} latest trade: ${price:.2f}")
    if price <= 0:
        print(STEP_FAIL, "price is zero")
        return False
    print(STEP_OK)

    # Build a minimal signal dict — same shape as the real scanner output.
    # size=0.02 (2% of equity) → should yield at least 1 share for >$50 accounts.
    size = 0.02
    signal = {
        "ticker": ticker,
        "close": price,
        "entry_price": price,
        "size": size,
        "p_rally": 0.75,          # realistic value; used for rotation logic
        "comp_score": 0.5,
        "atr_pct": 0.015,
        "range_low": round(price * 0.97, 2),
        "date": str(date.today()),
    }
    qty_expected = int(equity * size / price)
    print(f"\n  Signal: size={size:.0%}  price=${price:.2f}  "
          f"expected_qty={qty_expected}")
    if qty_expected < 1:
        print(STEP_FAIL,
              f"not enough equity (${equity:,.0f}) for 1 share at ${price:.2f} "
              f"with size={size:.0%}")
        return False

    _divider("3. execute_entries (buy)")
    state = load_positions()
    if any(p["ticker"] == ticker for p in state.get("positions", [])):
        print(f"  WARNING: {ticker} already in open positions — "
              f"skipping to avoid duplicate")
        return False

    buy_results = await execute_entries([signal], equity=equity)
    if not buy_results:
        print(STEP_FAIL, "execute_entries returned empty results")
        return False

    result = buy_results[0]
    print(f"  success={result.success}  skipped={result.skipped}  "
          f"qty={result.qty}  order_id={result.order_id}  "
          f"fill_price={result.fill_price}  error={result.error}")

    if not result.success:
        print(STEP_FAIL, f"buy failed: {result.error}")
        return False
    print(STEP_OK)

    _divider("4. add_signal_positions (save to DB)")
    # Mirror exactly what the bot does after execute_entries succeeds.
    signal["order_id"] = result.order_id
    state = load_positions()
    state = add_signal_positions(state, [signal])
    saved = any(p["ticker"] == ticker for p in state.get("positions", []))
    if not saved:
        print(STEP_FAIL, "position not found in DB after add_signal_positions")
        return False
    print(STEP_OK, f"position saved  (DB has {len(state['positions'])} open)")

    _divider("5. execute_exit (sell)")
    exit_result = await execute_exit(ticker, trail_order_id=None)
    print(f"  success={exit_result.success}  already_closed={exit_result.already_closed}  "
          f"qty={exit_result.qty}  fill_price={exit_result.fill_price}  "
          f"error={exit_result.error}")

    if not exit_result.success and not exit_result.already_closed:
        print(STEP_FAIL, f"exit failed: {exit_result.error}")
        # Still clean up DB even if Alpaca exit failed
    else:
        print(STEP_OK)

    _divider("6. DB cleanup")
    delete_position_meta(ticker)
    state_after = load_positions()
    still_there = any(p["ticker"] == ticker for p in state_after.get("positions", []))
    if still_there:
        print(STEP_FAIL, f"{ticker} still in DB after delete")
        return False
    print(STEP_OK, "position removed from DB")

    all_ok = result.success and (exit_result.success or exit_result.already_closed)
    return all_ok


async def main(ticker: str) -> None:
    init_pool()
    init_schema()
    try:
        ok = await run(ticker)
    finally:
        close_pool()

    _divider("RESULT")
    if ok:
        print("  ALL STEPS PASSED — paper trade cycle works end-to-end\n")
    else:
        print("  ONE OR MORE STEPS FAILED — see output above\n")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL",
                        help="Ticker to test trade (default: AAPL)")
    args = parser.parse_args()
    asyncio.run(main(args.ticker))
