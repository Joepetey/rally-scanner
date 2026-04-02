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

from test_helpers import TestRunner

from db import close_pool, init_pool, init_schema
from db.trading.positions import delete_position_meta, load_positions
from integrations.alpaca.account import get_account_equity, get_snapshots
from integrations.alpaca.entries import execute_entries
from integrations.alpaca.exits import execute_exit
from trading.positions import add_signal_positions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def run(t: TestRunner, ticker: str) -> bool:
    """Full production cycle: buy -> save position -> sell -> cleanup."""
    t.divider("1. Account connectivity")
    equity = await get_account_equity()
    print(f"  Account equity: ${equity:,.2f}")
    if not t.check(equity > 0, "equity is positive"):
        return False

    t.divider(f"2. Fetch live price for {ticker}")
    snapshots = await get_snapshots([ticker])
    if not t.check(ticker in snapshots, f"got snapshot for {ticker}"):
        return False
    price = snapshots[ticker]["price"]
    print(f"  {ticker} latest trade: ${price:.2f}")
    if not t.check(price > 0, "price is positive"):
        return False

    size = 0.02
    signal = {
        "ticker": ticker, "close": price, "entry_price": price, "size": size,
        "p_rally": 0.75, "comp_score": 0.5, "atr_pct": 0.015,
        "range_low": round(price * 0.97, 2), "date": str(date.today()),
    }
    qty_expected = int(equity * size / price)
    print(f"\n  Signal: size={size:.0%}  price=${price:.2f}  expected_qty={qty_expected}")
    if not t.check(qty_expected >= 1, "enough equity for 1 share"):
        return False

    t.divider("3. execute_entries (buy)")
    state = load_positions()
    if any(p["ticker"] == ticker for p in state.get("positions", [])):
        t.fail(f"{ticker} already in open positions -- skipping to avoid duplicate")
        return False

    buy_results = await execute_entries([signal], equity=equity)
    if not t.check(bool(buy_results), "execute_entries returned results"):
        return False
    result = buy_results[0]
    print(f"  success={result.success}  qty={result.qty}  order_id={result.order_id}  "
          f"fill_price={result.fill_price}  error={result.error}")
    if not t.check(result.success, "buy succeeded"):
        return False

    t.divider("4. add_signal_positions (save to DB)")
    signal["order_id"] = result.order_id
    state = load_positions()
    state = add_signal_positions(state, [signal])
    if not t.check(any(p["ticker"] == ticker for p in state.get("positions", [])),
                   f"position saved (DB has {len(state['positions'])} open)"):
        return False

    t.divider("5. execute_exit (sell)")
    exit_result = await execute_exit(ticker, trail_order_id=None)
    print(f"  success={exit_result.success}  already_closed={exit_result.already_closed}  "
          f"qty={exit_result.qty}  fill_price={exit_result.fill_price}")
    if not exit_result.success and not exit_result.already_closed:
        t.fail(f"exit failed: {exit_result.error}")

    t.divider("6. DB cleanup")
    delete_position_meta(ticker)
    state_after = load_positions()
    t.check(not any(p["ticker"] == ticker for p in state_after.get("positions", [])),
            "position removed from DB")

    return result.success and (exit_result.success or exit_result.already_closed)


async def main(ticker: str) -> None:
    t = TestRunner()
    init_pool()
    init_schema()
    try:
        await run(t, ticker)
    finally:
        close_pool()

    if not t.summary():
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL",
                        help="Ticker to test trade (default: AAPL)")
    args = parser.parse_args()
    asyncio.run(main(args.ticker))
