#!/usr/bin/env python
"""
One-shot: fetch open positions from Alpaca and upsert them into system_positions,
computing stop/target/trailing levels from recent OHLCV data.

Level formulas (same as add_signal_positions):
  - stop_price    = 40-bar range low
  - atr           = 20-bar ATR
  - target_price  = entry + PARAMS.profit_atr_mult * atr
  - trailing_stop = entry - PARAMS.trailing_stop_atr_mult * atr
  - highest_close = max close over [entry_price .. today]

Skips tickers that already have stop_price set (already managed by the scanner).

Usage:
    DATABASE_URL=postgresql://... ALPACA_API_KEY=... ALPACA_SECRET_KEY=... \\
        python scripts/import_alpaca_positions.py

    # or with a local .env:
    python scripts/import_alpaca_positions.py
"""

import asyncio
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _compute_levels(ticker: str, entry_price: float) -> dict:
    """Fetch recent OHLCV and compute stop/target/trailing/ATR for a ticker."""
    import pandas as pd
    import yfinance as yf
    from rally_ml.config import PARAMS

    atr_period = PARAMS.atr_period       # 20
    range_period = PARAMS.range_period   # 40
    # Fetch enough bars to cover both periods + some buffer
    lookback_days = max(atr_period, range_period) * 2
    start = (date.today() - timedelta(days=lookback_days * 2)).isoformat()

    df = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close"]].dropna().sort_index()
    # Drop today's partial bar
    today = pd.Timestamp.now().normalize()
    if len(df) and df.index[-1] >= today:
        df = df.iloc[:-1]

    if len(df) < atr_period:
        # Not enough data — use fallback percentages
        atr_val = entry_price * PARAMS.default_atr_pct
        range_low = entry_price * (1 - PARAMS.fallback_stop_pct)
        print(f"    [WARN] {ticker}: not enough bars, using fallback levels")
    else:
        # ATR(20)
        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(atr_period, min_periods=atr_period).mean()
        atr_val = float(atr_series.iloc[-1])

        # 40-bar range low
        n = min(range_period, len(df))
        range_low = float(df["Low"].iloc[-n:].min())

    # Highest close at or above entry (approximation: look back range_period bars)
    recent_closes = df["Close"].iloc[-range_period:] if len(df) >= range_period else df["Close"]
    above_entry = recent_closes[recent_closes >= entry_price]
    highest_close = float(above_entry.max()) if len(above_entry) else entry_price

    stop_price   = round(range_low, 2)
    target_price = round(entry_price + PARAMS.profit_atr_mult * atr_val, 2)
    trailing_stop = round(
        max(entry_price - PARAMS.trailing_stop_atr_mult * atr_val, range_low),
        2,
    )

    return {
        "stop_price": stop_price,
        "target_price": target_price,
        "trailing_stop": trailing_stop,
        "highest_close": round(highest_close, 4),
        "atr": round(atr_val, 4),
    }


async def main() -> None:
    from db import init_pool, init_schema
    from db.positions import load_all_position_meta, save_position_meta
    from integrations.alpaca.executor import get_all_positions

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set.")
        sys.exit(1)
    if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.")
        sys.exit(1)

    print("Initializing DB pool and schema...")
    init_pool()
    init_schema()

    print("Fetching open positions from Alpaca...")
    alpaca_positions = await get_all_positions()

    if not alpaca_positions:
        print("No open positions found in Alpaca.")
        return

    print(f"Found {len(alpaca_positions)} open position(s) in Alpaca:\n")
    for p in alpaca_positions:
        print(
            f"  {p['ticker']:<8}  qty={p['qty']}  avg_entry=${p['avg_entry_price']:.2f}"
            f"  market_value=${p['market_value']:.2f}"
            f"  unrealized_pl=${p['unrealized_pl']:.2f}"
        )

    existing = {p["ticker"]: p for p in load_all_position_meta()}
    today = date.today().isoformat()

    imported, updated, skipped = 0, 0, 0
    for p in alpaca_positions:
        ticker = p["ticker"]
        entry_price = p["avg_entry_price"]
        meta = existing.get(ticker, {})

        # Compute levels only if stop not already set (e.g. managed by scanner)
        if meta.get("stop_price", 0) != 0:
            print(f"\n  {ticker}: levels already set — reusing existing.")
            levels = {
                "stop_price": meta["stop_price"],
                "target_price": meta.get("target_price", 0),
                "trailing_stop": meta.get("trailing_stop", 0),
                "highest_close": meta.get("highest_close", entry_price),
                "atr": meta.get("atr", 0),
            }
        else:
            print(f"\n  Computing levels for {ticker} (entry=${entry_price:.2f})...")
            levels = await asyncio.to_thread(_compute_levels, ticker, entry_price)
            print(
                f"    stop=${levels['stop_price']:.2f}  "
                f"target=${levels['target_price']:.2f}  "
                f"trail=${levels['trailing_stop']:.2f}  "
                f"atr=${levels['atr']:.4f}"
            )

        is_new = ticker not in existing
        pos = {
            **meta,
            "ticker": ticker,
            "entry_price": entry_price,
            "entry_date": meta.get("entry_date") or today,
            "qty": p["qty"],
            "size": meta.get("size", 0),
            "bars_held": meta.get("bars_held", 0),
            "order_id": meta.get("order_id"),
            "trail_order_id": meta.get("trail_order_id"),
            "p_rally": meta.get("p_rally", 0),
            **levels,
        }

        # Place trailing stop on Alpaca if not already present
        if not pos.get("trail_order_id"):
            from rally_ml.config import PARAMS

            from integrations.alpaca.executor import _trading_client, place_trailing_stop

            atr_pct = levels["atr"] / entry_price if levels["atr"] else PARAMS.default_atr_pct
            trail_pct = round(max(PARAMS.trailing_stop_atr_mult * atr_pct * 100, 1.0), 2)
            print(f"    Placing trailing stop on Alpaca ({trail_pct:.2f}% trail)...")
            trail_id = await place_trailing_stop(ticker, p["qty"], trail_pct)
            if trail_id:
                pos["trail_order_id"] = trail_id
                print(f"    Trailing stop placed: {trail_id}")
            else:
                # Shares already held by an existing order — look it up and adopt it
                try:
                    from alpaca.trading.enums import QueryOrderStatus
                    from alpaca.trading.requests import GetOrdersRequest
                    client = _trading_client()
                    all_orders = client.get_orders(filter=GetOrdersRequest(
                        status=QueryOrderStatus.ALL,
                        symbols=[ticker],
                    ))
                    _closed = {"orderstatus.filled", "orderstatus.canceled",
                               "orderstatus.expired", "orderstatus.replaced"}
                    sell_orders = [
                        o for o in all_orders
                        if str(o.side).lower() == "orderside.sell"
                        and str(o.status).lower() not in _closed
                    ]
                    if sell_orders:
                        trail_id = str(sell_orders[0].id)
                        pos["trail_order_id"] = trail_id
                        print(f"    Adopted existing sell order as trailing stop: {trail_id}")
                    else:
                        print(f"    [WARN] No existing sell order found for {ticker}")
                except Exception as e:
                    print(f"    [WARN] Could not look up existing order for {ticker}: {e}")
        else:
            print(f"    Trailing stop already present: {pos['trail_order_id']}")
            skipped += 1
            continue

        save_position_meta(pos)
        action = "imported" if is_new else "updated"
        print(f"    [{action.upper()}] {ticker}")
        if is_new:
            imported += 1
        else:
            updated += 1

    print(f"\nDone: {imported} imported, {updated} updated, {skipped} skipped.")


if __name__ == "__main__":
    asyncio.run(main())
