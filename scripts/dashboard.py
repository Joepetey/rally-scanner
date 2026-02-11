#!/usr/bin/env python
"""
Terminal dashboard for market-rally portfolio state.

Usage:
    python scripts/dashboard.py              # show cached state
    python scripts/dashboard.py --live       # fetch live prices for open positions
    python scripts/dashboard.py --journal 20 # show last 20 closed trades
    python scripts/dashboard.py --equity 90  # show 90-day equity history
"""

import argparse
import sys
from datetime import datetime

from rally.positions import load_positions
from rally.portfolio import load_equity_history, load_trade_journal
from rally.persistence import load_manifest


W = 80


def _print_open_positions(positions: list[dict], live: bool = False) -> None:
    """Print open positions table."""
    if live and positions:
        try:
            import yfinance as yf
            tickers = [p["ticker"] for p in positions]
            data = yf.download(tickers, period="1d", progress=False)
            if len(tickers) == 1:
                price = float(data["Close"].iloc[-1])
                positions[0]["current_price"] = price
            else:
                for p in positions:
                    try:
                        price = float(data[p["ticker"]]["Close"].iloc[-1])
                        p["current_price"] = price
                        p["unrealized_pnl_pct"] = (price / p["entry_price"] - 1) * 100
                    except (KeyError, IndexError):
                        pass
        except Exception:
            pass

    print(f"\n  OPEN POSITIONS ({len(positions)})")
    print(f"  {'='*W}")

    if not positions:
        print("  (none)")
        return

    print(f"  {'Ticker':<8} {'Entry':>9} {'Current':>9} {'PnL%':>8} "
          f"{'Stop':>9} {'Target':>9} {'Bars':>5} {'Size':>6}")
    print(f"  {'-'*W}")

    total_exposure = 0
    total_weighted_pnl = 0

    for p in positions:
        pnl = p.get("unrealized_pnl_pct", 0)
        size = p.get("size", 0)
        total_exposure += size
        total_weighted_pnl += pnl * size
        sign = "+" if pnl >= 0 else ""
        print(f"  {p.get('ticker', '?'):<8} "
              f"${p.get('entry_price', 0):>8.2f} "
              f"${p.get('current_price', 0):>8.2f} "
              f"{sign}{pnl:>6.2f}% "
              f"${p.get('stop_price', 0):>8.2f} "
              f"${p.get('target_price', 0):>8.2f} "
              f"{p.get('bars_held', 0):>5d} "
              f"{size:>5.1%}")

    print(f"  {'-'*W}")
    sign = "+" if total_weighted_pnl >= 0 else ""
    print(f"  Portfolio: {len(positions)} positions, "
          f"{total_exposure:.1%} exposure, "
          f"{sign}{total_weighted_pnl:.2f}% weighted unrealized")


def _print_closed_trades(trades: list[dict]) -> None:
    """Print recent closed trades."""
    print(f"\n  RECENT CLOSED TRADES ({len(trades)})")
    print(f"  {'='*W}")

    if not trades:
        print("  (none)")
        return

    print(f"  {'Date':<12} {'Ticker':<8} {'PnL%':>8} {'Reason':<16} {'Bars':>5} {'Size':>6}")
    print(f"  {'-'*W}")

    for t in reversed(trades):
        pnl = float(t.get("realized_pnl_pct", 0))
        sign = "+" if pnl >= 0 else ""
        print(f"  {t.get('exit_date', '?'):<12} "
              f"{t.get('ticker', '?'):<8} "
              f"{sign}{pnl:>6.2f}% "
              f"{t.get('exit_reason', '?'):<16} "
              f"{t.get('bars_held', '?'):>5} "
              f"{float(t.get('size', 0)):>5.1%}")


def _print_equity_history(history: list[dict]) -> None:
    """Print equity history summary."""
    print(f"\n  EQUITY HISTORY ({len(history)} days)")
    print(f"  {'='*W}")

    if not history:
        print("  (no data yet — run orchestrator scan to start collecting)")
        return

    print(f"  {'Date':<12} {'Pos':>4} {'Exposure':>9} {'UnrlzPnL':>10} {'Signals':>8} {'Scanned':>8}")
    print(f"  {'-'*W}")

    for row in history[-20:]:  # last 20 days
        print(f"  {row.get('date', '?'):<12} "
              f"{row.get('n_positions', 0):>4} "
              f"{float(row.get('total_exposure', 0)):>8.1%} "
              f"{float(row.get('total_unrealized_pnl_pct', 0)):>+9.4f}% "
              f"{row.get('n_signals_today', 0):>8} "
              f"{row.get('n_scanned', 0):>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Market Rally dashboard")
    parser.add_argument("--live", action="store_true",
                        help="Fetch live prices for open positions")
    parser.add_argument("--journal", type=int, default=10,
                        help="Number of recent closed trades to show (default: 10)")
    parser.add_argument("--equity", type=int, default=30,
                        help="Days of equity history to show (default: 30)")
    args = parser.parse_args()

    # Load data
    state = load_positions()
    manifest = load_manifest()

    print(f"\n{'='*W}")
    print(f"  MARKET RALLY DASHBOARD")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
          f"{len(manifest)} trained models")
    print(f"{'='*W}")

    # Open positions
    _print_open_positions(state.get("positions", []), live=args.live)

    # Closed today
    closed_today = state.get("closed_today", [])
    if closed_today:
        print(f"\n  CLOSED TODAY ({len(closed_today)})")
        print(f"  {'='*W}")
        for c in closed_today:
            pnl = c.get("realized_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            print(f"  {c.get('ticker', '?'):<8} {c.get('exit_reason', '?'):<16} "
                  f"PnL: {sign}{pnl:.2f}%  ({c.get('bars_held', 0)} bars)")

    # Recent closed trades (from journal)
    trades = load_trade_journal(limit=args.journal)
    _print_closed_trades(trades)

    # Equity history
    history = load_equity_history(days=args.equity)
    _print_equity_history(history)

    print(f"\n{'='*W}\n")


if __name__ == "__main__":
    main()
