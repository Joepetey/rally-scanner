"""
Position state tracking — persists open positions across scanner runs.

Storage: models/positions.json
"""

import json
from datetime import datetime
from pathlib import Path

from config import PARAMS

POSITIONS_FILE = Path(__file__).parent / "models" / "positions.json"


def load_positions() -> dict:
    """Load positions from disk."""
    if not POSITIONS_FILE.exists():
        return {"positions": [], "closed_today": [], "last_updated": None}
    with open(POSITIONS_FILE) as f:
        return json.load(f)


def save_positions(state: dict):
    """Save positions to disk."""
    state["last_updated"] = datetime.now().isoformat()
    POSITIONS_FILE.parent.mkdir(exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def update_positions(state: dict, new_signals: list, all_results: list) -> dict:
    """
    Update position state:
      1. Update open positions with current prices, check exit conditions
      2. Add new signals not already in open positions
      3. Save to disk
    """
    p = PARAMS

    # Build price lookup from scan results
    price_map = {}
    for r in all_results:
        if r.get("status") == "ok":
            price_map[r["ticker"]] = r

    closed_today = []
    still_open = []

    for pos in state.get("positions", []):
        ticker = pos["ticker"]
        if ticker not in price_map:
            still_open.append(pos)
            continue

        current = price_map[ticker]
        pos["bars_held"] = pos.get("bars_held", 0) + 1
        pos["current_price"] = current["close"]
        pos["unrealized_pnl_pct"] = round(
            (current["close"] / pos["entry_price"] - 1) * 100, 2
        )

        # Update trailing stop
        if current["close"] > pos.get("highest_close", pos["entry_price"]):
            pos["highest_close"] = current["close"]
            new_trail = current["close"] - 1.5 * current.get("atr", current["close"] * 0.02)
            pos["trailing_stop"] = max(pos.get("trailing_stop", 0), new_trail)

        # Check exit conditions
        exit_reason = None
        if current["close"] <= pos["stop_price"]:
            exit_reason = "stop"
        elif current["close"] >= pos["target_price"]:
            exit_reason = "profit_target"
        elif pos["bars_held"] >= 2 and current["close"] < pos.get("trailing_stop", 0):
            exit_reason = "trail_stop"
        elif pos["bars_held"] >= p.time_stop_bars:
            exit_reason = "time_stop"
        elif current.get("rv_pctile", 0) > p.rv_exit_pct:
            exit_reason = "vol_exhaustion"

        if exit_reason:
            pos["exit_reason"] = exit_reason
            pos["exit_date"] = current["date"]
            pos["exit_price"] = current["close"]
            pos["realized_pnl_pct"] = pos["unrealized_pnl_pct"]
            pos["status"] = "closed"
            closed_today.append(pos)
        else:
            still_open.append(pos)

    # Add new positions from signals
    open_tickers = {p["ticker"] for p in still_open}
    for sig in new_signals:
        if sig["ticker"] not in open_tickers:
            atr_pct = sig.get("atr_pct", 0.02)
            atr_val = sig["close"] * atr_pct
            still_open.append({
                "ticker": sig["ticker"],
                "entry_date": sig["date"],
                "entry_price": sig["close"],
                "size": sig["size"],
                "stop_price": sig["range_low"],
                "target_price": round(sig["close"] + p.profit_atr_mult * atr_val, 2),
                "trailing_stop": round(sig["close"] - 1.5 * atr_val, 2),
                "highest_close": sig["close"],
                "atr": round(atr_val, 4),
                "bars_held": 0,
                "current_price": sig["close"],
                "unrealized_pnl_pct": 0.0,
                "status": "open",
            })

    state["positions"] = still_open
    state["closed_today"] = closed_today
    save_positions(state)
    return state


def print_positions(state: dict):
    """Print current positions and today's closes."""
    positions = state.get("positions", [])
    closed = state.get("closed_today", [])

    if positions:
        print(f"\n  {'='*72}")
        print(f"  OPEN POSITIONS ({len(positions)})")
        print(f"  {'='*72}")
        header = (f"  {'Ticker':<7} {'Entry':>8} {'Current':>8} {'PnL%':>7} "
                  f"{'Stop':>8} {'Target':>8} {'Bars':>5} {'Size':>6}")
        print(header)
        print(f"  {'-'*68}")
        for p in sorted(positions, key=lambda x: x.get("unrealized_pnl_pct", 0),
                        reverse=True):
            pnl = p.get("unrealized_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            print(f"  {p['ticker']:<7} {p['entry_price']:>8.2f} "
                  f"{p['current_price']:>8.2f} {sign}{pnl:>6.2f}% "
                  f"{p['stop_price']:>8.2f} {p['target_price']:>8.2f} "
                  f"{p.get('bars_held', 0):>5d} {p.get('size', 0):>5.1%}")

    if closed:
        print(f"\n  {'='*72}")
        print(f"  CLOSED TODAY ({len(closed)})")
        print(f"  {'='*72}")
        for c in closed:
            pnl = c.get("realized_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            print(f"  {c['ticker']:<7} {c.get('exit_reason', '?'):<15} "
                  f"PnL: {sign}{pnl:.2f}%  (held {c.get('bars_held', 0)} bars)")
