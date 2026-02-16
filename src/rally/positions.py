"""
Position state tracking — persists open positions across scanner runs.

Storage: models/positions.json
"""

import json
from datetime import datetime
from pathlib import Path

from .config import PARAMS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POSITIONS_FILE = PROJECT_ROOT / "models" / "positions.json"


def load_positions() -> dict:
    """Load positions from disk."""
    if not POSITIONS_FILE.exists():
        return {"positions": [], "closed_today": [], "last_updated": None}
    with open(POSITIONS_FILE) as f:
        return json.load(f)


def save_positions(state: dict) -> None:
    """Save positions to disk."""
    state["last_updated"] = datetime.now().isoformat()
    POSITIONS_FILE.parent.mkdir(exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def update_positions(state: dict, new_signals: list[dict], all_results: list[dict]) -> dict:
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


def close_position_intraday(ticker: str, price: float, reason: str) -> dict | None:
    """Close a single position during market hours. Returns closed position or None."""
    state = load_positions()
    positions = state["positions"]

    target_idx = None
    for i, pos in enumerate(positions):
        if pos["ticker"] == ticker:
            target_idx = i
            break

    if target_idx is None:
        return None

    pos = positions.pop(target_idx)
    pos["exit_reason"] = reason
    pos["exit_price"] = price
    pos["exit_date"] = datetime.now().strftime("%Y-%m-%d")
    pos["realized_pnl_pct"] = round((price / pos["entry_price"] - 1) * 100, 2)
    pos["status"] = "closed"

    state["closed_today"].append(pos)
    save_positions(state)
    return pos


def update_fill_prices(fills: dict[str, float]) -> int:
    """Update positions that have pending order_ids with actual fill prices."""
    state = load_positions()
    updated = 0
    for pos in state["positions"]:
        oid = pos.get("order_id")
        if oid and oid in fills:
            pos["entry_price"] = fills[oid]
            pos["unrealized_pnl_pct"] = round(
                (pos["current_price"] / fills[oid] - 1) * 100, 2
            )
            del pos["order_id"]
            updated += 1
    if updated:
        save_positions(state)
    return updated


def print_positions(state: dict) -> None:
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
