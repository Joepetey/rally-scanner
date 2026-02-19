"""
Position state tracking — persists open positions across scanner runs.

Storage: models/positions.json
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from ..config import PARAMS, TICKER_TO_GROUP
from ..utils import atomic_json_write

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
POSITIONS_FILE = PROJECT_ROOT / "models" / "positions.json"

# Async lock for concurrent position writes (scan + price alerts + reconciliation)
_positions_lock = asyncio.Lock()


def load_positions() -> dict:
    """Load positions from disk."""
    if not POSITIONS_FILE.exists():
        return {"positions": [], "closed_today": [], "last_updated": None}
    with open(POSITIONS_FILE) as f:
        return json.load(f)


def save_positions(state: dict) -> None:
    """Save positions to disk atomically (write to temp, then rename)."""
    state["last_updated"] = datetime.now().isoformat()
    atomic_json_write(POSITIONS_FILE, state, default=str)


def update_existing_positions(state: dict, all_results: list[dict]) -> dict:
    """Update open positions with current prices and check exit conditions.

    Only touches already-open positions — never adds new ones.
    Saves to disk and returns updated state.
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
            atr_fallback = current["close"] * p.default_atr_pct
            atr_val = current.get("atr", atr_fallback)
            new_trail = current["close"] - p.trailing_stop_atr_mult * atr_val
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

    state["positions"] = still_open
    state["closed_today"] = closed_today
    save_positions(state)
    return state


def add_signal_positions(state: dict, new_signals: list[dict]) -> dict:
    """Add new positions from signals, respecting portfolio + group exposure caps.

    Call this only after broker fill confirmation (Alpaca).
    Saves to disk and returns updated state.
    """
    p = PARAMS
    still_open = state.get("positions", [])

    open_tickers = {pos["ticker"] for pos in still_open}
    current_exposure = sum(pos.get("size", 0) for pos in still_open)
    max_exposure = p.max_portfolio_exposure

    # Pre-compute group counts/exposure from current open positions
    group_counts: dict[str, int] = {}
    group_exposures: dict[str, float] = {}
    for pos in still_open:
        g = TICKER_TO_GROUP.get(pos["ticker"])
        if g:
            group_counts[g] = group_counts.get(g, 0) + 1
            group_exposures[g] = group_exposures.get(g, 0) + pos.get("size", 0)

    for sig in new_signals:
        if sig["ticker"] in open_tickers:
            continue
        sig_size = sig.get("size", 0)
        if current_exposure + sig_size > max_exposure:
            continue
        # Group concentration check
        g = TICKER_TO_GROUP.get(sig["ticker"])
        if g:
            if group_counts.get(g, 0) >= p.max_group_positions:
                continue
            if group_exposures.get(g, 0) + sig_size > p.max_group_exposure:
                continue
        atr_pct = sig.get("atr_pct", p.default_atr_pct)
        atr_val = sig["close"] * atr_pct
        still_open.append({
            "ticker": sig["ticker"],
            "entry_date": sig["date"],
            "entry_price": sig["close"],
            "size": sig["size"],
            "stop_price": sig["range_low"],
            "target_price": round(sig["close"] + p.profit_atr_mult * atr_val, 2),
            "trailing_stop": round(sig["close"] - p.trailing_stop_atr_mult * atr_val, 2),
            "highest_close": sig["close"],
            "atr": round(atr_val, 4),
            "bars_held": 0,
            "current_price": sig["close"],
            "unrealized_pnl_pct": 0.0,
            "status": "open",
        })
        current_exposure += sig["size"]
        open_tickers.add(sig["ticker"])
        # Update group tracking
        g = TICKER_TO_GROUP.get(sig["ticker"])
        if g:
            group_counts[g] = group_counts.get(g, 0) + 1
            group_exposures[g] = group_exposures.get(g, 0) + sig["size"]

    state["positions"] = still_open
    save_positions(state)
    return state


def update_positions(state: dict, new_signals: list[dict], all_results: list[dict]) -> dict:
    """Update existing positions and add new signals (backward-compat wrapper).

    Combines update_existing_positions() + add_signal_positions().
    Used by backtests and tests that expect the old all-in-one behavior.
    """
    state = update_existing_positions(state, all_results)
    state = add_signal_positions(state, new_signals)
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


def get_trail_order_ids() -> dict[str, str]:
    """Return {ticker: trail_order_id} for all open positions with trailing stops."""
    state = load_positions()
    return {
        pos["ticker"]: pos["trail_order_id"]
        for pos in state.get("positions", [])
        if pos.get("trail_order_id")
    }


def close_position_by_trail_fill(ticker: str, fill_price: float) -> dict | None:
    """Close a position whose trailing stop filled at the broker."""
    return close_position_intraday(ticker, fill_price, "trail_stop_filled")


def get_total_exposure() -> float:
    """Return sum of size (fraction of equity) across all open positions."""
    state = load_positions()
    return sum(p.get("size", 0) for p in state.get("positions", []))


def get_group_exposure(group: str) -> tuple[int, float]:
    """Return (count, total_exposure) for positions in the given asset group."""
    state = load_positions()
    count = 0
    exposure = 0.0
    for p in state.get("positions", []):
        if TICKER_TO_GROUP.get(p["ticker"]) == group:
            count += 1
            exposure += p.get("size", 0)
    return count, exposure


def reconcile_with_broker(
    broker_positions: list[dict],
    trail_fills: dict[str, float],
) -> list[str]:
    """Compare local positions with broker state. Returns list of warning messages.

    - Auto-closes local positions whose trailing stops filled at broker.
    - Logs warnings for ghost positions (local but not at broker) and
      orphaned positions (at broker but not local).
    """
    warnings: list[str] = []
    state = load_positions()
    local_tickers = {p["ticker"] for p in state.get("positions", [])}
    broker_tickers = {p["ticker"] for p in broker_positions}

    # Trailing stop fills: close local positions
    for ticker, fill_price in trail_fills.items():
        if ticker in local_tickers:
            close_position_by_trail_fill(ticker, fill_price)
            warnings.append(
                f"Trailing stop filled for {ticker} at ${fill_price:.2f} — position closed"
            )

    # Ghost positions: we think we hold it but broker doesn't — auto-remove
    ghosts = local_tickers - broker_tickers - set(trail_fills)
    if ghosts:
        state = load_positions()
        state["positions"] = [
            p for p in state["positions"] if p["ticker"] not in ghosts
        ]
        save_positions(state)
        for ticker in sorted(ghosts):
            warnings.append(
                f"Ghost position removed: {ticker} (local only, not at broker)"
            )

    # Orphaned positions: broker holds it but we don't track it
    for ticker in broker_tickers - local_tickers:
        warnings.append(
            f"Orphaned position: {ticker} at broker but not in local state"
        )

    if warnings:
        for w in warnings:
            logger.warning("Reconciliation: %s", w)

    return warnings


def tighten_trailing_stop(ticker: str, new_stop: float) -> dict | None:
    """Tighten a position's trailing stop (only if new_stop > current).

    Returns the updated position dict, or None if not found or not tightened.
    """
    state = load_positions()
    for pos in state.get("positions", []):
        if pos["ticker"] == ticker:
            current = pos.get("trailing_stop", 0)
            if new_stop > current:
                pos["trailing_stop"] = round(new_stop, 2)
                save_positions(state)
                logger.info(
                    "Tightened trailing stop for %s: %.2f → %.2f",
                    ticker, current, new_stop,
                )
                return pos
            return None
    return None


async def async_close_position(ticker: str, price: float, reason: str) -> dict | None:
    """Thread-safe close via asyncio.Lock."""
    async with _positions_lock:
        return close_position_intraday(ticker, price, reason)


async def async_save_positions(state: dict) -> None:
    """Thread-safe save via asyncio.Lock."""
    async with _positions_lock:
        save_positions(state)


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
