"""
Position state tracking — business logic layer.

DB persistence is handled by db.positions.
Broker source of truth is Alpaca API.
"""

import asyncio
import logging
import os
from datetime import datetime

from config import PARAMS, TICKER_TO_GROUP
from db.positions import (
    delete_position_meta,
    get_closed_today,
    load_all_position_meta,
    load_position_meta,
    load_positions,
    record_closed_position,
    save_position_meta,
    save_positions,
    tighten_trailing_stop,
)

logger = logging.getLogger(__name__)

# Async lock for concurrent position writes (scan + price alerts)
_positions_lock = asyncio.Lock()

__all__ = [
    "get_merged_positions",
    "get_merged_positions_sync",
    "update_existing_positions",
    "add_signal_positions",
    "update_positions",
    "close_position_intraday",
    "update_fill_prices",
    "get_trail_order_ids",
    "close_position_by_trail_fill",
    "get_total_exposure",
    "get_group_exposure",
    "async_close_position",
    "async_save_positions",
    "print_positions",
]


# ---------------------------------------------------------------------------
# Alpaca merge — source of truth for which positions are open
# ---------------------------------------------------------------------------

async def get_merged_positions() -> dict:
    """Merge Alpaca broker positions with DB metadata.

    Alpaca is the source of truth for WHICH positions are open.
    DB provides stop/target/trailing/bars_held metadata.
    Returns same shape as load_positions().
    """
    from trading.alpaca_executor import get_all_positions

    broker = await get_all_positions()
    meta_map = {p["ticker"]: p for p in load_all_position_meta()}

    merged = []
    for bp in broker:
        meta = meta_map.get(bp["ticker"], {})
        qty = bp["qty"]
        entry = bp["avg_entry_price"]
        current = bp["market_value"] / qty if qty else 0
        pnl_pct = (
            round(bp["unrealized_pl"] / (entry * qty) * 100, 2)
            if qty and entry else 0
        )
        merged.append({
            # From Alpaca (source of truth)
            "ticker": bp["ticker"],
            "qty": qty,
            "entry_price": entry,
            "current_price": round(current, 4),
            "unrealized_pnl_pct": pnl_pct,
            # From DB metadata
            "stop_price": meta.get("stop_price", 0),
            "target_price": meta.get("target_price", 0),
            "trailing_stop": meta.get("trailing_stop", 0),
            "highest_close": meta.get("highest_close", entry),
            "atr": meta.get("atr", 0),
            "bars_held": meta.get("bars_held", 0),
            "size": meta.get("size", 0),
            "entry_date": meta.get("entry_date", ""),
            "order_id": meta.get("order_id"),
            "trail_order_id": meta.get("trail_order_id"),
            "status": "open",
        })

    return {
        "positions": merged,
        "closed_today": get_closed_today(),
        "last_updated": datetime.now().isoformat(),
    }


def get_merged_positions_sync() -> dict:
    """Sync wrapper for get_merged_positions().

    Priority:
    1. RALLY_API_URL set → fetch from Railway API (local dev)
    2. Alpaca keys set   → query Alpaca + local DB
    3. Fallback          → DB-only via load_positions()
    """
    api_url = os.environ.get("RALLY_API_URL")
    if api_url:
        return _fetch_remote_positions(api_url)
    if not _has_alpaca_keys():
        return load_positions()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(get_merged_positions())
    # Already in async context — can't nest asyncio.run, fall back to DB
    return load_positions()


def _fetch_remote_positions(api_url: str) -> dict:
    """Fetch positions from the Railway API endpoint."""
    import json
    import urllib.request

    url = f"{api_url.rstrip('/')}/api/positions"
    req = urllib.request.Request(url)
    api_key = os.environ.get("RALLY_API_KEY")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        logger.warning("Failed to fetch positions from %s, falling back to local", url)
        return load_positions()


def _has_alpaca_keys() -> bool:
    """Check if Alpaca API keys are configured."""
    return bool(
        os.environ.get("ALPACA_API_KEY")
        and os.environ.get("ALPACA_SECRET_KEY")
    )


# ---------------------------------------------------------------------------
# Position update logic
# ---------------------------------------------------------------------------

def update_existing_positions(state: dict, all_results: list[dict]) -> dict:
    """Update open positions with current prices and check exit conditions.

    Only touches already-open positions — never adds new ones.
    Persists to DB and returns updated state.
    """
    p = PARAMS

    price_map = {r["ticker"]: r for r in all_results if r.get("status") == "ok"}

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
            delete_position_meta(ticker)
            record_closed_position(pos)
        else:
            still_open.append(pos)
            save_position_meta(pos)

    state["positions"] = still_open
    state["closed_today"] = closed_today
    return state


def add_signal_positions(state: dict, new_signals: list[dict]) -> dict:
    """Add new positions from signals, respecting portfolio + group exposure caps.

    Call this only after broker fill confirmation (Alpaca).
    Persists to DB and returns updated state.
    """
    p = PARAMS
    still_open = state.get("positions", [])

    open_tickers = {pos["ticker"] for pos in still_open}
    current_exposure = sum(pos.get("size", 0) for pos in still_open)
    max_exposure = p.max_portfolio_exposure

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
        g = TICKER_TO_GROUP.get(sig["ticker"])
        if g:
            if group_counts.get(g, 0) >= p.max_group_positions:
                continue
            if group_exposures.get(g, 0) + sig_size > p.max_group_exposure:
                continue
        atr_pct = sig.get("atr_pct", p.default_atr_pct)
        atr_val = sig["close"] * atr_pct
        new_pos = {
            "ticker": sig["ticker"],
            "entry_date": sig["date"],
            "entry_price": sig["close"],
            "size": sig["size"],
            "stop_price": sig["range_low"],
            "target_price": round(sig["close"] + p.profit_atr_mult * atr_val, 2),
            "trailing_stop": round(
                sig["close"] - p.trailing_stop_atr_mult * atr_val, 2
            ),
            "highest_close": sig["close"],
            "atr": round(atr_val, 4),
            "bars_held": 0,
            "current_price": sig["close"],
            "unrealized_pnl_pct": 0.0,
            "status": "open",
        }
        still_open.append(new_pos)
        save_position_meta(new_pos)
        current_exposure += sig["size"]
        open_tickers.add(sig["ticker"])
        if g:
            group_counts[g] = group_counts.get(g, 0) + 1
            group_exposures[g] = group_exposures.get(g, 0) + sig["size"]

    state["positions"] = still_open
    return state


def update_positions(
    state: dict, new_signals: list[dict], all_results: list[dict],
) -> dict:
    """Update existing positions and add new signals (backward-compat wrapper)."""
    state = update_existing_positions(state, all_results)
    state = add_signal_positions(state, new_signals)
    return state


# ---------------------------------------------------------------------------
# Intraday operations
# ---------------------------------------------------------------------------

def close_position_intraday(ticker: str, price: float, reason: str) -> dict | None:
    """Close a single position during market hours. Returns closed position or None."""
    pos = load_position_meta(ticker)
    if pos is None:
        return None

    pos["exit_reason"] = reason
    pos["exit_price"] = price
    pos["exit_date"] = datetime.now().strftime("%Y-%m-%d")
    pos["realized_pnl_pct"] = round((price / pos["entry_price"] - 1) * 100, 2)
    pos["status"] = "closed"

    delete_position_meta(ticker)
    record_closed_position(pos)
    return pos


async def update_fill_prices(fills: dict[str, float]) -> int:
    """Update positions that have pending order_ids with actual fill prices."""
    async with _positions_lock:
        positions = load_all_position_meta()
        updated = 0
        for pos in positions:
            oid = pos.get("order_id")
            if oid and oid in fills:
                pos["entry_price"] = fills[oid]
                pos["order_id"] = None
                save_position_meta(pos)
                updated += 1
        return updated


def get_trail_order_ids() -> dict[str, str]:
    """Return {ticker: trail_order_id} for positions with trailing stops."""
    positions = load_all_position_meta()
    return {
        pos["ticker"]: pos["trail_order_id"]
        for pos in positions
        if pos.get("trail_order_id")
    }


def close_position_by_trail_fill(ticker: str, fill_price: float) -> dict | None:
    """Close a position whose trailing stop filled at the broker."""
    return close_position_intraday(ticker, fill_price, "trail_stop_filled")


def get_total_exposure() -> float:
    """Return sum of size (fraction of equity) across all open positions."""
    return sum(p.get("size", 0) for p in load_all_position_meta())


def get_group_exposure(group: str) -> tuple[int, float]:
    """Return (count, total_exposure) for positions in the given asset group."""
    matched = [p for p in load_all_position_meta() if TICKER_TO_GROUP.get(p["ticker"]) == group]
    return len(matched), sum(p.get("size", 0) for p in matched)


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def async_close_position(
    ticker: str, price: float, reason: str,
) -> dict | None:
    """Thread-safe close via asyncio.Lock."""
    async with _positions_lock:
        return close_position_intraday(ticker, price, reason)


async def async_save_positions(state: dict) -> None:
    """Thread-safe save via asyncio.Lock."""
    async with _positions_lock:
        save_positions(state)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_positions(state: dict) -> None:
    """Print current positions and today's closes."""
    positions = state.get("positions", [])
    closed = state.get("closed_today", [])

    if positions:
        print(f"\n  {'='*72}")
        print(f"  OPEN POSITIONS ({len(positions)})")
        print(f"  {'='*72}")
        header = (
            f"  {'Ticker':<7} {'Entry':>8} {'Current':>8} {'PnL%':>7} "
            f"{'Stop':>8} {'Target':>8} {'Bars':>5} {'Size':>6}"
        )
        print(header)
        print(f"  {'-'*68}")
        for p in sorted(
            positions,
            key=lambda x: x.get("unrealized_pnl_pct", 0),
            reverse=True,
        ):
            pnl = p.get("unrealized_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            print(
                f"  {p['ticker']:<7} {p['entry_price']:>8.2f} "
                f"{p.get('current_price', 0):>8.2f} {sign}{pnl:>6.2f}% "
                f"{p.get('stop_price', 0):>8.2f} "
                f"{p.get('target_price', 0):>8.2f} "
                f"{p.get('bars_held', 0):>5d} {p.get('size', 0):>5.1%}"
            )

    if closed:
        print(f"\n  {'='*72}")
        print(f"  CLOSED TODAY ({len(closed)})")
        print(f"  {'='*72}")
        for c in closed:
            pnl = c.get("realized_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            print(
                f"  {c['ticker']:<7} {c.get('exit_reason', '?'):<15} "
                f"PnL: {sign}{pnl:.2f}%  (held {c.get('bars_held', 0)} bars)"
            )
