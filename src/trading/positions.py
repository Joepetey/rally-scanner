"""
Position state tracking — business logic layer.

DB persistence is handled by db.positions.
Broker source of truth is Alpaca API.
"""

import asyncio
import json
import logging
from datetime import datetime

from config import PARAMS, TICKER_TO_GROUP
from integrations.alpaca.broker import get_all_positions
from db.positions import (
    clear_expired_queue,
    delete_position_meta,
    dequeue_signals,
    get_closed_today,
    get_unevaluated_skipped,
    load_all_position_meta,
    load_position_meta,
    load_positions,
    record_closed_position,
    remove_from_queue,
    save_position_meta,
    save_positions,
    tighten_trailing_stop,
    update_skipped_outcome,
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
    "process_signal_queue",
    "update_skipped_outcomes",
    "sync_positions_from_alpaca",
]


# ---------------------------------------------------------------------------
# Alpaca merge — source of truth for which positions are open
# ---------------------------------------------------------------------------

async def get_merged_positions() -> dict:
    """Sync with Alpaca broker state then return DB positions."""
    await sync_positions_from_alpaca()
    return load_positions()


async def sync_positions_from_alpaca() -> dict:
    """Reconcile DB with Alpaca broker state.

    Three cases:
    1. In Alpaca + in DB  → update qty/entry_price from broker, keep all metadata
    2. In Alpaca, not DB  → insert with broker values and PARAMS-derived stops
    3. In DB, not Alpaca  → broker closed it; recover fill price, delete + record closed
    """
    async with _positions_lock:
        # Local import to avoid circular dependency (executor imports from trading.positions)
        from integrations.alpaca.executor import get_recent_sell_fills

        broker_list = await get_all_positions()
        broker_map = {p["ticker"]: p for p in broker_list}
        meta_list = load_all_position_meta()
        meta_map = {p["ticker"]: p for p in meta_list}

        broker_tickers = set(broker_map)
        db_tickers = set(meta_map)

        # Case 3: in DB, gone from Alpaca — broker closed the position
        closed_tickers = db_tickers - broker_tickers
        fills = {}
        if closed_tickers:
            fills = await get_recent_sell_fills(list(closed_tickers))
        for ticker in closed_tickers:
            pos = meta_map[ticker]
            fill_price = fills.get(ticker, 0.0)
            pnl = (
                round((fill_price / pos["entry_price"] - 1) * 100, 2)
                if fill_price and pos.get("entry_price") else 0.0
            )
            pos["exit_price"] = fill_price
            pos["exit_date"] = datetime.now().strftime("%Y-%m-%d")
            pos["exit_reason"] = "broker_closed"
            pos["realized_pnl_pct"] = pnl
            pos["status"] = "closed"
            delete_position_meta(ticker)
            record_closed_position(pos)
            logger.info("sync: closed %s (broker_closed), fill=%.4f", ticker, fill_price)

        # Case 2: in Alpaca, not in DB — untracked position (manually opened / missed fill)
        # Derive stops/target from entry price using PARAMS defaults
        inserted_tickers = broker_tickers - db_tickers
        for ticker in inserted_tickers:
            bp = broker_map[ticker]
            p = PARAMS
            entry = bp["avg_entry_price"]
            atr_val = round(entry * p.default_atr_pct, 4)
            new_pos = {
                "ticker": ticker,
                "qty": bp["qty"],
                "entry_price": entry,
                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                "stop_price": round(entry * (1 - p.fallback_stop_pct), 4),
                "target_price": round(entry + p.profit_atr_mult * atr_val, 4),
                "trailing_stop": round(entry - p.trailing_stop_atr_mult * atr_val, 4),
                "highest_close": entry,
                "atr": atr_val,
                "bars_held": 0,
                "size": 0.0,
                "p_rally": 0.0,
                "status": "open",
            }
            save_position_meta(new_pos)
            logger.warning("sync: inserted untracked position %s with default stops", ticker)

        # Case 1: in both — update qty/entry from broker, keep all metadata unchanged
        matched_tickers = broker_tickers & db_tickers
        for ticker in matched_tickers:
            bp = broker_map[ticker]
            pos = meta_map[ticker]
            pos["qty"] = bp["qty"]
            pos["entry_price"] = bp["avg_entry_price"]
            save_position_meta(pos)

        return {
            "synced": len(matched_tickers),
            "closed": len(closed_tickers),
            "inserted": len(inserted_tickers),
        }


def get_merged_positions_sync() -> dict:
    """DB-only read; sync is handled proactively by the scheduler."""
    return load_positions()


# ---------------------------------------------------------------------------
# Position update logic
# ---------------------------------------------------------------------------

def update_existing_positions(
    state: dict, all_results: list[dict], commit_exits: bool = True,
) -> dict:
    """Update open positions with current prices and check exit conditions.

    Only touches already-open positions — never adds new ones.

    Args:
        state: current position state dict.
        all_results: scan results with latest prices.
        commit_exits: if True (default), immediately write exits to DB
            (delete from system_positions, insert into closed_positions).
            Pass False when the caller needs to confirm broker execution
            before committing — the caller is then responsible for calling
            delete_position_meta() and record_closed_position() per exit.
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

        # Profit lock: raise hard stop floor once close reaches lock level (idempotent)
        if p.profit_lock_pct > 0:
            lock_price = pos["entry_price"] * (1 + p.profit_lock_pct)
            if current["close"] >= lock_price:
                new_stop = max(pos.get("stop_price", 0), lock_price)
                if new_stop > pos.get("stop_price", 0):
                    pos["stop_price"] = round(new_stop, 4)

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
            if commit_exits:
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
            "p_rally": sig.get("p_rally", 0),
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
# Signal queue processing
# ---------------------------------------------------------------------------

def process_signal_queue() -> list[dict]:
    """Return valid queued signals ready to re-attempt, clearing expired entries.

    Signals are ordered by P(rally) descending. The caller is responsible for
    attempting execution and calling remove_from_queue on success.
    """
    expired = clear_expired_queue(PARAMS.signal_queue_max_age_days)
    if expired:
        logger.info("Cleared %d expired signals from queue", expired)

    return dequeue_signals(PARAMS.signal_queue_max_age_days)


def update_skipped_outcomes(results: list[dict]) -> int:
    """Fill in outcome data for previously skipped signals using today's scan prices.

    Looks up each unevaluated skipped signal, finds a matching current price in
    results, computes the return from signal_date close to today's close, and
    persists it. Returns number of outcomes recorded.
    """
    unevaluated = get_unevaluated_skipped()
    if not unevaluated:
        return 0

    price_map = {r["ticker"]: r["close"] for r in results if r.get("status") == "ok"}
    recorded = 0
    for entry in unevaluated:
        ticker = entry["ticker"]
        if ticker not in price_map:
            continue
        current_price = price_map[ticker]
        signal_close = entry.get("close", 0)
        if signal_close <= 0:
            continue
        outcome_pct = round((current_price / signal_close - 1) * 100, 2)
        update_skipped_outcome(ticker, str(entry["signal_date"]), current_price, outcome_pct)
        recorded += 1

    if recorded:
        logger.info("Updated outcomes for %d skipped signals", recorded)
    return recorded


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
