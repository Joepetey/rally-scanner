"""
Position state tracking — business logic layer.

DB persistence is handled by db.positions.
Broker source of truth is Alpaca API.
"""

import logging

from config import PARAMS, TICKER_TO_GROUP
from db.positions import (
    clear_expired_queue,
    delete_position_meta,
    dequeue_signals,
    get_unevaluated_skipped,
    record_closed_position,
    save_position_meta,
    save_positions,
    update_skipped_outcome,
)
from trading.position_exits import (  # noqa: F401 — re-export
    async_close_position,
    close_position_by_trail_fill,
    close_position_intraday,
    get_group_exposure,
    get_total_exposure,
    get_trail_order_ids,
    update_fill_prices,
)
from trading.position_sync import (  # noqa: F401 — re-export
    _positions_lock,
    get_merged_positions,
    get_merged_positions_sync,
    sync_positions_from_alpaca,
)

logger = logging.getLogger(__name__)

__all__ = [
    "get_merged_positions",
    "get_merged_positions_sync",
    "update_existing_positions",
    "update_position_for_price",
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
# Position update logic
# ---------------------------------------------------------------------------

def _ratchet_trailing_stop(pos: dict, current: dict, p) -> None:
    """Update trailing stop if close is a new high. Mutates pos in-place."""
    if current["close"] > pos.get("highest_close", pos["entry_price"]):
        pos["highest_close"] = current["close"]
        atr_fallback = current["close"] * p.default_atr_pct
        atr_val = current.get("atr", atr_fallback)
        new_trail = current["close"] - p.trailing_stop_atr_mult * atr_val
        pos["trailing_stop"] = max(pos.get("trailing_stop", 0), new_trail)


def _evaluate_exit_conditions(pos: dict, current: dict, p) -> str | None:
    """Return exit reason string if any condition triggered, else None."""
    if current["close"] <= pos["stop_price"]:
        return "stop"
    if current["close"] >= pos["target_price"]:
        return "profit_target"
    if pos["bars_held"] >= 2 and current["close"] < pos.get("trailing_stop", 0):
        return "trail_stop"
    if pos["bars_held"] >= p.time_stop_bars:
        return "time_stop"
    if current.get("rv_pctile", 0) > p.rv_exit_pct:
        return "vol_exhaustion"
    return None


def update_position_for_price(pos: dict, current_price: float) -> bool:
    """Ratchet trailing stop and apply profit lock for a real-time price tick.

    Mutates pos in-place. Returns True if any field changed so the caller
    can decide whether to persist. ATR is taken from pos["atr"] (stored at
    entry time); falls back to entry_price * default_atr_pct if absent.
    """
    p = PARAMS
    changed = False

    if current_price > pos.get("highest_close", pos["entry_price"]):
        pos["highest_close"] = current_price
        atr_val = pos.get("atr", current_price * p.default_atr_pct)
        new_trail = current_price - p.trailing_stop_atr_mult * atr_val
        if new_trail > pos.get("trailing_stop", 0):
            pos["trailing_stop"] = round(new_trail, 4)
        changed = True

    if p.profit_lock_pct > 0:
        entry = pos["entry_price"]
        lock_price = round(entry * (1 + p.profit_lock_pct), 4)
        if current_price >= lock_price and pos.get("stop_price", 0) < lock_price:
            pos["stop_price"] = lock_price
            changed = True

    return changed


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

        _ratchet_trailing_stop(pos, current, p)

        # Profit lock: raise hard stop floor once close reaches lock level (idempotent)
        if p.profit_lock_pct > 0:
            lock_price = pos["entry_price"] * (1 + p.profit_lock_pct)
            if current["close"] >= lock_price:
                new_stop = max(pos.get("stop_price", 0), lock_price)
                if new_stop > pos.get("stop_price", 0):
                    pos["stop_price"] = round(new_stop, 4)

        exit_reason = _evaluate_exit_conditions(pos, current, p)

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
# Async helpers
# ---------------------------------------------------------------------------

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
