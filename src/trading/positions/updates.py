"""Position update logic — daily exit evaluation and real-time price ratcheting."""

import logging

from rally_ml.config import PARAMS

from db.trading.positions import (
    delete_position_meta,
    record_closed_position,
    save_position_meta,
)

logger = logging.getLogger(__name__)


def ratchet_trailing_stop(
    pos: dict, high: float, atr_val: float, p=None,
) -> bool:
    """Ratchet trailing stop if high is a new high-water mark.

    Mutates pos in-place. Returns True if any field changed.
    Works for both daily close (update_existing_positions) and
    real-time ticks (update_position_for_price).
    """
    if p is None:
        p = PARAMS
    if high <= pos.get("highest_close", pos["entry_price"]):
        return False
    pos["highest_close"] = high
    new_trail = high - p.trailing_stop_atr_mult * atr_val
    if new_trail > pos.get("trailing_stop", 0):
        pos["trailing_stop"] = round(new_trail, 4)
    return True


def _apply_profit_lock(pos: dict, current_price: float, p) -> bool:
    """Raise hard stop floor once price reaches lock level. Returns True if changed."""
    if p.profit_lock_pct <= 0:
        return False
    entry = pos["entry_price"]
    lock_price = round(entry * (1 + p.profit_lock_pct), 4)
    if current_price >= lock_price and pos.get("stop_price", 0) < lock_price:
        pos["stop_price"] = lock_price
        return True
    return False


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
    atr_val = pos.get("atr", current_price * p.default_atr_pct)
    changed = ratchet_trailing_stop(pos, current_price, atr_val, p)
    if _apply_profit_lock(pos, current_price, p):
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

        atr_fallback = current["close"] * p.default_atr_pct
        atr_val = current.get("atr", atr_fallback)
        ratchet_trailing_stop(pos, current["close"], atr_val, p)
        _apply_profit_lock(pos, current["close"], p)

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
