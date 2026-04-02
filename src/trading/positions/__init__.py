"""Position state tracking — re-export facade.

Actual logic lives in focused submodules:
- broker_sync: Alpaca reconciliation
- updates: daily exit evaluation, trailing stop ratcheting
- queue: signal queue processing, constraint filtering
- intraday: intraday closes, fill updates, exposure queries
- display: terminal position display
"""

from rally_ml.config import PARAMS

from db.trading.positions import save_position_meta
from trading.positions.broker_sync import (
    build_untracked_position,
    compute_sync_updates,
    get_merged_positions,
    get_merged_positions_sync,
    sync_positions_from_alpaca,
)
from trading.positions.display import print_positions
from trading.positions.intraday import (
    async_close_position,
    async_save_positions,
    build_closed_record,
    close_position_by_trail_fill,
    close_position_intraday,
    get_group_exposure,
    get_total_exposure,
    get_trail_order_ids,
    update_fill_prices,
)
from trading.positions.queue import (
    filter_signals_by_constraints,
    process_signal_queue,
    update_skipped_outcomes,
)
from trading.positions.updates import (
    update_existing_positions,
    update_position_for_price,
)

__all__ = [
    "get_merged_positions",
    "get_merged_positions_sync",
    "update_existing_positions",
    "update_position_for_price",
    "add_signal_positions",
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
    "build_untracked_position",
    "compute_sync_updates",
    "filter_signals_by_constraints",
    "build_closed_record",
]


def add_signal_positions(state: dict, new_signals: list[dict]) -> dict:
    """Add new positions from signals, respecting portfolio + group exposure caps.

    Call this only after broker fill confirmation (Alpaca).
    Persists to DB and returns updated state.
    """
    p = PARAMS
    still_open = state.get("positions", [])

    accepted = filter_signals_by_constraints(
        still_open, new_signals,
        p.max_portfolio_exposure, p.max_group_positions, p.max_group_exposure,
    )

    for sig in accepted:
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

    state["positions"] = still_open
    return state
