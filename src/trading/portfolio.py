"""Portfolio tracking — daily snapshots, trade journal, equity history."""

import logging
from datetime import datetime

from db.portfolio import (
    get_high_water_mark,
    load_equity_history,
    load_trade_journal,
    record_closed_trades,
    save_snapshot,
    set_high_water_mark,
)

# Re-export for callers that import from trading.portfolio
__all__ = [
    "update_daily_snapshot",
    "record_closed_trades",
    "load_equity_history",
    "load_trade_journal",
    "compute_drawdown",
    "is_circuit_breaker_active",
]

logger = logging.getLogger(__name__)


def update_daily_snapshot(positions_state: dict, scan_results: list[dict]) -> dict:
    """Persist today's portfolio snapshot. Called after each daily scan."""
    open_positions = positions_state.get("positions", [])

    total_exposure = sum(p.get("size", 0) for p in open_positions)
    total_unrealized = sum(
        p.get("unrealized_pnl_pct", 0) * p.get("size", 0)
        for p in open_positions
    )

    snapshot = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "n_positions": len(open_positions),
        "total_exposure": round(total_exposure, 4),
        "total_unrealized_pnl_pct": round(total_unrealized, 6),
        "n_signals_today": sum(1 for r in scan_results if r.get("signal")),
        "n_scanned": sum(1 for r in scan_results if r.get("status") == "ok"),
    }

    save_snapshot(snapshot)

    logger.info(
        "Daily snapshot: %d positions, %.1f%% exposure",
        snapshot["n_positions"], snapshot["total_exposure"] * 100,
    )
    return snapshot


def compute_drawdown(equity: float) -> float:
    """Compute current drawdown from equity high-water mark.

    Returns drawdown as a positive fraction (e.g. 0.10 = 10% drawdown).
    Returns 0.0 if equity is at/above the high-water mark.
    """
    if equity <= 0:
        return 0.0

    hwm = get_high_water_mark()
    if hwm <= 0:
        hwm = equity

    if equity >= hwm:
        set_high_water_mark(equity)
        return 0.0

    return (hwm - equity) / hwm


def is_circuit_breaker_active(equity: float) -> bool:
    """Check if the drawdown circuit breaker should block new entries."""
    from config import PARAMS
    if not PARAMS.circuit_breaker_enabled:
        return False
    dd = compute_drawdown(equity)
    if dd >= PARAMS.max_drawdown_pct:
        logger.warning(
            "Circuit breaker ACTIVE: drawdown %.1f%% >= threshold %.1f%%",
            dd * 100, PARAMS.max_drawdown_pct * 100,
        )
        return True
    return False
