"""Portfolio tracking — daily snapshots, trade journal, equity history."""

import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "models"
EQUITY_LOG = DATA_DIR / "equity_history.csv"
TRADE_JOURNAL = DATA_DIR / "trade_journal.csv"

EQUITY_FIELDS = [
    "date", "n_positions", "total_exposure",
    "total_unrealized_pnl_pct", "n_signals_today", "n_scanned",
]

JOURNAL_FIELDS = [
    "exit_date", "ticker", "entry_date", "entry_price", "exit_price",
    "realized_pnl_pct", "size", "bars_held", "exit_reason",
]


def update_daily_snapshot(positions_state: dict, scan_results: list[dict]) -> dict:
    """Append today's portfolio snapshot to equity_history.csv.

    Called by the orchestrator after each daily scan.
    """
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

    file_exists = EQUITY_LOG.exists()
    with open(EQUITY_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EQUITY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: snapshot[k] for k in EQUITY_FIELDS})

    logger.info(f"Daily snapshot: {snapshot['n_positions']} positions, "
                f"{snapshot['total_exposure']:.1%} exposure")
    return snapshot


def record_closed_trades(closed_trades: list[dict]) -> None:
    """Append closed trades to the trade journal CSV."""
    if not closed_trades:
        return

    file_exists = TRADE_JOURNAL.exists()
    with open(TRADE_JOURNAL, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for trade in closed_trades:
            trade.setdefault("exit_date", datetime.now().strftime("%Y-%m-%d"))
            writer.writerow(trade)

    logger.info(f"Recorded {len(closed_trades)} closed trades to journal")


def load_equity_history(days: int | None = None) -> list[dict]:
    """Load equity history from CSV. Optionally limit to last N days."""
    if not EQUITY_LOG.exists():
        return []
    with open(EQUITY_LOG) as f:
        rows = list(csv.DictReader(f))
    if days and len(rows) > days:
        rows = rows[-days:]
    return rows


def compute_drawdown(equity: float) -> float:
    """Compute current drawdown from equity high-water mark.

    Uses the equity history to find the peak, then compares to current equity.
    Returns drawdown as a positive fraction (e.g. 0.10 = 10% drawdown).
    Returns 0.0 if no history or equity is at/above the high-water mark.
    """
    rows = load_equity_history()
    if not rows or equity <= 0:
        return 0.0

    # The equity history stores total_unrealized_pnl_pct (weighted by size).
    # For the circuit breaker we need to track actual equity values.
    # Use a simple approach: if the caller provides current equity (from Alpaca),
    # compare to the max equity we've seen.
    # For now, we approximate: use equity directly vs stored high-water mark.
    hwm_file = DATA_DIR / "high_water_mark.txt"
    if hwm_file.exists():
        try:
            hwm = float(hwm_file.read_text().strip())
        except (ValueError, OSError):
            hwm = equity
    else:
        hwm = equity

    # Update high-water mark if equity is higher
    if equity >= hwm:
        hwm = equity
        DATA_DIR.mkdir(exist_ok=True)
        hwm_file.write_text(str(hwm))
        return 0.0

    drawdown = (hwm - equity) / hwm
    return drawdown


def is_circuit_breaker_active(equity: float) -> bool:
    """Check if the drawdown circuit breaker should block new entries."""
    from ..config import PARAMS
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


def load_trade_journal(limit: int | None = None) -> list[dict]:
    """Load trade journal from CSV. Optionally limit to most recent N."""
    if not TRADE_JOURNAL.exists():
        return []
    with open(TRADE_JOURNAL) as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[-limit:]
    return rows
