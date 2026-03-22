"""Portfolio tracking persistence — snapshots, trade journal, high-water mark."""

import logging
from datetime import datetime

from db.pool import get_conn, row_to_dict

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


def save_snapshot(snapshot: dict) -> None:
    """Upsert a daily portfolio snapshot into portfolio_snapshots."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO portfolio_snapshots
                   (snapshot_date, n_positions, total_exposure,
                    total_unrealized_pnl_pct, n_signals_today, n_scanned)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (snapshot_date) DO UPDATE SET
                   n_positions=EXCLUDED.n_positions,
                   total_exposure=EXCLUDED.total_exposure,
                   total_unrealized_pnl_pct=EXCLUDED.total_unrealized_pnl_pct,
                   n_signals_today=EXCLUDED.n_signals_today,
                   n_scanned=EXCLUDED.n_scanned""",
            (
                snapshot["date"], snapshot["n_positions"],
                snapshot["total_exposure"], snapshot["total_unrealized_pnl_pct"],
                snapshot["n_signals_today"], snapshot["n_scanned"],
            ),
        )


def record_closed_trades(closed_trades: list[dict]) -> None:
    """Insert closed trades into the trade_journal table."""
    if not closed_trades:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        for trade in closed_trades:
            cur.execute(
                """INSERT INTO trade_journal
                       (exit_date, ticker, entry_date, entry_price, exit_price,
                        realized_pnl_pct, size, bars_held, exit_reason)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    trade.get("exit_date"), trade.get("ticker"),
                    trade.get("entry_date"), trade.get("entry_price"),
                    trade.get("exit_price"), trade.get("realized_pnl_pct"),
                    trade.get("size"), trade.get("bars_held"),
                    trade.get("exit_reason"),
                ),
            )


def load_equity_history(days: int | None = None) -> list[dict]:
    """Load portfolio snapshots in chronological order. Optionally limit to last N snapshots."""
    cols = (
        "snapshot_date AS date, n_positions, total_exposure, "
        "total_unrealized_pnl_pct, n_signals_today, n_scanned"
    )
    with get_conn() as conn:
        cur = conn.cursor()
        if days:
            # Last N rows by date, returned in chronological order
            cur.execute(
                f"SELECT {cols} FROM portfolio_snapshots "
                "ORDER BY snapshot_date DESC LIMIT %s",
                [days],
            )
            return list(reversed([row_to_dict(r) for r in cur.fetchall()]))
        cur.execute(f"SELECT {cols} FROM portfolio_snapshots ORDER BY snapshot_date ASC")
        return [row_to_dict(r) for r in cur.fetchall()]


def load_trade_journal(limit: int | None = None) -> list[dict]:
    """Load trade journal entries, optionally limited to most recent N."""
    query = (
        "SELECT exit_date, ticker, entry_date, entry_price, exit_price, "
        "realized_pnl_pct, size, bars_held, exit_reason "
        "FROM trade_journal ORDER BY exit_date DESC, id DESC"
    )
    params: list = []
    if limit:
        query += " LIMIT %s"
        params = [limit]

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = [row_to_dict(r) for r in cur.fetchall()]
    # Return in chronological order (oldest first), matching CSV behaviour
    return list(reversed(rows))


def get_high_water_mark() -> float:
    """Get the portfolio equity high-water mark."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT value FROM portfolio_meta WHERE key = 'high_water_mark'"
        )
        row = cur.fetchone()
    return float(row["value"]) if row else 0.0


def set_high_water_mark(value: float) -> None:
    """Update the portfolio equity high-water mark."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE portfolio_meta SET value = %s, updated_at = NOW() "
            "WHERE key = 'high_water_mark'",
            (str(value),),
        )
