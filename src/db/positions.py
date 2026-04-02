"""System position persistence — open/closed position CRUD against PostgreSQL."""

import logging
from datetime import datetime

from db.pool import get_conn, row_to_dict
from db.scan_results import (  # noqa: F401 — re-export
    load_current_signals,
    load_watchlist,
    save_latest_scan,
    save_watchlist,
)
from db.signal_queue import (  # noqa: F401 — re-export
    clear_expired_queue,
    dequeue_signals,
    enqueue_signal,
    get_unevaluated_skipped,
    log_skipped_signal,
    remove_from_queue,
    update_skipped_outcome,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# system_positions CRUD
# ---------------------------------------------------------------------------

_UPSERT_POSITION_SQL = """INSERT INTO system_positions
       (ticker, entry_price, entry_date, stop_price, target_price,
        trailing_stop, highest_close, atr, bars_held, size, qty,
        order_id, trail_order_id, target_order_id, p_rally,
        current_price, unrealized_pnl_pct, updated_at)
   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
   ON CONFLICT (ticker) DO UPDATE SET
       entry_price=EXCLUDED.entry_price, entry_date=EXCLUDED.entry_date,
       stop_price=EXCLUDED.stop_price, target_price=EXCLUDED.target_price,
       trailing_stop=EXCLUDED.trailing_stop,
       highest_close=EXCLUDED.highest_close,
       atr=EXCLUDED.atr, bars_held=EXCLUDED.bars_held, size=EXCLUDED.size,
       qty=EXCLUDED.qty, order_id=EXCLUDED.order_id,
       trail_order_id=EXCLUDED.trail_order_id,
       target_order_id=EXCLUDED.target_order_id,
       p_rally=EXCLUDED.p_rally,
       current_price=EXCLUDED.current_price,
       unrealized_pnl_pct=EXCLUDED.unrealized_pnl_pct,
       updated_at=NOW()"""


def _position_params(pos: dict) -> tuple:
    """Extract ordered parameters for the position UPSERT."""
    return (
        pos["ticker"], pos.get("entry_price", 0), pos.get("entry_date") or None,
        pos.get("stop_price", 0), pos.get("target_price", 0),
        pos.get("trailing_stop", 0),
        pos.get("highest_close", pos.get("entry_price", 0)),
        pos.get("atr", 0), pos.get("bars_held", 0), pos.get("size", 0),
        pos.get("qty", 0), pos.get("order_id"), pos.get("trail_order_id"),
        pos.get("target_order_id"), pos.get("p_rally", 0),
        pos.get("current_price", pos.get("entry_price", 0)),
        pos.get("unrealized_pnl_pct", 0),
    )


def save_position_meta(pos: dict) -> None:
    """UPSERT a single position row into system_positions."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_UPSERT_POSITION_SQL, _position_params(pos))


def load_position_meta(ticker: str) -> dict | None:
    """Load a single position's metadata."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM system_positions WHERE ticker = %s", (ticker,)
        )
        row = cur.fetchone()
    if row is None:
        return None
    d = row_to_dict(row)
    d["status"] = "open"
    return d


def load_all_position_meta() -> list[dict]:
    """Load all open position metadata."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM system_positions")
        rows = cur.fetchall()
    result = []
    for r in rows:
        d = row_to_dict(r)
        d["status"] = "open"
        result.append(d)
    return result


def delete_position_meta(ticker: str) -> None:
    """Remove a position from the DB."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM system_positions WHERE ticker = %s", (ticker,))


# ---------------------------------------------------------------------------
# closed_positions CRUD
# ---------------------------------------------------------------------------

def record_closed_position(pos: dict) -> None:
    """Record a closed position in the closed_positions table."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO closed_positions
                   (ticker, entry_price, entry_date, exit_price, exit_date,
                    exit_reason, realized_pnl_pct, bars_held, size)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                pos["ticker"], pos.get("entry_price", 0), pos.get("entry_date") or None,
                pos.get("exit_price", 0), pos.get("exit_date") or None,
                pos.get("exit_reason", ""), pos.get("realized_pnl_pct", 0),
                pos.get("bars_held", 0), pos.get("size", 0),
            ),
        )


def get_closed_today() -> list[dict]:
    """Get positions closed today."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM closed_positions WHERE exit_date = CURRENT_DATE"
        )
        return [row_to_dict(r) for r in cur.fetchall()]


def get_recently_closed_tickers(days: int) -> set[str]:
    """Return tickers with exits in the last N calendar days."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT ticker FROM closed_positions WHERE exit_date >= CURRENT_DATE - %s",
            (days,),
        )
        return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Bulk state operations
# ---------------------------------------------------------------------------

def load_positions() -> dict:
    """Load positions from DB. Returns {"positions": [...], "closed_today": [...]}."""
    return {
        "positions": load_all_position_meta(),
        "closed_today": get_closed_today(),
        "last_updated": datetime.now().isoformat(),
    }


def save_positions(state: dict) -> None:
    """Sync DB with the given state dict. Atomic: DELETE all open + re-insert."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM system_positions")
        for pos in state.get("positions", []):
            if not pos.get("ticker"):
                continue
            cur.execute(_UPSERT_POSITION_SQL, _position_params(pos))
        # Record any closed positions from today not yet in DB
        for closed in state.get("closed_today", []):
            if closed.get("exit_date") != today:
                continue
            cur.execute(
                "SELECT 1 FROM closed_positions WHERE ticker = %s AND exit_date = %s",
                (closed["ticker"], today),
            )
            if cur.fetchone() is None:
                cur.execute(
                    """INSERT INTO closed_positions
                           (ticker, entry_price, entry_date, exit_price, exit_date,
                            exit_reason, realized_pnl_pct, bars_held, size)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        closed["ticker"], closed.get("entry_price", 0),
                        closed.get("entry_date") or None, closed.get("exit_price", 0),
                        closed.get("exit_date") or None, closed.get("exit_reason", ""),
                        closed.get("realized_pnl_pct", 0), closed.get("bars_held", 0),
                        closed.get("size", 0),
                    ),
                )

