"""Signal queue and skipped-signal CRUD against PostgreSQL."""

import logging
from datetime import datetime

from db.pool import get_conn, row_to_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal queue CRUD
# ---------------------------------------------------------------------------

def enqueue_signal(sig: dict, reason: str) -> None:
    """Add or replace a signal in the queue (UPSERT by ticker)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO signal_queue
                   (ticker, p_rally, comp_score, close, size, range_low, atr_pct,
                    signal_date, skip_reason, queued_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
               ON CONFLICT (ticker) DO UPDATE SET
                   p_rally=EXCLUDED.p_rally, comp_score=EXCLUDED.comp_score,
                   close=EXCLUDED.close, size=EXCLUDED.size,
                   range_low=EXCLUDED.range_low, atr_pct=EXCLUDED.atr_pct,
                   signal_date=EXCLUDED.signal_date, skip_reason=EXCLUDED.skip_reason,
                   queued_at=NOW()""",
            (
                sig["ticker"], sig.get("p_rally", 0), sig.get("comp_score", 0),
                sig["close"], sig["size"], sig.get("range_low", 0), sig.get("atr_pct", 0),
                sig.get("date", datetime.now().strftime("%Y-%m-%d")), reason,
            ),
        )


def dequeue_signals(max_age_days: int) -> list[dict]:
    """Return valid queued signals ordered by p_rally desc (best first)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM signal_queue
               WHERE signal_date >= CURRENT_DATE - (%s || ' days')::interval
               ORDER BY p_rally DESC""",
            (str(max_age_days),),
        )
        return [row_to_dict(r) for r in cur.fetchall()]


def remove_from_queue(ticker: str) -> None:
    """Remove a ticker from the signal queue."""
    with get_conn() as conn:
        conn.cursor().execute(
            "DELETE FROM signal_queue WHERE ticker = %s", (ticker,),
        )


def clear_expired_queue(max_age_days: int) -> int:
    """Delete expired queue entries. Returns number removed."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """DELETE FROM signal_queue
               WHERE signal_date < CURRENT_DATE - (%s || ' days')::interval""",
            (str(max_age_days),),
        )
        return cur.rowcount


# ---------------------------------------------------------------------------
# Skipped signals CRUD (opportunity cost tracking)
# ---------------------------------------------------------------------------

def log_skipped_signal(sig: dict, reason: str) -> None:
    """Record a signal that was skipped (no-op on duplicate)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO skipped_signals
                   (ticker, signal_date, p_rally, comp_score, close, size, skip_reason)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (ticker, signal_date) DO NOTHING""",
            (
                sig["ticker"],
                sig.get("date", datetime.now().strftime("%Y-%m-%d")),
                sig.get("p_rally", 0), sig.get("comp_score", 0),
                sig["close"], sig.get("size", 0), reason,
            ),
        )


def get_unevaluated_skipped() -> list[dict]:
    """Return skipped signals from prior days without outcome data."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM skipped_signals
               WHERE outcome_date IS NULL AND signal_date < CURRENT_DATE
               ORDER BY signal_date DESC"""
        )
        return [row_to_dict(r) for r in cur.fetchall()]


def update_skipped_outcome(
    ticker: str, signal_date: str, outcome_price: float, outcome_pct: float,
) -> None:
    """Fill in outcome fields for a previously skipped signal."""
    with get_conn() as conn:
        conn.cursor().execute(
            """UPDATE skipped_signals
               SET outcome_date = CURRENT_DATE, outcome_price = %s, outcome_pct = %s
               WHERE ticker = %s AND signal_date = %s""",
            (outcome_price, outcome_pct, ticker, signal_date),
        )
