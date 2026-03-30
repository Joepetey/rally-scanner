"""DB helpers for operational event logging: orders, discord messages, price alerts, scheduler."""

from datetime import UTC, datetime

from db.pool import get_conn


def log_order(
    ticker: str,
    side: str,
    order_type: str,
    qty: int,
    context: str,
    order_id: str | None = None,
    status: str = "pending",
    fill_price: float | None = None,
    error: str | None = None,
) -> int:
    """Insert an order record. Returns the new row id."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO orders
                (ticker, side, order_type, qty, context, order_id, status, fill_price, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (ticker, side, order_type, qty, context, order_id, status, fill_price, error),
        )
        return cur.fetchone()["id"]


def update_order_fill(order_id: str, fill_price: float, filled_at: datetime | None = None) -> None:
    """Mark an order as filled with the confirmed fill price."""
    ts = filled_at or datetime.now(UTC)
    with get_conn() as conn:
        conn.cursor().execute(
            """
            UPDATE orders
               SET status = 'filled', fill_price = %s, filled_at = %s
             WHERE order_id = %s
            """,
            (fill_price, ts, order_id),
        )


def log_discord_message(msg_type: str, title: str | None, summary: str | None) -> None:
    """Record a Discord message send event."""
    with get_conn() as conn:
        conn.cursor().execute(
            """
            INSERT INTO discord_messages (msg_type, title, summary)
            VALUES (%s, %s, %s)
            """,
            (msg_type, title, summary),
        )


def log_price_alert(
    alert_date: str,
    ticker: str,
    alert_type: str,
    current_price: float | None,
    level_price: float | None,
    entry_price: float | None,
    pnl_pct: float | None,
) -> bool:
    """Insert a price alert record. Returns True if inserted (new alert), False if duplicate."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO price_alert_log
                (alert_date, ticker, alert_type, current_price, level_price, entry_price, pnl_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, alert_type, alert_date) DO NOTHING
            """,
            (alert_date, ticker, alert_type, current_price, level_price, entry_price, pnl_pct),
        )
        return cur.rowcount > 0


def clear_price_alerts(ticker: str, alert_date: str) -> None:
    """Delete all price alert dedup records for a ticker on a given date.

    Used by the simulation runner to reset the dedup between scenarios so
    consecutive runs of the same alert type (e.g. stop_breached) on the same
    day don't suppress each other.
    """
    with get_conn() as conn:
        conn.cursor().execute(
            "DELETE FROM price_alert_log WHERE ticker = %s AND alert_date = %s",
            (ticker, alert_date),
        )


def log_scheduler_event(event_type: str) -> int:
    """Insert a running scheduler event. Returns the row id for later update."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO scheduler_events (event_type, status)
            VALUES (%s, 'running')
            RETURNING id
            """,
            (event_type,),
        )
        return cur.fetchone()["id"]


def finish_scheduler_event(
    event_id: int,
    status: str,
    n_signals: int | None = None,
    n_exits: int | None = None,
    duration_s: float | None = None,
    error: str | None = None,
) -> None:
    """Update a scheduler event row with final status and metrics."""
    with get_conn() as conn:
        conn.cursor().execute(
            """
            UPDATE scheduler_events
               SET finished_at = NOW(),
                   status      = %s,
                   n_signals   = %s,
                   n_exits     = %s,
                   duration_s  = %s,
                   error       = %s
             WHERE id = %s
            """,
            (status, n_signals, n_exits, duration_s, error, event_id),
        )
