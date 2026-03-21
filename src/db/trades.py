"""Per-user trade tracking — FIFO open/close with P&L calculation."""

from datetime import datetime

from db.pool import get_conn, row_to_dict
from db.users import get_capital


def open_trade(
    discord_id: int,
    ticker: str,
    entry_price: float,
    entry_date: str | None = None,
    size: float = 1.0,
    stop_price: float | None = None,
    target_price: float | None = None,
    notes: str | None = None,
) -> int:
    """Record a new trade entry. Returns the trade ID."""
    if entry_date is None:
        entry_date = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO trades (discord_id, ticker, entry_price, entry_date, "
            "size, stop_price, target_price, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (discord_id, ticker.upper(), entry_price, entry_date, size,
             stop_price, target_price, notes),
        )
        return cur.fetchone()["id"]


def close_trade(
    discord_id: int,
    ticker: str,
    exit_price: float,
    exit_date: str | None = None,
    notes: str | None = None,
) -> dict | None:
    """Close the oldest open trade for this user+ticker (FIFO). Returns closed trade or None."""
    if exit_date is None:
        exit_date = datetime.now().strftime("%Y-%m-%d")

    with get_conn() as conn:
        cur = conn.cursor()

        # Find oldest open trade for this ticker
        cur.execute(
            "SELECT id, entry_price, entry_date, size, stop_price, target_price, notes "
            "FROM trades "
            "WHERE discord_id = %s AND ticker = %s AND status = 'open' "
            "ORDER BY entry_date ASC, id ASC LIMIT 1",
            (discord_id, ticker.upper()),
        )
        row = cur.fetchone()
        if row is None:
            return None

        row = row_to_dict(row)
        trade_id = row["id"]
        entry_price = row["entry_price"]
        size = row["size"]
        pnl_pct = round((exit_price / entry_price - 1) * 100, 2)

        # Dollar PnL
        capital = get_capital(discord_id)
        pnl_dollar = None
        if capital > 0 and size > 0:
            pnl_dollar = round(capital * size * (exit_price / entry_price - 1), 2)

        # Merge notes
        combined_notes = row.get("notes") or ""
        if notes:
            combined_notes = f"{combined_notes}; {notes}".strip("; ")

        cur.execute(
            "UPDATE trades SET exit_price = %s, exit_date = %s, pnl_pct = %s, "
            "pnl_dollar = %s, status = 'closed', notes = %s WHERE id = %s",
            (exit_price, exit_date, pnl_pct, pnl_dollar, combined_notes or None,
             trade_id),
        )

    return {
        "id": trade_id,
        "ticker": ticker.upper(),
        "entry_price": entry_price,
        "entry_date": row["entry_date"],
        "exit_price": exit_price,
        "exit_date": exit_date,
        "size": size,
        "stop_price": row["stop_price"],
        "target_price": row["target_price"],
        "pnl_pct": pnl_pct,
        "pnl_dollar": pnl_dollar,
    }


def get_open_trades(discord_id: int) -> list[dict]:
    """Get all open trades for a user."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, ticker, entry_price, entry_date, size, "
            "stop_price, target_price, notes "
            "FROM trades WHERE discord_id = %s AND status = 'open' "
            "ORDER BY entry_date DESC",
            (discord_id,),
        )
        return [row_to_dict(r) for r in cur.fetchall()]


def get_trade_history(
    discord_id: int,
    ticker: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Get trade history for a user, optionally filtered by ticker."""
    query = (
        "SELECT id, ticker, entry_price, entry_date, exit_price, exit_date, "
        "size, stop_price, target_price, pnl_pct, pnl_dollar, status, notes "
        "FROM trades WHERE discord_id = %s"
    )
    params: list = [discord_id]

    if ticker:
        query += " AND ticker = %s"
        params.append(ticker.upper())

    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        return [row_to_dict(r) for r in cur.fetchall()]


def get_pnl_summary(discord_id: int, days: int | None = None) -> dict:
    """Compute P&L summary for a user's closed trades."""
    query = (
        "SELECT pnl_pct, pnl_dollar, size FROM trades "
        "WHERE discord_id = %s AND status = 'closed'"
    )
    params: list = [discord_id]

    if days:
        query += " AND exit_date >= CURRENT_DATE - %s * INTERVAL '1 day'"
        params.append(days)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()

    if not rows:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "total_pnl_dollar": 0.0,
        }

    pnls = [r["pnl_pct"] for r in rows]
    dollar_pnls = [r["pnl_dollar"] for r in rows if r["pnl_dollar"] is not None]
    wins = sum(1 for p in pnls if p > 0)

    return {
        "n_trades": len(pnls),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(sum(pnls) / len(pnls), 2),
        "win_rate": round(wins / len(pnls) * 100, 1),
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "total_pnl_dollar": round(sum(dollar_pnls), 2) if dollar_pnls else 0.0,
    }
