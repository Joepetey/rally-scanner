"""SQLite persistence for Discord bot — per-user trade tracking.

Schema:
    users  — Discord user registration (auto on first command)
    trades — Per-user trade entries with FIFO close logic

Database location: models/rally_discord.db
"""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "models" / "rally_discord.db"

_local = threading.local()


def get_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a thread-local SQLite connection."""
    path = str(db_path or DB_PATH)
    if not hasattr(_local, "connections"):
        _local.connections = {}
    if path not in _local.connections:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.connections[path] = conn
    return _local.connections[path]


def init_db(db_path: str | Path | None = None) -> None:
    """Create tables if they don't exist."""
    conn = get_db(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            discord_id   INTEGER PRIMARY KEY,
            username     TEXT NOT NULL,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            discord_id   INTEGER NOT NULL REFERENCES users(discord_id),
            ticker       TEXT NOT NULL,
            entry_price  REAL NOT NULL,
            entry_date   TEXT NOT NULL,
            exit_price   REAL,
            exit_date    TEXT,
            size         REAL DEFAULT 1.0,
            pnl_pct      REAL,
            status       TEXT NOT NULL DEFAULT 'open',
            notes        TEXT,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_trades_user_status
            ON trades(discord_id, status);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def ensure_user(discord_id: int, username: str, db_path: str | Path | None = None) -> None:
    """Insert or update a user. Called at the start of every command."""
    conn = get_db(db_path)
    conn.execute(
        "INSERT INTO users (discord_id, username) VALUES (?, ?) "
        "ON CONFLICT(discord_id) DO UPDATE SET username = excluded.username",
        (discord_id, username),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Trade CRUD
# ---------------------------------------------------------------------------

def open_trade(
    discord_id: int,
    ticker: str,
    entry_price: float,
    entry_date: str | None = None,
    size: float = 1.0,
    notes: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Record a new trade entry. Returns the trade ID."""
    conn = get_db(db_path)
    if entry_date is None:
        entry_date = datetime.now().strftime("%Y-%m-%d")
    cur = conn.execute(
        "INSERT INTO trades (discord_id, ticker, entry_price, entry_date, size, notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (discord_id, ticker.upper(), entry_price, entry_date, size, notes),
    )
    conn.commit()
    return cur.lastrowid


def close_trade(
    discord_id: int,
    ticker: str,
    exit_price: float,
    exit_date: str | None = None,
    notes: str | None = None,
    db_path: str | Path | None = None,
) -> dict | None:
    """Close the oldest open trade for this user+ticker (FIFO). Returns closed trade or None."""
    conn = get_db(db_path)
    if exit_date is None:
        exit_date = datetime.now().strftime("%Y-%m-%d")

    # Find oldest open trade for this ticker
    row = conn.execute(
        "SELECT id, entry_price, entry_date, size FROM trades "
        "WHERE discord_id = ? AND ticker = ? AND status = 'open' "
        "ORDER BY entry_date ASC, id ASC LIMIT 1",
        (discord_id, ticker.upper()),
    ).fetchone()

    if row is None:
        return None

    trade_id = row["id"]
    entry_price = row["entry_price"]
    pnl_pct = round((exit_price / entry_price - 1) * 100, 2)

    # Build notes: append exit notes to existing
    existing_notes = conn.execute(
        "SELECT notes FROM trades WHERE id = ?", (trade_id,)
    ).fetchone()["notes"]
    combined_notes = existing_notes or ""
    if notes:
        combined_notes = f"{combined_notes}; {notes}".strip("; ")

    conn.execute(
        "UPDATE trades SET exit_price = ?, exit_date = ?, pnl_pct = ?, "
        "status = 'closed', notes = ? WHERE id = ?",
        (exit_price, exit_date, pnl_pct, combined_notes or None, trade_id),
    )
    conn.commit()

    return {
        "id": trade_id,
        "ticker": ticker.upper(),
        "entry_price": entry_price,
        "entry_date": row["entry_date"],
        "exit_price": exit_price,
        "exit_date": exit_date,
        "size": row["size"],
        "pnl_pct": pnl_pct,
    }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_open_trades(
    discord_id: int, db_path: str | Path | None = None
) -> list[dict]:
    """Get all open trades for a user."""
    conn = get_db(db_path)
    rows = conn.execute(
        "SELECT id, ticker, entry_price, entry_date, size, notes "
        "FROM trades WHERE discord_id = ? AND status = 'open' "
        "ORDER BY entry_date DESC",
        (discord_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_trade_history(
    discord_id: int,
    ticker: str | None = None,
    limit: int = 50,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Get trade history for a user, optionally filtered by ticker."""
    conn = get_db(db_path)
    query = (
        "SELECT id, ticker, entry_price, entry_date, exit_price, exit_date, "
        "size, pnl_pct, status, notes FROM trades WHERE discord_id = ?"
    )
    params: list = [discord_id]

    if ticker:
        query += " AND ticker = ?"
        params.append(ticker.upper())

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_pnl_summary(
    discord_id: int,
    days: int | None = None,
    db_path: str | Path | None = None,
) -> dict:
    """Compute P&L summary for a user's closed trades."""
    conn = get_db(db_path)
    query = (
        "SELECT pnl_pct, size FROM trades "
        "WHERE discord_id = ? AND status = 'closed'"
    )
    params: list = [discord_id]

    if days:
        cutoff = datetime.now().strftime("%Y-%m-%d")
        query += " AND exit_date >= date(?, ?)"
        params.extend([cutoff, f"-{days} days"])

    rows = conn.execute(query, params).fetchall()

    if not rows:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    pnls = [r["pnl_pct"] for r in rows]
    wins = sum(1 for p in pnls if p > 0)

    return {
        "n_trades": len(pnls),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(sum(pnls) / len(pnls), 2),
        "win_rate": round(wins / len(pnls) * 100, 1),
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
    }
