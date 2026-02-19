"""SQLite persistence for Discord bot — per-user trade tracking.

Schema:
    users  — Discord user registration (auto on first command)
    trades — Per-user trade entries with FIFO close logic
    conversation_history — Claude conversation state per user

Database location: models/rally_discord.db
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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
            capital      REAL DEFAULT 0.0,
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
            stop_price   REAL,
            target_price REAL,
            size         REAL DEFAULT 1.0,
            pnl_pct      REAL,
            pnl_dollar   REAL,
            status       TEXT NOT NULL DEFAULT 'open',
            notes        TEXT,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS conversation_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            discord_id   INTEGER NOT NULL REFERENCES users(discord_id),
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            history      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_user_status
            ON trades(discord_id, status);

        CREATE INDEX IF NOT EXISTS idx_conversation_user
            ON conversation_history(discord_id, created_at DESC);
    """)
    # Migrate existing databases: add new columns if missing
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema."""
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    if "capital" not in existing:
        conn.execute("ALTER TABLE users ADD COLUMN capital REAL DEFAULT 0.0")

    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()
    }
    for col, typ in [
        ("stop_price", "REAL"),
        ("target_price", "REAL"),
        ("pnl_dollar", "REAL"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")


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

def set_capital(
    discord_id: int,
    capital: float,
    db_path: str | Path | None = None,
) -> None:
    """Set the user's portfolio capital."""
    conn = get_db(db_path)
    conn.execute(
        "UPDATE users SET capital = ? WHERE discord_id = ?",
        (capital, discord_id),
    )
    conn.commit()


def get_capital(
    discord_id: int,
    db_path: str | Path | None = None,
) -> float:
    """Get the user's portfolio capital. Returns 0.0 if not set."""
    conn = get_db(db_path)
    row = conn.execute(
        "SELECT capital FROM users WHERE discord_id = ?", (discord_id,)
    ).fetchone()
    return float(row["capital"]) if row and row["capital"] else 0.0


def open_trade(
    discord_id: int,
    ticker: str,
    entry_price: float,
    entry_date: str | None = None,
    size: float = 1.0,
    stop_price: float | None = None,
    target_price: float | None = None,
    notes: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Record a new trade entry. Returns the trade ID."""
    conn = get_db(db_path)
    if entry_date is None:
        entry_date = datetime.now().strftime("%Y-%m-%d")
    cur = conn.execute(
        "INSERT INTO trades (discord_id, ticker, entry_price, entry_date, "
        "size, stop_price, target_price, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (discord_id, ticker.upper(), entry_price, entry_date, size,
         stop_price, target_price, notes),
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
        "SELECT id, entry_price, entry_date, size, stop_price, target_price "
        "FROM trades "
        "WHERE discord_id = ? AND ticker = ? AND status = 'open' "
        "ORDER BY entry_date ASC, id ASC LIMIT 1",
        (discord_id, ticker.upper()),
    ).fetchone()

    if row is None:
        return None

    trade_id = row["id"]
    entry_price = row["entry_price"]
    size = row["size"]
    pnl_pct = round((exit_price / entry_price - 1) * 100, 2)

    # Dollar PnL: uses capital * size fraction * return
    capital = get_capital(discord_id, db_path)
    pnl_dollar = None
    if capital > 0 and size > 0:
        pnl_dollar = round(capital * size * (exit_price / entry_price - 1), 2)

    # Build notes: append exit notes to existing
    existing_notes = conn.execute(
        "SELECT notes FROM trades WHERE id = ?", (trade_id,)
    ).fetchone()["notes"]
    combined_notes = existing_notes or ""
    if notes:
        combined_notes = f"{combined_notes}; {notes}".strip("; ")

    conn.execute(
        "UPDATE trades SET exit_price = ?, exit_date = ?, pnl_pct = ?, "
        "pnl_dollar = ?, status = 'closed', notes = ? WHERE id = ?",
        (exit_price, exit_date, pnl_pct, pnl_dollar, combined_notes or None,
         trade_id),
    )
    conn.commit()

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


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_open_trades(
    discord_id: int, db_path: str | Path | None = None
) -> list[dict]:
    """Get all open trades for a user."""
    conn = get_db(db_path)
    rows = conn.execute(
        "SELECT id, ticker, entry_price, entry_date, size, "
        "stop_price, target_price, notes "
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
        "size, stop_price, target_price, pnl_pct, pnl_dollar, status, notes "
        "FROM trades WHERE discord_id = ?"
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
        "SELECT pnl_pct, pnl_dollar, size FROM trades "
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


# ---------------------------------------------------------------------------
# Conversation History
# ---------------------------------------------------------------------------

def get_conversation_history(
    discord_id: int,
    limit_messages: int = 20,
    db_path: str | Path | None = None
) -> list[dict]:
    """Get recent conversation history for a user.

    Args:
        discord_id: Discord user ID
        limit_messages: Max messages to return (prevents token overflow)
        db_path: Optional database path

    Returns:
        List of message dicts in Claude API format
    """
    conn = get_db(db_path)
    row = conn.execute(
        "SELECT history FROM conversation_history "
        "WHERE discord_id = ? "
        "ORDER BY created_at DESC "
        "LIMIT 1",
        (discord_id,),
    ).fetchone()

    if row:
        history = json.loads(row["history"])
        # Limit to recent messages to avoid token overflow
        history = history[-limit_messages:]
        # Ensure truncation doesn't break tool_use/tool_result pairs.
        # The Claude API requires: (1) first message is role="user",
        # (2) every tool_result has a matching tool_use in the prior
        # assistant message.  Walk forward to find a safe start.
        def _is_tool_result(msg):
            content = msg.get("content")
            return isinstance(content, list) and any(
                isinstance(c, dict) and c.get("type") == "tool_result"
                for c in content
            )

        def _is_plain_user(msg):
            return msg.get("role") == "user" and not _is_tool_result(msg)

        # Find first plain user message — the only safe start boundary
        for i, msg in enumerate(history):
            if _is_plain_user(msg):
                return history[i:]
        # No safe start found — drop the whole slice rather than
        # sending malformed history to the API
        return []
    return []


def save_conversation_history(
    discord_id: int,
    history: list[dict],
    db_path: str | Path | None = None
) -> None:
    """Save updated conversation history for a user.

    Args:
        discord_id: Discord user ID
        history: Full conversation history as list of message dicts
        db_path: Optional database path
    """
    conn = get_db(db_path)
    conn.execute(
        "INSERT INTO conversation_history (discord_id, history) VALUES (?, ?)",
        (discord_id, json.dumps(history)),
    )
    conn.commit()


def clear_conversation_history(
    discord_id: int,
    db_path: str | Path | None = None
) -> None:
    """Clear conversation history for a user (start fresh conversation).

    Args:
        discord_id: Discord user ID
        db_path: Optional database path
    """
    conn = get_db(db_path)
    conn.execute(
        "DELETE FROM conversation_history WHERE discord_id = ?",
        (discord_id,),
    )
    conn.commit()
