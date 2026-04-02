#!/usr/bin/env python
"""
One-shot migration: copy system trading state from SQLite/CSV to PostgreSQL.

Migrates:
  - system_positions (open positions — must carry over)
  - closed_positions  (historical record)
  - equity_history.csv → portfolio_snapshots
  - trade_journal.csv  → trade_journal
  - high_water_mark.txt → portfolio_meta

Does NOT migrate: users, trades, conversation_history (starting fresh).

Usage:
    DATABASE_URL=postgresql://... python scripts/migrate_system_state.py
    # or with a local .env:
    python scripts/migrate_system_state.py
"""

import csv
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DB_PATH = PROJECT_ROOT / "models" / "rally_discord.db"
EQUITY_LOG = PROJECT_ROOT / "models" / "equity_history.csv"
TRADE_JOURNAL = PROJECT_ROOT / "models" / "trade_journal.csv"
HWM_FILE = PROJECT_ROOT / "models" / "high_water_mark.txt"


def main() -> None:
    from db import init_pool, init_schema
    from db.core.pool import get_conn

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set.")
        sys.exit(1)

    print("Initializing PostgreSQL pool and schema...")
    init_pool()
    init_schema()

    if not DB_PATH.exists():
        print(f"SQLite DB not found at {DB_PATH} — skipping SQLite migration.")
    else:
        _migrate_sqlite(get_conn)

    if EQUITY_LOG.exists():
        _migrate_equity_history(get_conn)
    else:
        print(f"No equity_history.csv at {EQUITY_LOG} — skipping.")

    if TRADE_JOURNAL.exists():
        _migrate_trade_journal(get_conn)
    else:
        print(f"No trade_journal.csv at {TRADE_JOURNAL} — skipping.")

    if HWM_FILE.exists():
        _migrate_high_water_mark(get_conn)
    else:
        print(f"No high_water_mark.txt at {HWM_FILE} — skipping.")

    print("Migration complete.")


def _migrate_sqlite(get_conn) -> None:
    print(f"Connecting to SQLite: {DB_PATH}")
    src = sqlite3.connect(DB_PATH)
    src.row_factory = sqlite3.Row

    # system_positions
    rows = src.execute("SELECT * FROM system_positions").fetchall()
    print(f"  Migrating {len(rows)} open positions...")
    with get_conn() as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute(
                """INSERT INTO system_positions
                       (ticker, entry_price, entry_date, stop_price, target_price,
                        trailing_stop, highest_close, atr, bars_held, size, qty,
                        order_id, trail_order_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (ticker) DO NOTHING""",
                (
                    r["ticker"], r["entry_price"], r["entry_date"],
                    r["stop_price"], r["target_price"], r["trailing_stop"],
                    r["highest_close"], r["atr"], r["bars_held"],
                    r["size"], r["qty"], r["order_id"], r["trail_order_id"],
                ),
            )
    print(f"  ✓ {len(rows)} open positions migrated.")

    # closed_positions
    rows = src.execute("SELECT * FROM closed_positions").fetchall()
    print(f"  Migrating {len(rows)} closed positions...")
    with get_conn() as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute(
                """INSERT INTO closed_positions
                       (ticker, entry_price, entry_date, exit_price, exit_date,
                        exit_reason, realized_pnl_pct, bars_held, size)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    r["ticker"], r["entry_price"], r["entry_date"],
                    r["exit_price"], r["exit_date"], r["exit_reason"],
                    r["realized_pnl_pct"], r["bars_held"], r["size"],
                ),
            )
    print(f"  ✓ {len(rows)} closed positions migrated.")

    src.close()


def _migrate_equity_history(get_conn) -> None:
    with open(EQUITY_LOG) as f:
        rows = list(csv.DictReader(f))
    print(f"  Migrating {len(rows)} equity snapshots...")
    with get_conn() as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute(
                """INSERT INTO portfolio_snapshots
                       (snapshot_date, n_positions, total_exposure,
                        total_unrealized_pnl_pct, n_signals_today, n_scanned)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (snapshot_date) DO NOTHING""",
                (
                    r["date"], int(r["n_positions"]), float(r["total_exposure"]),
                    float(r["total_unrealized_pnl_pct"]),
                    int(r["n_signals_today"]), int(r["n_scanned"]),
                ),
            )
    print(f"  ✓ {len(rows)} equity snapshots migrated.")


def _migrate_trade_journal(get_conn) -> None:
    with open(TRADE_JOURNAL) as f:
        rows = list(csv.DictReader(f))
    print(f"  Migrating {len(rows)} trade journal entries...")
    with get_conn() as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute(
                """INSERT INTO trade_journal
                       (exit_date, ticker, entry_date, entry_price, exit_price,
                        realized_pnl_pct, size, bars_held, exit_reason)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    r["exit_date"], r["ticker"], r["entry_date"],
                    float(r["entry_price"]), float(r["exit_price"]),
                    float(r["realized_pnl_pct"]) if r.get("realized_pnl_pct") else None,
                    float(r["size"]) if r.get("size") else None,
                    int(r["bars_held"]) if r.get("bars_held") else None,
                    r["exit_reason"],
                ),
            )
    print(f"  ✓ {len(rows)} trade journal entries migrated.")


def _migrate_high_water_mark(get_conn) -> None:
    try:
        hwm = float(HWM_FILE.read_text().strip())
    except (ValueError, OSError):
        print(f"  Could not read {HWM_FILE} — skipping.")
        return
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE portfolio_meta SET value = %s WHERE key = 'high_water_mark'",
            (str(hwm),),
        )
    print(f"  ✓ High-water mark migrated: {hwm}")


if __name__ == "__main__":
    main()
