---
name: PostgreSQL migration
description: SQLite/CSV replaced with PostgreSQL; all DB functions in src/db/
type: project
---

Migrated from SQLite + CSV files to PostgreSQL (Railway-hosted).

**Why:** Decouple storage from Railway volume, get proper ACID guarantees, queryable audit trail.

**How to apply:** When adding new persistence, put DB functions in `src/db/`, not in the calling module.

## New Structure: `src/db/`
- `pool.py` — `ThreadedConnectionPool`, `get_conn()` context manager, `row_to_dict()` helper
- `schema.py` — `init_schema()` runs DDL at startup (idempotent)
- `users.py` — `ensure_user`, `get_capital`, `set_capital`
- `trades.py` — `open_trade`, `close_trade`, `get_open_trades`, `get_trade_history`, `get_pnl_summary`
- `conversations.py` — `get/save/clear_conversation_history`
- `positions.py` — all system position CRUD (`save_position_meta`, `load_all_position_meta`, `save_positions`, etc.)
- `portfolio.py` — `save_snapshot`, `record_closed_trades`, `load_equity_history`, `load_trade_journal`, HWM

## Tables
users, trades, conversation_history, system_positions, closed_positions, portfolio_snapshots, trade_journal, portfolio_meta

## Deleted
- `src/bot/discord_db.py` — all functions moved to `src/db/`
- `models/rally_discord.db`, CSV files, `high_water_mark.txt` — replaced by PostgreSQL

## Tests
- DB tests require `TEST_DATABASE_URL` env var; skip gracefully if absent
- `pg_db` fixture in conftest: initializes pool, truncates tables before/after each test
- `tmp_models_dir` fixture now depends on `pg_db` (most position/portfolio tests need both)

## Railway Deployment
- Add PostgreSQL service to `satisfied-communication` project → auto-injects `DATABASE_URL`
- Optionally run `railway run python scripts/migrate_system_state.py` to copy open positions from old SQLite
