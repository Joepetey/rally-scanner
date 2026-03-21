"""PostgreSQL schema initialization — runs CREATE TABLE IF NOT EXISTS at startup."""

from db.pool import get_conn


def init_schema() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                discord_id  BIGINT PRIMARY KEY,
                username    TEXT NOT NULL,
                capital     DOUBLE PRECISION DEFAULT 0.0,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id           BIGSERIAL PRIMARY KEY,
                discord_id   BIGINT NOT NULL REFERENCES users(discord_id),
                ticker       TEXT NOT NULL,
                entry_price  DOUBLE PRECISION NOT NULL,
                entry_date   DATE NOT NULL,
                exit_price   DOUBLE PRECISION,
                exit_date    DATE,
                stop_price   DOUBLE PRECISION,
                target_price DOUBLE PRECISION,
                size         DOUBLE PRECISION DEFAULT 1.0,
                pnl_pct      DOUBLE PRECISION,
                pnl_dollar   DOUBLE PRECISION,
                status       TEXT NOT NULL DEFAULT 'open',
                notes        TEXT,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_user_status
                ON trades(discord_id, status)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                discord_id  BIGINT PRIMARY KEY REFERENCES users(discord_id),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                history     TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_positions (
                ticker          TEXT PRIMARY KEY,
                entry_price     DOUBLE PRECISION NOT NULL,
                entry_date      DATE NOT NULL,
                stop_price      DOUBLE PRECISION DEFAULT 0,
                target_price    DOUBLE PRECISION DEFAULT 0,
                trailing_stop   DOUBLE PRECISION DEFAULT 0,
                highest_close   DOUBLE PRECISION DEFAULT 0,
                atr             DOUBLE PRECISION DEFAULT 0,
                bars_held       INTEGER DEFAULT 0,
                size            DOUBLE PRECISION DEFAULT 0,
                qty             INTEGER DEFAULT 0,
                order_id        TEXT,
                trail_order_id  TEXT,
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS closed_positions (
                id               BIGSERIAL PRIMARY KEY,
                ticker           TEXT NOT NULL,
                entry_price      DOUBLE PRECISION NOT NULL,
                entry_date       DATE NOT NULL,
                exit_price       DOUBLE PRECISION NOT NULL,
                exit_date        DATE NOT NULL,
                exit_reason      TEXT NOT NULL,
                realized_pnl_pct DOUBLE PRECISION,
                bars_held        INTEGER DEFAULT 0,
                size             DOUBLE PRECISION DEFAULT 0,
                created_at       TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id                       BIGSERIAL PRIMARY KEY,
                snapshot_date            DATE NOT NULL UNIQUE,
                n_positions              INTEGER NOT NULL DEFAULT 0,
                total_exposure           DOUBLE PRECISION NOT NULL DEFAULT 0,
                total_unrealized_pnl_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
                n_signals_today          INTEGER NOT NULL DEFAULT 0,
                n_scanned                INTEGER NOT NULL DEFAULT 0,
                created_at               TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id               BIGSERIAL PRIMARY KEY,
                exit_date        DATE NOT NULL,
                ticker           TEXT NOT NULL,
                entry_date       DATE NOT NULL,
                entry_price      DOUBLE PRECISION NOT NULL,
                exit_price       DOUBLE PRECISION NOT NULL,
                realized_pnl_pct DOUBLE PRECISION,
                size             DOUBLE PRECISION,
                bars_held        INTEGER,
                exit_reason      TEXT NOT NULL,
                created_at       TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_meta (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            INSERT INTO portfolio_meta (key, value)
            VALUES ('high_water_mark', '0.0')
            ON CONFLICT (key) DO NOTHING
        """)
