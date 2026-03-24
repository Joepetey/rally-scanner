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

        # Add p_rally to system_positions if not present (migration-safe)
        cur.execute("""
            ALTER TABLE system_positions
            ADD COLUMN IF NOT EXISTS p_rally DOUBLE PRECISION DEFAULT 0
        """)

        # Add exit order tracking columns if not present (migration-safe)
        cur.execute("""
            ALTER TABLE system_positions
            ADD COLUMN IF NOT EXISTS target_order_id TEXT
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS signal_queue (
                id          BIGSERIAL PRIMARY KEY,
                ticker      TEXT NOT NULL UNIQUE,
                p_rally     DOUBLE PRECISION NOT NULL,
                comp_score  DOUBLE PRECISION NOT NULL DEFAULT 0,
                close       DOUBLE PRECISION NOT NULL,
                size        DOUBLE PRECISION NOT NULL,
                range_low   DOUBLE PRECISION NOT NULL,
                atr_pct     DOUBLE PRECISION NOT NULL DEFAULT 0,
                signal_date DATE NOT NULL,
                skip_reason TEXT NOT NULL DEFAULT 'capital',
                queued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS skipped_signals (
                id            BIGSERIAL PRIMARY KEY,
                ticker        TEXT NOT NULL,
                signal_date   DATE NOT NULL,
                p_rally       DOUBLE PRECISION NOT NULL,
                comp_score    DOUBLE PRECISION NOT NULL DEFAULT 0,
                close         DOUBLE PRECISION NOT NULL,
                size          DOUBLE PRECISION NOT NULL,
                skip_reason   TEXT NOT NULL,
                outcome_date  DATE,
                outcome_price DOUBLE PRECISION,
                outcome_pct   DOUBLE PRECISION,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (ticker, signal_date)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id           BIGSERIAL PRIMARY KEY,
                ticker       TEXT NOT NULL,
                side         TEXT NOT NULL,
                order_type   TEXT NOT NULL,
                qty          INTEGER NOT NULL DEFAULT 0,
                context      TEXT NOT NULL,
                requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                order_id     TEXT UNIQUE,
                status       TEXT NOT NULL DEFAULT 'pending',
                fill_price   DOUBLE PRECISION,
                filled_at    TIMESTAMPTZ,
                error        TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS discord_messages (
                id       BIGSERIAL PRIMARY KEY,
                sent_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                msg_type TEXT NOT NULL,
                title    TEXT,
                summary  TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_alert_log (
                id            BIGSERIAL PRIMARY KEY,
                alerted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                alert_date    DATE NOT NULL,
                ticker        TEXT NOT NULL,
                alert_type    TEXT NOT NULL,
                current_price DOUBLE PRECISION,
                level_price   DOUBLE PRECISION,
                entry_price   DOUBLE PRECISION,
                pnl_pct       DOUBLE PRECISION,
                UNIQUE (ticker, alert_type, alert_date)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_events (
                id          BIGSERIAL PRIMARY KEY,
                started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                finished_at TIMESTAMPTZ,
                event_type  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'running',
                n_signals   INTEGER,
                n_exits     INTEGER,
                duration_s  DOUBLE PRECISION,
                error       TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id          BIGSERIAL PRIMARY KEY,
                scan_date   DATE NOT NULL,
                ticker      TEXT NOT NULL,
                p_rally     DOUBLE PRECISION NOT NULL,
                comp_score  DOUBLE PRECISION NOT NULL DEFAULT 0,
                close       DOUBLE PRECISION NOT NULL,
                is_signal   BOOLEAN NOT NULL DEFAULT FALSE,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (scan_date, ticker)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_manifest (
                ticker      TEXT PRIMARY KEY,
                saved_at    TIMESTAMPTZ NOT NULL,
                train_start TEXT NOT NULL,
                train_end   TEXT NOT NULL,
                r_up        DOUBLE PRECISION NOT NULL,
                d_dn        DOUBLE PRECISION NOT NULL,
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS regime_states (
                ticker          TEXT PRIMARY KEY,
                p_compressed    DOUBLE PRECISION NOT NULL,
                p_normal        DOUBLE PRECISION NOT NULL,
                p_expanding     DOUBLE PRECISION NOT NULL,
                dominant_regime TEXT NOT NULL,
                recorded_at     TIMESTAMPTZ NOT NULL,
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS universe_cache (
                id         INT PRIMARY KEY DEFAULT 1,
                tickers    TEXT[] NOT NULL,
                source     TEXT NOT NULL,
                fetched_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CHECK (id = 1)
            )
        """)
