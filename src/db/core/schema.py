"""PostgreSQL schema initialization — runs CREATE TABLE IF NOT EXISTS at startup."""

from db.core.pool import get_conn


def init_schema() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    with get_conn() as conn:
        cur = conn.cursor()
        _init_user_tables(cur)
        _init_position_tables(cur)
        _init_signal_tables(cur)
        _init_scan_tables(cur)
        _init_event_tables(cur)
        _init_portfolio_tables(cur)
        _init_ml_tables(cur)
        _run_migrations(cur)


# ---------------------------------------------------------------------------
# Domain-specific table creation
# ---------------------------------------------------------------------------


def _init_user_tables(cur) -> None:
    """Users and conversation history."""
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
            discord_id   BIGINT,
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


def _init_position_tables(cur) -> None:
    """System positions (open + closed) and signal queue."""
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


def _init_signal_tables(cur) -> None:
    """Signal queue, skipped signals, and orders."""
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
        CREATE TABLE IF NOT EXISTS current_signals (
            ticker       TEXT PRIMARY KEY,
            p_rally      DOUBLE PRECISION NOT NULL DEFAULT 0,
            comp_score   DOUBLE PRECISION NOT NULL DEFAULT 0,
            fail_dn      DOUBLE PRECISION NOT NULL DEFAULT 0,
            close        DOUBLE PRECISION NOT NULL DEFAULT 0,
            size         DOUBLE PRECISION NOT NULL DEFAULT 0,
            atr_pct      DOUBLE PRECISION NOT NULL DEFAULT 0,
            range_low    DOUBLE PRECISION NOT NULL DEFAULT 0,
            trend        INTEGER NOT NULL DEFAULT 0,
            golden_cross INTEGER NOT NULL DEFAULT 0,
            rsi          DOUBLE PRECISION NOT NULL DEFAULT 0,
            is_position  BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)


def _init_scan_tables(cur) -> None:
    """Watchlist, scan results, and current watchlist snapshot."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id            BIGSERIAL PRIMARY KEY,
            scan_date     DATE NOT NULL,
            ticker        TEXT NOT NULL,
            p_rally       DOUBLE PRECISION NOT NULL,
            p_rally_raw   DOUBLE PRECISION NOT NULL DEFAULT 0,
            comp_score    DOUBLE PRECISION NOT NULL DEFAULT 0,
            fail_dn       DOUBLE PRECISION NOT NULL DEFAULT 0,
            trend         INTEGER NOT NULL DEFAULT 0,
            golden_cross  INTEGER NOT NULL DEFAULT 0,
            hmm_compressed DOUBLE PRECISION NOT NULL DEFAULT 0,
            rv_pctile     DOUBLE PRECISION NOT NULL DEFAULT 0,
            atr_pct       DOUBLE PRECISION NOT NULL DEFAULT 0,
            macd_hist     DOUBLE PRECISION NOT NULL DEFAULT 0,
            vol_ratio     DOUBLE PRECISION NOT NULL DEFAULT 1,
            vix_pctile    DOUBLE PRECISION NOT NULL DEFAULT 0,
            rsi           DOUBLE PRECISION NOT NULL DEFAULT 0,
            close         DOUBLE PRECISION NOT NULL,
            size          DOUBLE PRECISION NOT NULL DEFAULT 0,
            is_signal     BOOLEAN NOT NULL DEFAULT FALSE,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (scan_date, ticker)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS latest_scan_results (
            ticker          TEXT PRIMARY KEY,
            p_rally         DOUBLE PRECISION NOT NULL DEFAULT 0,
            p_rally_raw     DOUBLE PRECISION NOT NULL DEFAULT 0,
            comp_score      DOUBLE PRECISION NOT NULL DEFAULT 0,
            fail_dn         DOUBLE PRECISION NOT NULL DEFAULT 0,
            trend           INTEGER NOT NULL DEFAULT 0,
            golden_cross    INTEGER NOT NULL DEFAULT 0,
            hmm_compressed  DOUBLE PRECISION NOT NULL DEFAULT 0,
            rv_pctile       DOUBLE PRECISION NOT NULL DEFAULT 0,
            atr_pct         DOUBLE PRECISION NOT NULL DEFAULT 0,
            macd_hist       DOUBLE PRECISION NOT NULL DEFAULT 0,
            vol_ratio       DOUBLE PRECISION NOT NULL DEFAULT 1,
            vix_pctile      DOUBLE PRECISION NOT NULL DEFAULT 0,
            rsi             DOUBLE PRECISION NOT NULL DEFAULT 0,
            close           DOUBLE PRECISION NOT NULL DEFAULT 0,
            size            DOUBLE PRECISION NOT NULL DEFAULT 0,
            is_signal       BOOLEAN NOT NULL DEFAULT FALSE,
            is_position     BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS current_watchlist (
            ticker         TEXT PRIMARY KEY,
            p_rally        DOUBLE PRECISION NOT NULL DEFAULT 0,
            p_rally_raw    DOUBLE PRECISION NOT NULL DEFAULT 0,
            comp_score     DOUBLE PRECISION NOT NULL DEFAULT 0,
            fail_dn        DOUBLE PRECISION NOT NULL DEFAULT 0,
            close          DOUBLE PRECISION NOT NULL DEFAULT 0,
            size           DOUBLE PRECISION NOT NULL DEFAULT 0,
            trend          INTEGER NOT NULL DEFAULT 0,
            golden_cross   INTEGER NOT NULL DEFAULT 0,
            hmm_compressed DOUBLE PRECISION NOT NULL DEFAULT 0,
            rv_pctile      DOUBLE PRECISION NOT NULL DEFAULT 0,
            atr_pct        DOUBLE PRECISION NOT NULL DEFAULT 0,
            macd_hist      DOUBLE PRECISION NOT NULL DEFAULT 0,
            vol_ratio      DOUBLE PRECISION NOT NULL DEFAULT 1,
            vix_pctile     DOUBLE PRECISION NOT NULL DEFAULT 0,
            rsi            DOUBLE PRECISION NOT NULL DEFAULT 0,
            updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)


def _init_event_tables(cur) -> None:
    """Discord messages, price alerts, and scheduler events."""
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


def _init_portfolio_tables(cur) -> None:
    """Portfolio snapshots, trade journal, and metadata."""
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


def _init_ml_tables(cur) -> None:
    """Model manifest, regime states, and universe cache."""
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


# ---------------------------------------------------------------------------
# Migrations — ALTER TABLE for existing deployments
# ---------------------------------------------------------------------------


def _run_migrations(cur) -> None:
    """Add columns and alter constraints for existing tables.

    All migrations use IF NOT EXISTS / IF EXISTS so they're idempotent.
    """
    # system_positions: add columns added after initial deployment
    cur.execute("""
        ALTER TABLE system_positions
        ADD COLUMN IF NOT EXISTS p_rally DOUBLE PRECISION DEFAULT 0
    """)
    cur.execute("""
        ALTER TABLE system_positions
        ADD COLUMN IF NOT EXISTS current_price DOUBLE PRECISION DEFAULT 0
    """)
    cur.execute("""
        ALTER TABLE system_positions
        ADD COLUMN IF NOT EXISTS unrealized_pnl_pct DOUBLE PRECISION DEFAULT 0
    """)
    cur.execute("""
        ALTER TABLE system_positions
        ADD COLUMN IF NOT EXISTS target_order_id TEXT
    """)

    # current_signals: add range_low if missing
    cur.execute("""
        ALTER TABLE current_signals
        ADD COLUMN IF NOT EXISTS range_low DOUBLE PRECISION NOT NULL DEFAULT 0
    """)

    # watchlist: add columns added after initial deployment
    for col, definition in [
        ("p_rally_raw",    "DOUBLE PRECISION NOT NULL DEFAULT 0"),
        ("fail_dn",        "DOUBLE PRECISION NOT NULL DEFAULT 0"),
        ("trend",          "INTEGER NOT NULL DEFAULT 0"),
        ("golden_cross",   "INTEGER NOT NULL DEFAULT 0"),
        ("hmm_compressed", "DOUBLE PRECISION NOT NULL DEFAULT 0"),
        ("rv_pctile",      "DOUBLE PRECISION NOT NULL DEFAULT 0"),
        ("atr_pct",        "DOUBLE PRECISION NOT NULL DEFAULT 0"),
        ("macd_hist",      "DOUBLE PRECISION NOT NULL DEFAULT 0"),
        ("vol_ratio",      "DOUBLE PRECISION NOT NULL DEFAULT 1"),
        ("vix_pctile",     "DOUBLE PRECISION NOT NULL DEFAULT 0"),
        ("rsi",            "DOUBLE PRECISION NOT NULL DEFAULT 0"),
        ("size",           "DOUBLE PRECISION NOT NULL DEFAULT 0"),
    ]:
        cur.execute(
            f"ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS {col} {definition}"
        )

    # trades: relax discord_id constraint (single-user mode)
    cur.execute("ALTER TABLE trades ALTER COLUMN discord_id DROP NOT NULL")
    cur.execute("ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_discord_id_fkey")
