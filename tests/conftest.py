"""Shared test fixtures — synthetic OHLCV data and PostgreSQL test DB."""

import os

import numpy as np
import pandas as pd
import pytest

from tests.helpers.alpaca_mock import AlpacaMock


@pytest.fixture
def ohlcv_df():
    """Synthetic OHLCV DataFrame with 500 bars of random-walk price data.

    Produces realistic-looking data with proper OHLCV relationships:
    - Close follows a random walk with slight upward drift
    - High >= max(Open, Close), Low <= min(Open, Close)
    - Volume is random positive integers
    """
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2020-01-02", periods=n)

    # Random walk close
    returns = np.random.normal(0.0005, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(returns))

    # Open: previous close + small gap
    open_ = np.roll(close, 1) * (1 + np.random.normal(0, 0.002, n))
    open_[0] = close[0] * 0.999

    # High/Low: extend beyond open/close
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max + np.abs(np.random.normal(0, 0.005, n)) * close
    low = bar_min - np.abs(np.random.normal(0, 0.005, n)) * close

    volume = np.random.randint(1_000_000, 50_000_000, n)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


@pytest.fixture
def ohlcv_with_vix(ohlcv_df):
    """OHLCV DataFrame with a VIX_Close column added."""
    np.random.seed(99)
    n = len(ohlcv_df)
    vix = 15 + np.cumsum(np.random.normal(0, 0.5, n))
    vix = np.clip(vix, 9, 80)
    ohlcv_df["VIX_Close"] = vix
    return ohlcv_df


@pytest.fixture
def pg_db():
    """Initialize PostgreSQL pool and truncate all tables before each test.

    Requires TEST_DATABASE_URL env var. Skips the test if not set.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set — skipping DB test")

    os.environ["DATABASE_URL"] = url

    import db.core.pool as pool_mod
    from db import init_pool, init_schema
    from db.core.pool import get_conn

    # Re-initialize pool so it uses TEST_DATABASE_URL
    if pool_mod._pool is not None:
        pool_mod._pool.closeall()
    pool_mod._pool = None
    init_pool()
    init_schema()

    def _truncate():
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                TRUNCATE users, trades, conversation_history, system_positions,
                         closed_positions, portfolio_snapshots, trade_journal CASCADE
            """)
            cur.execute(
                "UPDATE portfolio_meta SET value = '0.0' WHERE key = 'high_water_mark'"
            )

    _truncate()
    yield
    _truncate()


# ---------------------------------------------------------------------------
# Alpaca mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def alpaca_mock(monkeypatch):
    """Shared in-memory Alpaca fake.

    Patches _trading_client in both executor and broker modules so any call
    to _trading_client() returns the mock without touching real Alpaca APIs.
    """
    mock = AlpacaMock()
    monkeypatch.setattr("integrations.alpaca.broker._trading_client", lambda: mock)
    monkeypatch.setattr("integrations.alpaca.account._trading_client", lambda: mock)
    monkeypatch.setattr("integrations.alpaca.entries._trading_client", lambda: mock)
    monkeypatch.setattr("integrations.alpaca.exits._trading_client", lambda: mock)
    monkeypatch.setattr("integrations.alpaca.fills._trading_client", lambda: mock)
    return mock


@pytest.fixture
def alpaca_immediate_fill(alpaca_mock):
    """Market orders fill instantly at $150.00."""
    alpaca_mock.set_fill_behavior("immediate", fill_price=150.0)
    return alpaca_mock


@pytest.fixture
def alpaca_pending_fill(alpaca_mock):
    """Orders are submitted but not yet filled (filled_avg_price=None)."""
    alpaca_mock.set_fill_behavior("pending")
    return alpaca_mock


@pytest.fixture
def alpaca_oco_locked(alpaca_mock):
    """close_position raises 40310000 on first attempt, succeeds on retry."""
    alpaca_mock.set_close_behavior("oco_locked", unlock_after=1)
    return alpaca_mock


@pytest.fixture
def alpaca_phantom_position(alpaca_mock):
    """Broker has an open AAPL position not tracked in our DB."""
    alpaca_mock.add_open_position("AAPL", qty=10, avg_entry_price=150.0)
    return alpaca_mock


@pytest.fixture
def tmp_models_dir(tmp_path, monkeypatch, pg_db):
    """Redirect ML model persistence to a temp directory and set up a fresh DB."""
    models = tmp_path / "models"
    models.mkdir()

    import core.persistence as persist
    monkeypatch.setattr(persist, "MODELS_DIR", models)

    return models
