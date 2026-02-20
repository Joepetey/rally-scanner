"""Shared test fixtures — synthetic OHLCV data."""

import numpy as np
import pandas as pd
import pytest


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
def tmp_models_dir(tmp_path, monkeypatch):
    """Redirect persistence/portfolio/positions to a temp directory."""
    models = tmp_path / "models"
    models.mkdir()

    import rally.bot.discord_db as discord_db
    import rally.core.persistence as persist
    import rally.trading.portfolio as portfolio
    import rally.trading.positions as positions

    monkeypatch.setattr(persist, "MODELS_DIR", models)
    monkeypatch.setattr(portfolio, "DATA_DIR", models)
    monkeypatch.setattr(portfolio, "EQUITY_LOG", models / "equity_history.csv")
    monkeypatch.setattr(portfolio, "TRADE_JOURNAL", models / "trade_journal.csv")

    # Point positions DB to temp directory
    db_path = models / "rally_test.db"
    monkeypatch.setattr(discord_db, "DB_PATH", db_path)

    # Reset DB initialization flag so tables get created fresh
    monkeypatch.setattr(positions, "_db_initialized", False)

    # Initialize DB tables
    discord_db.init_db(db_path)

    return models
