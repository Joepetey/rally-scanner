"""Tests for probability calibration — core/calibrate.py (MIC-98).

Verifies auto-calibrated thresholds are bounded, asset class lookup works,
and output types match scanner expectations.
"""

import numpy as np
import pandas as pd

from config import AssetConfig
from core.calibrate import (
    D_DN_MAX,
    D_DN_MIN,
    R_UP_MAX,
    R_UP_MIN,
    calibrate_thresholds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, daily_vol: float = 0.015, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    returns = np.random.normal(0.0005, daily_vol, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


# ---------------------------------------------------------------------------
# calibrate_thresholds
# ---------------------------------------------------------------------------


class TestCalibrateThresholds:

    def test_returns_asset_config(self):
        df = _make_ohlcv()
        result = calibrate_thresholds(df, "AAPL")
        assert isinstance(result, AssetConfig)
        assert result.ticker == "AAPL"

    def test_r_up_within_bounds(self):
        df = _make_ohlcv()
        result = calibrate_thresholds(df, "AAPL")
        assert R_UP_MIN <= result.r_up <= R_UP_MAX

    def test_d_dn_within_bounds(self):
        df = _make_ohlcv()
        result = calibrate_thresholds(df, "AAPL")
        assert D_DN_MIN <= result.d_dn <= D_DN_MAX

    def test_d_dn_is_half_r_up(self):
        df = _make_ohlcv()
        result = calibrate_thresholds(df, "AAPL")
        expected_d_dn = np.clip(result.r_up / 2.0, D_DN_MIN, D_DN_MAX)
        assert result.d_dn == round(expected_d_dn, 4)

    def test_high_vol_produces_higher_r_up(self):
        """Higher daily volatility should produce larger r_up."""
        df_low = _make_ohlcv(daily_vol=0.005, seed=1)
        df_high = _make_ohlcv(daily_vol=0.04, seed=2)
        r_low = calibrate_thresholds(df_low, "X").r_up
        r_high = calibrate_thresholds(df_high, "X").r_up
        assert r_high >= r_low

    def test_known_ticker_preserves_asset_class(self):
        """Known ticker (e.g., AAPL) should get asset_class from config."""
        df = _make_ohlcv()
        result = calibrate_thresholds(df, "AAPL")
        assert result.asset_class == "equity"

    def test_crypto_ticker_preserves_asset_class(self):
        """Known crypto (BTC-USD) should get asset_class='crypto'."""
        df = _make_ohlcv()
        result = calibrate_thresholds(df, "BTC-USD")
        assert result.asset_class == "crypto"

    def test_unknown_ticker_defaults_to_equity(self):
        df = _make_ohlcv()
        result = calibrate_thresholds(df, "UNKNOWN_TICKER_XYZ")
        assert result.asset_class == "equity"

    def test_output_type_is_float(self):
        """Scanner expects float values, not numpy scalars."""
        df = _make_ohlcv()
        result = calibrate_thresholds(df, "AAPL")
        assert isinstance(result.r_up, float)
        assert isinstance(result.d_dn, float)

    def test_values_are_rounded_to_4_decimals(self):
        df = _make_ohlcv()
        result = calibrate_thresholds(df, "AAPL")
        assert result.r_up == round(result.r_up, 4)
        assert result.d_dn == round(result.d_dn, 4)
