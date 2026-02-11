"""Tests for label computation."""

import numpy as np
import pandas as pd

from rally.config import PARAMS, AssetConfig
from rally.labels import compute_labels


def test_labels_shape(ohlcv_df):
    asset = AssetConfig(ticker="TEST", asset_class="equity", r_up=0.03, d_dn=0.015)
    labels = compute_labels(ohlcv_df, asset)
    assert len(labels) == len(ohlcv_df)


def test_labels_last_H_are_nan(ohlcv_df):
    asset = AssetConfig(ticker="TEST", asset_class="equity", r_up=0.03, d_dn=0.015)
    labels = compute_labels(ohlcv_df, asset)
    H = PARAMS.forward_horizon
    assert labels.iloc[-H:].isna().all()


def test_labels_binary(ohlcv_df):
    asset = AssetConfig(ticker="TEST", asset_class="equity", r_up=0.03, d_dn=0.015)
    labels = compute_labels(ohlcv_df, asset)
    valid = labels.dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_labels_known_rally():
    """Construct a price series where a rally is guaranteed."""
    n = 20
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = np.array([100.0] * n)
    # Bar 0: close=100. Bars 1-10: close jumps to 106 (6% gain)
    # Low stays at 100 (no drawdown), so d_dn=0.015 is safe
    close[1:11] = 106.0
    low = np.full(n, 100.0)
    high = close + 1.0

    df = pd.DataFrame({
        "Open": close,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": [1_000_000] * n,
    }, index=dates)

    asset = AssetConfig(ticker="TEST", asset_class="equity", r_up=0.05, d_dn=0.015)
    labels = compute_labels(df, asset)
    assert labels.iloc[0] == 1.0  # bar 0 should be labeled as rally


def test_labels_known_no_rally():
    """Construct a flat price series — no rally should be detected."""
    n = 20
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = np.full(n, 100.0)
    df = pd.DataFrame({
        "Open": close,
        "High": close + 0.01,
        "Low": close - 0.01,
        "Close": close,
        "Volume": [1_000_000] * n,
    }, index=dates)

    asset = AssetConfig(ticker="TEST", asset_class="equity", r_up=0.03, d_dn=0.015)
    labels = compute_labels(df, asset)
    valid = labels.dropna()
    assert (valid == 0.0).all()
