"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd

from rally.features import (
    build_features,
    compute_compression,
    compute_range,
    compute_trend,
    compute_failed_breakdown,
    FEATURE_COLS,
)


def test_compute_compression_columns(ohlcv_df):
    result = compute_compression(ohlcv_df)
    for col in ["ATR", "ATR_pct", "BB_width", "RV", "p_ATR", "p_BB", "p_RV", "COMP_SCORE"]:
        assert col in result.columns, f"Missing column: {col}"


def test_compute_compression_score_range(ohlcv_df):
    result = compute_compression(ohlcv_df)
    valid = result["COMP_SCORE"].dropna()
    assert len(valid) > 0
    assert valid.min() >= -0.01  # allow tiny float imprecision
    assert valid.max() <= 1.01


def test_compute_range_columns(ohlcv_df):
    result = compute_compression(ohlcv_df)
    result = compute_range(result)
    for col in ["RangeHigh", "RangeLow", "RangeWidth", "RangeTightness"]:
        assert col in result.columns


def test_compute_trend_binary(ohlcv_df):
    result = compute_compression(ohlcv_df)
    result = compute_range(result)
    result = compute_failed_breakdown(result)
    result = compute_trend(result)
    valid = result["Trend"].dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_build_features_all_columns(ohlcv_with_vix):
    result = build_features(ohlcv_with_vix)
    for col in FEATURE_COLS:
        assert col in result.columns, f"Missing feature: {col}"


def test_build_features_no_inf(ohlcv_with_vix):
    result = build_features(ohlcv_with_vix)
    for col in FEATURE_COLS:
        valid = result[col].dropna()
        assert not np.isinf(valid).any(), f"Inf values in {col}"


def test_build_features_live_mode(ohlcv_with_vix):
    result = build_features(ohlcv_with_vix, live=True)
    assert "FAIL_DN_SCORE" in result.columns
    # Live mode lags FAIL_DN_SCORE — first trap_lookforward bars should be NaN
    # (or zero, depending on whether there are traps in that range)
    assert len(result) == len(ohlcv_with_vix)


def test_build_features_preserves_index(ohlcv_with_vix):
    result = build_features(ohlcv_with_vix)
    assert result.index.equals(ohlcv_with_vix.index)
