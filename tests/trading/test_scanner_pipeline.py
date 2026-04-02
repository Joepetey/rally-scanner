"""Tests for scanner pipeline — pipeline/scanner.py (MIC-92).

Mocks data fetching, model loading, and features to test scan logic in isolation.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from rally_ml.config import CONFIGS_BY_NAME, PARAMS, AssetConfig
from rally_ml.core.features import ALL_FEATURE_COLS, build_features
from rally_ml.core.train import fit_model

from pipeline.scanner import resolve_config, scan_single

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    returns = np.random.normal(0.0005, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1) * (1 + np.random.normal(0, 0.002, n))
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": np.random.randint(1_000_000, 50_000_000, n),
    }, index=dates)


def _make_artifacts(ohlcv_df: pd.DataFrame) -> dict | None:
    """Train real artifacts from synthetic data."""
    np.random.seed(42)
    vix = 15 + np.cumsum(np.random.normal(0, 0.5, len(ohlcv_df)))
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df["VIX_Close"] = np.clip(vix, 9, 80)

    df = build_features(ohlcv_df, live=False)
    H = PARAMS.forward_horizon
    n = len(df)
    c = df["Close"].values
    label = np.full(n, np.nan)
    for t in range(n - H):
        future_max = c[t + 1: t + H + 1].max()
        label[t] = 1.0 if future_max / c[t] - 1 >= 0.02 else 0.0
    df["RALLY_ST"] = label

    artifacts = fit_model(df, ALL_FEATURE_COLS, target_col="RALLY_ST")
    if artifacts:
        artifacts["asset_config"] = AssetConfig(
            ticker="TEST", asset_class="equity", r_up=0.03, d_dn=0.015,
        ).model_dump()
    return artifacts


# ---------------------------------------------------------------------------
# scan_single — happy path
# ---------------------------------------------------------------------------


class TestScanSingle:

    def test_happy_path_returns_expected_keys(self):
        ohlcv = _make_ohlcv()
        artifacts = _make_artifacts(ohlcv.copy())
        assert artifacts is not None, "Could not train artifacts for test"

        np.random.seed(99)
        vix = pd.Series(
            np.clip(15 + np.cumsum(np.random.normal(0, 0.5, len(ohlcv))), 9, 80),
            index=ohlcv.index, name="VIX_Close",
        )

        result = scan_single("TEST", artifacts, vix_data=vix, ohlcv_data=ohlcv)
        assert result["status"] == "ok"
        expected_keys = {
            "ticker", "status", "close", "p_rally", "signal",
            "comp_score", "size", "atr_pct",
        }
        assert expected_keys.issubset(result.keys())

    def test_p_rally_in_unit_interval(self):
        ohlcv = _make_ohlcv()
        artifacts = _make_artifacts(ohlcv.copy())
        assert artifacts is not None

        result = scan_single("TEST", artifacts, ohlcv_data=ohlcv)
        assert result["status"] == "ok"
        assert 0.0 <= result["p_rally"] <= 1.0

    def test_insufficient_data_returns_status(self):
        """DataFrame with < 300 bars triggers insufficient_data."""
        small_df = _make_ohlcv(n=50, seed=11)
        artifacts = {
            "asset_config": AssetConfig(
                ticker="TEST", asset_class="equity", r_up=0.03, d_dn=0.015,
            ).model_dump(),
        }
        result = scan_single("TEST", artifacts, ohlcv_data=small_df)
        assert result["status"] == "insufficient_data"

    def test_signal_field_is_bool(self):
        ohlcv = _make_ohlcv()
        artifacts = _make_artifacts(ohlcv.copy())
        assert artifacts is not None

        result = scan_single("TEST", artifacts, ohlcv_data=ohlcv)
        assert isinstance(result["signal"], bool)


# ---------------------------------------------------------------------------
# resolve_config
# ---------------------------------------------------------------------------


class TestResolveConfig:

    def test_conservative_config(self):
        cfg = resolve_config("conservative")
        assert cfg is CONFIGS_BY_NAME["conservative"]
        assert cfg.p_rally == CONFIGS_BY_NAME["conservative"].p_rally

    def test_baseline_config(self):
        cfg = resolve_config("baseline")
        assert cfg is CONFIGS_BY_NAME["baseline"]

    def test_unknown_config_raises(self):
        with pytest.raises(SystemExit):
            resolve_config("nonexistent_config")

    def test_all_configs_are_valid(self):
        """Every named config resolves without error."""
        for name in CONFIGS_BY_NAME:
            cfg = resolve_config(name)
            assert cfg.p_rally > 0


# ---------------------------------------------------------------------------
# scan_all — integration (mocked external deps)
# ---------------------------------------------------------------------------


class TestScanAll:

    @staticmethod
    def _mock_pool():
        """Return a patched ProcessPoolExecutor that runs synchronously."""
        from concurrent.futures import Future
        from unittest.mock import MagicMock

        def mock_submit(fn, item):
            f = Future()
            f.set_result(fn(item))
            return f

        mock_pool = MagicMock()
        mock_pool.__enter__ = lambda s: s
        mock_pool.__exit__ = lambda s, *a: None
        mock_pool.submit = mock_submit
        return patch("pipeline.scanner.ProcessPoolExecutor", return_value=mock_pool)

    @patch("pipeline.scanner.fetch_quotes")
    @patch("pipeline.scanner.fetch_daily_batch")
    @patch("pipeline.scanner.fetch_vix_safe")
    @patch("pipeline.scanner.load_manifest")
    @patch("pipeline.scanner.load_model")
    def test_scan_all_returns_results(
        self, mock_load_model, mock_manifest, mock_vix, mock_batch, mock_quotes,
    ):
        from pipeline.scanner import scan_all

        ohlcv = _make_ohlcv()
        artifacts = _make_artifacts(ohlcv.copy())
        if artifacts is None:
            pytest.skip("Could not train artifacts")

        mock_manifest.return_value = {"TEST": {"saved_at": "2024-01-01"}}
        mock_load_model.return_value = artifacts
        mock_vix.return_value = None
        mock_batch.return_value = {"TEST": ohlcv}
        mock_quotes.return_value = {}

        original = {k: getattr(PARAMS, k) for k in [
            "p_rally_threshold", "comp_score_threshold", "vol_target_k",
            "max_risk_frac", "profit_atr_mult", "time_stop_bars",
        ]}
        try:
            with self._mock_pool():
                results = scan_all(tickers=["TEST"], config_name="conservative")
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["ticker"] == "TEST"
        finally:
            for k, v in original.items():
                setattr(PARAMS, k, v)

    @patch("pipeline.scanner.load_manifest")
    def test_scan_all_empty_manifest(self, mock_manifest):
        from pipeline.scanner import scan_all

        mock_manifest.return_value = {}
        original = {k: getattr(PARAMS, k) for k in [
            "p_rally_threshold", "comp_score_threshold", "vol_target_k",
            "max_risk_frac", "profit_atr_mult", "time_stop_bars",
        ]}
        try:
            results = scan_all(config_name="conservative")
            assert results == []
        finally:
            for k, v in original.items():
                setattr(PARAMS, k, v)

    @patch("pipeline.scanner.fetch_quotes")
    @patch("pipeline.scanner.fetch_daily_batch")
    @patch("pipeline.scanner.fetch_vix_safe")
    @patch("pipeline.scanner.load_manifest")
    @patch("pipeline.scanner.load_model")
    def test_scan_all_batch_fetch_failure_falls_back(
        self, mock_load_model, mock_manifest, mock_vix, mock_batch, mock_quotes,
    ):
        """When batch fetch raises, scan_all sets ohlcv_cache={} and continues."""
        from pipeline.scanner import scan_all

        ohlcv = _make_ohlcv()
        artifacts = _make_artifacts(ohlcv.copy())
        if artifacts is None:
            pytest.skip("Could not train artifacts")

        mock_manifest.return_value = {"TEST": {"saved_at": "2024-01-01"}}
        mock_load_model.return_value = artifacts
        mock_vix.return_value = None
        mock_batch.side_effect = Exception("batch failed")
        mock_quotes.return_value = {}

        original = {k: getattr(PARAMS, k) for k in [
            "p_rally_threshold", "comp_score_threshold", "vol_target_k",
            "max_risk_frac", "profit_atr_mult", "time_stop_bars",
        ]}
        try:
            with self._mock_pool():
                results = scan_all(tickers=["TEST"], config_name="conservative")
            assert isinstance(results, list)
        finally:
            for k, v in original.items():
                setattr(PARAMS, k, v)
