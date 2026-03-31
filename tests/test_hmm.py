"""Tests for HMM regime detection + training artifacts (MIC-94).

Uses real numpy/sklearn/hmmlearn with synthetic data — fast enough to run without mocking.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import PARAMS
from core.hmm import N_STATES, fit_hmm, predict_hmm_probs
from core.train import fit_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_df(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV + features DataFrame with 2 clear volatility regimes."""
    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)

    # Regime 1 (low vol): first half
    # Regime 2 (high vol): second half
    half = n // 2
    returns_low = np.random.normal(0.001, 0.005, half)
    returns_high = np.random.normal(0.001, 0.025, half)
    returns = np.concatenate([returns_low, returns_high])
    close = 100.0 * np.exp(np.cumsum(returns))

    open_ = np.roll(close, 1) * (1 + np.random.normal(0, 0.001, n))
    open_[0] = close[0] * 0.999
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.003, n)))
    volume = np.random.randint(1_000_000, 50_000_000, n)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)

    # Compute HMM features manually (minimal versions)
    log_ret = np.log(close / np.roll(close, 1))
    log_ret[0] = 0
    rv = pd.Series(log_ret).rolling(20, min_periods=20).std().values
    atr = ((high - low) / close)
    atr_pct = pd.Series(atr).rolling(20, min_periods=20).mean().values
    bb_ma = pd.Series(close).rolling(20, min_periods=20).mean()
    bb_std = pd.Series(close).rolling(20, min_periods=20).std()
    bb_width = (2 * bb_std / bb_ma).values

    df["RV"] = rv
    df["ATR_pct"] = atr_pct
    df["BB_width"] = bb_width

    return df


def _make_full_training_df(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Build features + synthetic labels for fit_model testing."""
    from core.features import build_features

    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    returns = np.random.normal(0.0005, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1) * (1 + np.random.normal(0, 0.002, n))
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.005, n)))

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": np.random.randint(1_000_000, 50_000_000, n),
    }, index=dates)

    # Add VIX
    vix = 15 + np.cumsum(np.random.normal(0, 0.5, n))
    df["VIX_Close"] = np.clip(vix, 9, 80)

    df = build_features(df, live=False)

    # Synthetic label
    H = PARAMS.forward_horizon
    label = np.full(n, np.nan)
    c = df["Close"].values
    for t in range(n - H):
        future_max = c[t + 1: t + H + 1].max()
        label[t] = 1.0 if future_max / c[t] - 1 >= 0.02 else 0.0
    df["RALLY_ST"] = label

    return df


# ---------------------------------------------------------------------------
# fit_hmm
# ---------------------------------------------------------------------------


class TestFitHmm:

    def test_returns_model_scaler_state_order(self):
        df = _make_training_df()
        model, scaler, state_order = fit_hmm(df)
        assert model is not None
        assert scaler is not None
        assert state_order is not None
        assert model.n_components == N_STATES
        assert len(state_order) == N_STATES

    def test_state_order_is_permutation(self):
        df = _make_training_df()
        _, _, state_order = fit_hmm(df)
        assert sorted(state_order) == list(range(N_STATES))

    def test_convergence_failure_on_single_row(self):
        """Single-row DataFrame should raise during HMM fitting."""
        df = pd.DataFrame({
            "RV": [0.01],
            "ATR_pct": [0.02],
            "BB_width": [0.03],
        }, index=pd.bdate_range("2020-01-02", periods=1))

        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            fit_hmm(df)

    def test_uses_available_features_only(self):
        """If only RV is present (no ATR_pct, BB_width), it should still work."""
        df = _make_training_df()
        df = df.drop(columns=["ATR_pct", "BB_width"])
        # Should not crash — uses whatever HMM_FEATURES are available
        model, scaler, state_order = fit_hmm(df)
        assert model is not None


# ---------------------------------------------------------------------------
# predict_hmm_probs
# ---------------------------------------------------------------------------


class TestPredictHmmProbs:

    def test_output_shape_matches_input(self):
        df = _make_training_df()
        model, scaler, state_order = fit_hmm(df)

        probs = predict_hmm_probs(model, scaler, state_order, df)
        assert len(probs) == len(df)

    def test_probabilities_sum_to_one(self):
        df = _make_training_df()
        model, scaler, state_order = fit_hmm(df)

        probs = predict_hmm_probs(model, scaler, state_order, df)
        prob_cols = ["P_compressed", "P_normal", "P_expanding"]
        valid = probs[prob_cols].dropna()
        row_sums = valid.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_output_columns(self):
        df = _make_training_df()
        model, scaler, state_order = fit_hmm(df)

        probs = predict_hmm_probs(model, scaler, state_order, df)
        expected = {"P_compressed", "P_normal", "P_expanding", "HMM_transition_signal"}
        assert expected == set(probs.columns)

    def test_graceful_degradation_with_none_model(self):
        """When model is None, returns zeros."""
        df = _make_training_df()
        probs = predict_hmm_probs(None, None, None, df)
        assert len(probs) == len(df)
        assert (probs == 0.0).all().all()


# ---------------------------------------------------------------------------
# fit_model — full training pipeline
# ---------------------------------------------------------------------------


class TestFitModel:

    def test_artifacts_contain_all_required_keys(self):
        from core.features import ALL_FEATURE_COLS

        df = _make_full_training_df()
        artifacts = fit_model(df, ALL_FEATURE_COLS, target_col="RALLY_ST")
        assert artifacts is not None

        required = {
            "lr_model", "lr_scaler", "iso_calibrator",
            "hmm_model", "hmm_scaler", "state_order",
            "feature_cols", "coefs", "intercept",
        }
        assert required.issubset(artifacts.keys())

    def test_artifact_keys_match_scanner_expectations(self):
        """Scanner accesses specific keys — verify they exist."""
        from core.features import ALL_FEATURE_COLS

        df = _make_full_training_df()
        artifacts = fit_model(df, ALL_FEATURE_COLS, target_col="RALLY_ST")
        assert artifacts is not None

        # Keys accessed by pipeline/scanner.py scan_single()
        scanner_keys = ["lr_model", "lr_scaler", "iso_calibrator", "feature_cols",
                        "hmm_model", "hmm_scaler", "state_order"]
        for key in scanner_keys:
            assert key in artifacts, f"Scanner expects '{key}' but missing from artifacts"

    def test_returns_none_for_insufficient_data(self):
        tiny = pd.DataFrame({
            "COMP_SCORE": np.random.rand(10),
            "RALLY_ST": np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float),
        })
        result = fit_model(tiny, ["COMP_SCORE"], target_col="RALLY_ST")
        assert result is None

    def test_returns_none_for_too_few_positives(self):
        n = 200
        df = pd.DataFrame({
            "COMP_SCORE": np.random.rand(n),
            "RALLY_ST": np.zeros(n),  # zero positives
        })
        result = fit_model(df, ["COMP_SCORE"], target_col="RALLY_ST")
        assert result is None

    def test_hmm_failure_produces_none_hmm_artifacts(self):
        """When HMM fails (no HMM features), artifacts still produced with hmm_model=None."""
        n = 300
        np.random.seed(77)
        df = pd.DataFrame({
            "feat1": np.random.rand(n),
            "feat2": np.random.rand(n),
            "RALLY_ST": np.random.choice([0.0, 1.0], n, p=[0.7, 0.3]),
        })
        # No HMM features (RV, ATR_pct, BB_width) → fit_hmm will fail
        artifacts = fit_model(df, ["feat1", "feat2"], target_col="RALLY_ST")
        assert artifacts is not None
        assert artifacts["hmm_model"] is None
        assert artifacts["hmm_scaler"] is None
        assert artifacts["state_order"] is None
        # LR model should still be trained
        assert artifacts["lr_model"] is not None
        assert artifacts["iso_calibrator"] is not None


# ---------------------------------------------------------------------------
# Model artifact round-trip (save/load)
# ---------------------------------------------------------------------------


class TestModelRoundTrip:

    def test_save_load_produces_same_predictions(self):
        """Train, save, load → predictions must match."""
        import joblib

        from core.features import ALL_FEATURE_COLS

        df = _make_full_training_df()
        artifacts = fit_model(df, ALL_FEATURE_COLS, target_col="RALLY_ST")
        assert artifacts is not None

        # Add HMM columns to df (fit_model does this internally during training)
        hmm_probs = predict_hmm_probs(
            artifacts.get("hmm_model"), artifacts.get("hmm_scaler"),
            artifacts.get("state_order"), df,
        )
        df = df.join(hmm_probs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "TEST.joblib"
            joblib.dump(artifacts, path)
            loaded = joblib.load(path)

        # Verify predictions match
        feat_cols = artifacts["feature_cols"]
        valid = df[feat_cols].dropna()
        X = artifacts["lr_scaler"].transform(valid.values)

        orig_probs = artifacts["lr_model"].predict_proba(X)[:, 1]
        loaded_probs = loaded["lr_model"].predict_proba(X)[:, 1]
        np.testing.assert_array_equal(orig_probs, loaded_probs)

        orig_cal = artifacts["iso_calibrator"].predict(orig_probs)
        loaded_cal = loaded["iso_calibrator"].predict(loaded_probs)
        np.testing.assert_array_equal(orig_cal, loaded_cal)
