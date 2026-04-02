"""End-to-end ML pipeline tests — features, training, signals, integration."""

import numpy as np
import pandas as pd
from rally_ml.config import PARAMS
from rally_ml.core.features import (
    FEATURE_COLS,
    _rsi,
    build_features,
    compute_compression,
    compute_range,
)
from rally_ml.core.train import fit_model

from trading.signals import compute_position_size, generate_signals

# Warmup: percentile_window (252) is the bottleneck; use 253 as safe cutoff
WARMUP = PARAMS.percentile_window + 1


# ───────────────────────────────────────────────────────────────────────
# Group 1: Feature correctness
# ───────────────────────────────────────────────────────────────────────


class TestFeatureCorrectness:

    def test_rsi_bounded_0_to_100(self, ohlcv_df):
        rsi = _rsi(ohlcv_df["Close"], PARAMS.rsi_period)
        valid = rsi.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0
        assert not np.any(np.isinf(valid.values))

    def test_atr_pct_always_positive(self, ohlcv_df):
        df = compute_compression(ohlcv_df)
        valid = df["ATR_pct"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_compression_score_bounded(self, ohlcv_df):
        df = build_features(ohlcv_df, live=True)
        valid = df["COMP_SCORE"].iloc[WARMUP:].dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0 - 1e-9
        assert valid.max() <= 1.0 + 1e-9

    def test_range_tightness_finite_when_atr_nonzero(self, ohlcv_df):
        df = compute_compression(ohlcv_df)
        df = compute_range(df)
        mask = df["ATR_pct"] > 0
        tightness = df.loc[mask, "RangeTightness"].dropna()
        assert np.all(np.isfinite(tightness.values))

    def test_features_have_no_lookahead_bias(self, ohlcv_df):
        """Features computed on first 100 bars must equal the first 100 rows
        of features computed on 101 bars (live mode uses lagged trap score)."""
        n = 100
        df_short = ohlcv_df.iloc[:n].copy()
        df_long = ohlcv_df.iloc[: n + 1].copy()

        feat_short = build_features(df_short, live=True)
        feat_long = build_features(df_long, live=True)

        # Compare shared columns on the first n rows
        cols = [c for c in FEATURE_COLS if c in feat_short.columns and c in feat_long.columns]
        short_vals = feat_short[cols].values
        long_vals = feat_long[cols].iloc[:n].values

        # Where both are non-NaN, they must match
        both_valid = ~np.isnan(short_vals) & ~np.isnan(long_vals)
        np.testing.assert_allclose(
            short_vals[both_valid], long_vals[both_valid], rtol=1e-10,
        )

    def test_trend_feature_is_binary(self, ohlcv_df):
        df = build_features(ohlcv_df, live=True)
        valid = df["Trend"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})


# ───────────────────────────────────────────────────────────────────────
# Group 2: Model training
# ───────────────────────────────────────────────────────────────────────


def _prepare_training_df(ohlcv_with_vix: pd.DataFrame) -> pd.DataFrame:
    """Build features and attach a synthetic RALLY_ST label."""
    df = build_features(ohlcv_with_vix, live=False)

    # Synthetic label: price rose > 2% within next 10 bars
    close = df["Close"].values
    n = len(df)
    label = np.zeros(n)
    H = PARAMS.forward_horizon
    for t in range(n - H):
        future_max = close[t + 1 : t + H + 1].max()
        if future_max / close[t] - 1 >= 0.02:
            label[t] = 1.0
        else:
            label[t] = 0.0
    label[n - H :] = np.nan
    df["RALLY_ST"] = label
    return df


class TestModelTraining:

    def test_trained_model_produces_probabilities_in_unit_interval(self, ohlcv_with_vix):
        from rally_ml.core.features import ALL_FEATURE_COLS
        from rally_ml.core.hmm import predict_hmm_probs

        df = _prepare_training_df(ohlcv_with_vix)
        artifacts = fit_model(df, ALL_FEATURE_COLS, target_col="RALLY_ST")
        assert artifacts is not None, "fit_model returned None on 500-bar data"

        lr = artifacts["lr_model"]
        scaler = artifacts["lr_scaler"]
        iso = artifacts["iso_calibrator"]
        feat_cols = artifacts["feature_cols"]

        # Add HMM columns to df (fit_model does this internally during training)
        hmm_probs = predict_hmm_probs(
            artifacts.get("hmm_model"),
            artifacts.get("hmm_scaler"),
            artifacts.get("state_order"),
            df,
        )
        df = df.join(hmm_probs)

        valid = df[feat_cols + ["RALLY_ST"]].dropna()
        X = scaler.transform(valid[feat_cols].values)
        raw_probs = lr.predict_proba(X)[:, 1]
        cal_probs = iso.predict(raw_probs)

        assert np.all(cal_probs >= 0.0)
        assert np.all(cal_probs <= 1.0)

    def test_model_artifacts_contain_required_keys(self, ohlcv_with_vix):
        from rally_ml.core.features import ALL_FEATURE_COLS

        df = _prepare_training_df(ohlcv_with_vix)
        artifacts = fit_model(df, ALL_FEATURE_COLS, target_col="RALLY_ST")
        assert artifacts is not None

        for key in ("lr_model", "lr_scaler", "iso_calibrator", "feature_cols"):
            assert key in artifacts, f"Missing key: {key}"

    def test_fit_model_returns_none_insufficient_data(self):
        """A tiny DataFrame (10 rows) should cause fit_model to return None."""
        np.random.seed(7)
        tiny = pd.DataFrame({
            "COMP_SCORE": np.random.rand(10),
            "RALLY_ST": np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float),
        })
        result = fit_model(tiny, ["COMP_SCORE"], target_col="RALLY_ST")
        assert result is None


# ───────────────────────────────────────────────────────────────────────
# Group 3: Signal generation
# ───────────────────────────────────────────────────────────────────────


def _make_signal_row(**overrides) -> pd.DataFrame:
    """Single-row DataFrame with all columns needed by generate_signals / compute_position_size."""
    defaults = {
        "P_RALLY": 0.70,
        "COMP_SCORE": 0.75,
        "FAIL_DN_SCORE": 1.5,
        "Trend": 1.0,
        "ATR_pct": 0.02,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestSignalGeneration:

    def test_signal_fires_when_all_conditions_met(self):
        row = _make_signal_row()
        sig = generate_signals(row)
        assert sig.iloc[0] is True or sig.iloc[0] == True  # noqa: E712

    def test_signal_silent_when_p_rally_below_threshold(self):
        row = _make_signal_row(P_RALLY=PARAMS.p_rally_threshold - 0.01)
        sig = generate_signals(row)
        assert not sig.iloc[0]

    def test_signal_silent_when_comp_score_below_threshold(self):
        row = _make_signal_row(COMP_SCORE=PARAMS.comp_score_threshold - 0.01)
        sig = generate_signals(row)
        assert not sig.iloc[0]

    def test_signal_silent_when_atr_pct_is_zero(self):
        row = _make_signal_row(ATR_pct=0.0)
        size = compute_position_size(row)
        assert size.iloc[0] == 0.0

    def test_signal_requires_trend_when_flag_set(self):
        row = _make_signal_row(Trend=0.0)
        sig = generate_signals(row, require_trend=True)
        assert not sig.iloc[0]


# ───────────────────────────────────────────────────────────────────────
# Group 4: Pipeline integration
# ───────────────────────────────────────────────────────────────────────


class TestPipelineIntegration:

    def test_pipeline_handles_insufficient_data(self):
        """50 bars is far below the 300-bar minimum in scan_single."""
        from pipeline.scanner import scan_single

        np.random.seed(11)
        n = 50
        dates = pd.bdate_range("2023-01-02", periods=n)
        close = 100 + np.cumsum(np.random.normal(0, 0.5, n))
        df_small = pd.DataFrame({
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        }, index=dates)

        # Minimal artifacts dict — scan_single checks len(df) before using them
        from rally_ml.config import AssetConfig

        artifacts = {
            "asset_config": AssetConfig(
                ticker="TEST", asset_class="equity", r_up=0.03, d_dn=0.02,
            ).model_dump(),
            "lr_model": None,
            "lr_scaler": None,
            "iso_calibrator": None,
            "feature_cols": FEATURE_COLS,
        }

        result = scan_single("TEST", artifacts, ohlcv_data=df_small)
        assert result["status"] == "insufficient_data"

    def test_pipeline_output_schema_is_stable(self, ohlcv_with_vix):
        """Train a model, run scan_single, verify output keys."""
        from rally_ml.config import AssetConfig
        from rally_ml.core.features import ALL_FEATURE_COLS

        from pipeline.scanner import scan_single

        df = _prepare_training_df(ohlcv_with_vix)
        artifacts = fit_model(df, ALL_FEATURE_COLS, target_col="RALLY_ST")
        assert artifacts is not None, "fit_model returned None — cannot test scan"

        artifacts["asset_config"] = AssetConfig(
            ticker="TEST", asset_class="equity", r_up=0.03, d_dn=0.02,
        ).model_dump()

        result = scan_single("TEST", artifacts, ohlcv_data=ohlcv_with_vix)

        expected_keys = {
            "ticker", "status", "close", "p_rally", "comp_score",
            "signal", "size", "atr_pct",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )
        assert result["status"] == "ok"
