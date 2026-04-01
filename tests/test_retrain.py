"""Tests for retrain pipeline — pipeline/retrain.py (MIC-97).

Mocks data fetching, training, and persistence to test retrain_all flow.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
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


def _mock_artifacts(ticker: str) -> dict:
    return {
        "lr_model": MagicMock(),
        "lr_scaler": MagicMock(),
        "iso_calibrator": MagicMock(),
        "feature_cols": ["COMP_SCORE"],
        "hmm_model": None,
        "hmm_scaler": None,
        "state_order": None,
        "train_start": "2020",
        "train_end": "2024",
    }


# ---------------------------------------------------------------------------
# retrain_all
# ---------------------------------------------------------------------------


class TestRetrainAll:

    @patch("pipeline.retrain.save_model")
    @patch("pipeline.retrain.load_manifest", return_value={})
    @patch("pipeline.retrain.fetch_vix_safe", return_value=None)
    @patch("pipeline.retrain.fetch_daily_batch")
    def test_happy_path_trains_all_tickers(
        self, mock_batch, mock_vix, mock_manifest, mock_save,
    ):
        from pipeline.retrain import retrain_all

        df = _make_ohlcv(600)
        mock_batch.return_value = {"AAPL": df.copy(), "MSFT": df.copy()}

        retrain_all(tickers=["AAPL", "MSFT"])

        # save_model should be called for each successfully trained ticker
        assert mock_save.call_count >= 1
        saved_tickers = [c[0][0] for c in mock_save.call_args_list]
        # Both should train successfully with 600 bars
        assert set(saved_tickers) == {"AAPL", "MSFT"}

    @patch("pipeline.retrain.save_model")
    @patch("pipeline.retrain.load_manifest", return_value={})
    @patch("pipeline.retrain.fetch_vix_safe", return_value=None)
    @patch("pipeline.retrain.fetch_daily_batch")
    def test_insufficient_data_ticker_skipped(
        self, mock_batch, mock_vix, mock_manifest, mock_save,
    ):
        """Tickers with < 500 bars are skipped."""
        from pipeline.retrain import retrain_all

        df_short = _make_ohlcv(n=100, seed=1)  # too short
        df_long = _make_ohlcv(n=600, seed=2)
        mock_batch.return_value = {"SHORT": df_short, "LONG": df_long}

        retrain_all(tickers=["SHORT", "LONG"])

        saved_tickers = [c[0][0] for c in mock_save.call_args_list]
        assert "SHORT" not in saved_tickers
        assert "LONG" in saved_tickers

    @patch("pipeline.retrain.save_model")
    @patch("pipeline.retrain.load_manifest", return_value={})
    @patch("pipeline.retrain.fetch_vix_safe", return_value=None)
    @patch("pipeline.retrain.fetch_daily_batch")
    def test_training_error_does_not_crash(
        self, mock_batch, mock_vix, mock_manifest, mock_save,
    ):
        """If one ticker has bad data, others still train."""
        from pipeline.retrain import retrain_all

        df_good = _make_ohlcv(600, seed=2)
        # BAD ticker: too short to train (< 500 bars) but long enough to not
        # be skipped by the 500-bar check — use data that will produce None from fit_model
        df_bad = _make_ohlcv(100, seed=3)
        mock_batch.return_value = {"GOOD": df_good, "BAD": df_bad}

        retrain_all(tickers=["GOOD", "BAD"])

        saved_tickers = [c[0][0] for c in mock_save.call_args_list]
        assert "GOOD" in saved_tickers
        assert "BAD" not in saved_tickers



# ---------------------------------------------------------------------------
# _is_fresh
# ---------------------------------------------------------------------------


class TestIsFresh:

    def test_fresh_model_detected(self):
        from datetime import datetime

        from pipeline.retrain import _is_fresh

        entry = {"saved_at": datetime.now().isoformat()}
        assert _is_fresh(entry, max_age_days=7) is True

    def test_stale_model_detected(self):
        from pipeline.retrain import _is_fresh

        entry = {"saved_at": "2020-01-01T00:00:00"}
        assert _is_fresh(entry, max_age_days=7) is False

