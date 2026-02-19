"""Tests for trading rules — signal generation and position sizing."""

import pandas as pd

from rally.trading.signals import compute_position_size, generate_signals


def _make_preds(**overrides):
    """Build a 1-row prediction DataFrame with sensible defaults."""
    defaults = {
        "P_RALLY": 0.65,
        "COMP_SCORE": 0.70,
        "FAIL_DN_SCORE": 0.5,
        "Trend": 1.0,
        "ATR_pct": 0.02,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def test_signal_fires_when_above_thresholds():
    preds = _make_preds(P_RALLY=0.65, COMP_SCORE=0.70, FAIL_DN_SCORE=0.5)
    signals = generate_signals(preds)
    assert signals.iloc[0]


def test_signal_blocked_by_low_p_rally():
    preds = _make_preds(P_RALLY=0.30)
    signals = generate_signals(preds)
    assert not signals.iloc[0]


def test_signal_blocked_by_low_comp_score():
    preds = _make_preds(COMP_SCORE=0.20)
    signals = generate_signals(preds)
    assert not signals.iloc[0]


def test_signal_trend_filter():
    preds = _make_preds(Trend=0.0)
    # Without require_trend: should still fire
    assert generate_signals(preds, require_trend=False).iloc[0]
    # With require_trend: should block
    assert not generate_signals(preds, require_trend=True).iloc[0]


def test_position_size_positive():
    preds = _make_preds(P_RALLY=0.70, ATR_pct=0.02)
    size = compute_position_size(preds)
    assert size.iloc[0] > 0


def test_position_size_capped():
    preds = _make_preds(P_RALLY=0.99, ATR_pct=0.001)  # extreme values
    size = compute_position_size(preds)
    from rally.config import PARAMS
    assert size.iloc[0] <= PARAMS.max_risk_frac


def test_position_size_zero_below_threshold():
    preds = _make_preds(P_RALLY=0.30)  # below threshold
    size = compute_position_size(preds)
    assert size.iloc[0] == 0.0
