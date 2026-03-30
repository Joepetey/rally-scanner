"""Trading rules — entry signals and position sizing."""

import pandas as pd

from config import PARAMS


def generate_signals(preds: pd.DataFrame, require_trend: bool = False) -> pd.Series:
    """
    Entry signal: LONG at close if all conditions met.
    Returns boolean Series.
    """
    p = PARAMS
    signal = (
        (preds["P_RALLY"] > p.p_rally_threshold)
        & (preds["COMP_SCORE"] > p.comp_score_threshold)
        & (preds["FAIL_DN_SCORE"] > p.fail_dn_score_threshold)
    )
    if require_trend:
        signal = signal & (preds["Trend"] == 1.0)
    return signal


def compute_position_size(preds: pd.DataFrame) -> pd.Series:
    """
    Vol-targeted sizing: size = k * (P_RALLY - threshold) / ATR_pct
    Clamped to [min_position_size, max_risk]. Returns 0 for positions below minimum.
    """
    p = PARAMS
    atr = preds["ATR_pct"].replace(0, float("nan"))
    raw_size = p.vol_target_k * (preds["P_RALLY"] - p.p_rally_threshold) / atr
    clipped = raw_size.clip(lower=0.0, upper=p.max_risk_frac)
    return clipped.where(clipped >= p.min_position_size, 0.0)
