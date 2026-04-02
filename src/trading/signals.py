"""Trading rules — entry signals and position sizing."""

import pandas as pd
from rally_ml.config import PARAMS
from rally_ml.config.trading import TradingConfig


def generate_signals(
    preds: pd.DataFrame,
    require_trend: bool = False,
    config: TradingConfig | None = None,
) -> pd.Series:
    """
    Entry signal: LONG at close if all conditions met.
    Returns boolean Series.
    """
    if config is not None:
        p_rally_thresh = config.p_rally
        comp_thresh = config.comp_score
    else:
        p_rally_thresh = PARAMS.p_rally_threshold
        comp_thresh = PARAMS.comp_score_threshold

    signal = (
        (preds["P_RALLY"] > p_rally_thresh)
        & (preds["COMP_SCORE"] > comp_thresh)
        & (preds["FAIL_DN_SCORE"] > PARAMS.fail_dn_score_threshold)
    )
    if require_trend:
        signal = signal & (preds["Trend"] == 1.0)
    return signal


def compute_position_size(
    preds: pd.DataFrame,
    config: TradingConfig | None = None,
) -> pd.Series:
    """
    Vol-targeted sizing: size = k * (P_RALLY - threshold) / ATR_pct
    Clamped to [min_position_size, max_risk]. Returns 0 for positions below minimum.
    """
    if config is not None:
        vol_k = config.vol_k
        p_rally_thresh = config.p_rally
        max_risk = config.max_risk
    else:
        vol_k = PARAMS.vol_target_k
        p_rally_thresh = PARAMS.p_rally_threshold
        max_risk = PARAMS.max_risk_frac

    atr = preds["ATR_pct"].replace(0, float("nan"))
    raw_size = vol_k * (preds["P_RALLY"] - p_rally_thresh) / atr
    clipped = raw_size.clip(lower=0.0, upper=max_risk)
    return clipped.where(clipped >= PARAMS.min_position_size, 0.0)
