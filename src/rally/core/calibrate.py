"""
Auto-calibration of r_up / d_dn thresholds based on historical volatility.

Logic:
  1. Compute 20-day realized volatility over trailing 252 days
  2. Take median daily RV
  3. Scale to 10-bar horizon: rv_10 = median_daily_rv * sqrt(10)
  4. r_up = rv_10 * 0.80 (about 80% of a 1-sigma move)
  5. d_dn = r_up / 2.0 (maintain 2:1 ratio from hand-tuned assets)
  6. Clamp to sensible ranges
"""

import numpy as np
import pandas as pd

from ..config import PARAMS, AssetConfig

R_UP_MULT = 0.80
R_UP_MIN, R_UP_MAX = 0.015, 0.10
D_DN_MIN, D_DN_MAX = 0.008, 0.05


def calibrate_thresholds(df: pd.DataFrame, ticker: str) -> AssetConfig:
    """Given OHLCV DataFrame, compute auto-calibrated AssetConfig."""
    close = df["Close"]
    log_ret = np.log(close / close.shift(1))

    # 20-day rolling realized vol (daily)
    rv_daily = log_ret.rolling(20, min_periods=20).std()

    # Median over last 252 trading days
    recent_rv = rv_daily.iloc[-252:]
    median_rv = float(recent_rv.median())

    # Scale to forward horizon
    rv_h = median_rv * np.sqrt(PARAMS.forward_horizon)

    # Set thresholds
    r_up = float(np.clip(rv_h * R_UP_MULT, R_UP_MIN, R_UP_MAX))
    d_dn = float(np.clip(r_up / 2.0, D_DN_MIN, D_DN_MAX))

    return AssetConfig(
        ticker=ticker,
        asset_class="equity",
        r_up=round(r_up, 4),
        d_dn=round(d_dn, 4),
    )
