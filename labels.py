"""
Target variable construction — RALLY_ST binary label.
"""

import numpy as np
import pandas as pd

from config import AssetConfig, PARAMS


def compute_labels(df: pd.DataFrame, asset: AssetConfig) -> pd.Series:
    """
    RALLY_ST(t) = 1 if:
        max(Close[t+1 : t+H]) / Close[t] - 1  >= r_up
    AND
        min(Low[t+1 : t+H])   / Close[t] - 1  >= -d_dn
    Else 0

    Returns a Series aligned to df.index (NaN where label cannot be computed).
    """
    H = PARAMS.forward_horizon
    r_up = asset.r_up
    d_dn = asset.d_dn
    close = df["Close"].values
    low = df["Low"].values
    n = len(df)

    label = np.full(n, np.nan)

    for t in range(n - H):
        future_close = close[t + 1: t + H + 1]
        future_low = low[t + 1: t + H + 1]

        max_return = future_close.max() / close[t] - 1
        max_drawdown = future_low.min() / close[t] - 1

        if max_return >= r_up and max_drawdown >= -d_dn:
            label[t] = 1.0
        else:
            label[t] = 0.0

    return pd.Series(label, index=df.index, name="RALLY_ST")
