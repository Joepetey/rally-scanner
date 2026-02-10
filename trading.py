"""
Trading rules — entry signals, position sizing, stops, and exits.
"""

import numpy as np
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
    Vol-targeted sizing: size = k * (P_RALLY - 0.5) / ATR_pct
    Clamped to [0, max_risk].
    """
    p = PARAMS
    raw_size = p.vol_target_k * (preds["P_RALLY"] - 0.5) / preds["ATR_pct"]
    return raw_size.clip(lower=0.0, upper=p.max_risk_frac)


def simulate_trades(preds: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    """
    Simulate trades bar-by-bar with exits:
      1. Stop-loss at structural level (RangeLow)
      2. +2*ATR profit target
      3. ATR trailing stop (entry - 1.5*ATR, ratchets up)
      4. Time stop: 10 bars
      5. RV percentile > 80% and bearish close (expansion exhaustion)

    Returns a DataFrame of completed trades.
    """
    p = PARAMS
    n = len(preds)
    close = preds["Close"].values
    high = preds["High"].values
    low = preds["Low"].values
    atr = preds["ATR"].values

    # RV percentile
    rv_pct = preds["p_RV"].values if "p_RV" in preds.columns else np.full(n, 0.5)

    trades = []
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    stop_price = 0.0
    trailing_stop = 0.0
    size = 0.0
    bars_held = 0
    highest_close = 0.0

    for i in range(n):
        if in_trade:
            bars_held += 1

            # Track highest close for trailing stop
            if close[i] > highest_close:
                highest_close = close[i]
                # Ratchet trailing stop up: highest close - 1.5*ATR
                new_trail = highest_close - 1.5 * atr[entry_idx]
                trailing_stop = max(trailing_stop, new_trail)

            # Check exit conditions (order matters)
            exit_reason = None
            exit_price = close[i]

            # Hard stop-loss (structural)
            if low[i] <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"

            # Profit target: +2*ATR from entry
            elif high[i] >= entry_price + p.profit_atr_mult * atr[entry_idx]:
                exit_price = entry_price + p.profit_atr_mult * atr[entry_idx]
                exit_reason = "profit_target"

            # ATR trailing stop (only after 2+ bars to let trade breathe)
            elif bars_held >= 2 and close[i] < trailing_stop:
                exit_reason = "trail_stop"

            # Time stop
            elif bars_held >= p.time_stop_bars:
                exit_reason = "time_stop"

            # Expansion exhaustion
            elif rv_pct[i] > p.rv_exit_pct and close[i] < close[i - 1]:
                exit_reason = "vol_exhaustion"

            if exit_reason:
                pnl_pct = (exit_price / entry_price - 1)
                slice_low = low[entry_idx + 1: i + 1]
                slice_high = high[entry_idx + 1: i + 1]
                mae = (min(slice_low.min(), entry_price) if len(slice_low) > 0
                       else entry_price) / entry_price - 1
                mfe = (max(slice_high.max(), entry_price) if len(slice_high) > 0
                       else entry_price) / entry_price - 1

                trades.append({
                    "entry_date": preds.index[entry_idx],
                    "exit_date": preds.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "size": size,
                    "pnl_pct": pnl_pct,
                    "pnl_sized": pnl_pct * size,
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                    "mae": mae,
                    "mfe": mfe,
                    "comp_score": preds["COMP_SCORE"].iloc[entry_idx],
                    "p_rally": preds["P_RALLY"].iloc[entry_idx],
                    "trend": preds["Trend"].iloc[entry_idx],
                })
                in_trade = False

        if not in_trade and signal.iloc[i]:
            entry_idx = i
            entry_price = close[i]
            stop_price = preds["RangeLow"].iloc[i]
            if np.isnan(stop_price) or stop_price >= entry_price:
                stop_price = entry_price * 0.97  # fallback: 3% stop
            size = compute_position_size(preds.iloc[[i]]).iloc[0]
            if size > 0:
                in_trade = True
                bars_held = 0
                highest_close = entry_price
                trailing_stop = entry_price - 1.5 * atr[i]

    return pd.DataFrame(trades) if trades else pd.DataFrame()
