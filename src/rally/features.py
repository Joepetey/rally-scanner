"""
Feature engineering — compression, range structure, failed breakdowns, trend.
"""

import numpy as np
import pandas as pd

from .config import PARAMS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of current value within rolling window (0-1)."""
    def _pct_rank(arr):
        if np.isnan(arr[-1]):
            return np.nan
        val = arr[-1]
        return np.nanmean(arr[:-1] <= val)
    return series.rolling(window, min_periods=window).apply(_pct_rank, raw=True)


# ---------------------------------------------------------------------------
# 3.1  Volatility Compression
# ---------------------------------------------------------------------------

def compute_compression(df: pd.DataFrame) -> pd.DataFrame:
    p = PARAMS
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Raw measures
    atr_raw = _atr(high, low, close, p.atr_period)
    atr_pct = atr_raw / close

    ma = close.rolling(p.bb_period, min_periods=p.bb_period).mean()
    std = close.rolling(p.bb_period, min_periods=p.bb_period).std()
    bb_width = (2 * std * 2) / ma  # (upper - lower) / MA  (2-sigma bands)

    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(p.rv_period, min_periods=p.rv_period).std()

    # Percentile ranks
    w = p.percentile_window
    p_atr = _rolling_percentile_rank(atr_pct, w)
    p_bb = _rolling_percentile_rank(bb_width, w)
    p_rv = _rolling_percentile_rank(rv, w)

    comp_score = 1 - (p_atr + p_bb + p_rv) / 3

    df = df.copy()
    df["ATR"] = atr_raw
    df["ATR_pct"] = atr_pct
    df["BB_width"] = bb_width
    df["RV"] = rv
    df["p_ATR"] = p_atr
    df["p_BB"] = p_bb
    df["p_RV"] = p_rv
    df["COMP_SCORE"] = comp_score
    return df


# ---------------------------------------------------------------------------
# 3.2  Range Structure
# ---------------------------------------------------------------------------

def compute_range(df: pd.DataFrame) -> pd.DataFrame:
    p = PARAMS
    df = df.copy()
    df["RangeHigh"] = df["High"].rolling(p.range_period, min_periods=p.range_period).max()
    df["RangeLow"] = df["Low"].rolling(p.range_period, min_periods=p.range_period).min()
    df["RangeWidth"] = (df["RangeHigh"] - df["RangeLow"]) / df["Close"]
    df["RangeTightness"] = df["RangeWidth"] / df["ATR_pct"]
    return df


# ---------------------------------------------------------------------------
# 3.3  Failed Breakdown Detector  (TRAP ENGINE)
# ---------------------------------------------------------------------------

def compute_failed_breakdown(df: pd.DataFrame, live: bool = False) -> pd.DataFrame:
    """
    Detect bear traps: breakdown attempts that fail and reverse.

    Uses a shorter range (trap_range_period) for breakdown detection and
    relaxed confirmation conditions to generate enough signals while
    preserving the core insight: traps during compression resolve upward.

    Produces FAIL_DN_SCORE — continuous, positive when trap detected.
    """
    p = PARAMS
    k = p.trap_lookforward
    df = df.copy()

    rsi = _rsi(df["Close"], p.rsi_period)
    df["RSI"] = rsi

    # Use shorter range for breakdown detection (more frequent signals)
    trap_range_low = df["Low"].rolling(p.trap_range_period, min_periods=p.trap_range_period).min()
    trap_rl_prev = trap_range_low.shift(1)

    # Breakdown: Low penetrates or probes within 0.3*ATR of range low
    atr_vals = df["ATR"]
    breakdown = df["Low"] <= trap_rl_prev + 0.3 * atr_vals

    # RV percentile for trap qualification (backward-looking)
    rv_pctile = _rolling_percentile_rank(df["RV"], p.percentile_window)

    fail_dn_score = pd.Series(0.0, index=df.index)
    n = len(df)

    for i in range(n):
        if not breakdown.iloc[i]:
            continue
        rl = trap_rl_prev.iloc[i]
        if np.isnan(rl):
            continue
        atr_val = atr_vals.iloc[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        # Check: was vol compressed at the breakdown bar?
        rv_pct_i = rv_pctile.iloc[i] if not np.isnan(rv_pctile.iloc[i]) else 1.0
        if rv_pct_i > p.rv_trap_percentile:
            continue

        best = 0.0
        rsi_i = rsi.iloc[i]

        for j in range(1, k + 1):
            idx = i + j
            if idx >= n:
                break
            close_j = df["Close"].iloc[idx]
            if np.isnan(close_j):
                continue

            # Core condition: close recovers above range low
            if close_j > rl:
                score = (close_j - rl) / atr_val
                # Bonus if RSI also recovering
                rsi_j = rsi.iloc[idx]
                if not np.isnan(rsi_j) and not np.isnan(rsi_i) and rsi_j > rsi_i:
                    score *= 1.5
                best = max(best, score)

        fail_dn_score.iloc[i] = best

    if live:
        # In live mode, lag by k bars so we only use confirmed (past) data.
        # Score at bar i was computed using bars i+1..i+k (future).
        # Shifting forward by k means at bar i we see the score from bar i-k,
        # which was confirmed by bars i-k+1..i (all past). Same signal, 5-bar delay.
        df["FAIL_DN_SCORE"] = fail_dn_score.shift(k)
    else:
        df["FAIL_DN_SCORE"] = fail_dn_score
    return df


# ---------------------------------------------------------------------------
# 3.4  Trend Context
# ---------------------------------------------------------------------------

def compute_trend(df: pd.DataFrame) -> pd.DataFrame:
    p = PARAMS
    df = df.copy()
    ma200 = df["Close"].rolling(p.ma_long, min_periods=p.ma_long).mean()
    df["MA200"] = ma200
    df["Trend"] = (df["Close"] > ma200).astype(float)
    return df


# ---------------------------------------------------------------------------
# 3.5  Volume Confirmation
# ---------------------------------------------------------------------------

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume ratio and percentile — confirms breakout conviction."""
    p = PARAMS
    df = df.copy()
    vol = df["Volume"].astype(float)
    vol_ma = vol.rolling(20, min_periods=20).mean()
    df["VOL_RATIO"] = vol / vol_ma
    df["p_VOL"] = _rolling_percentile_rank(df["VOL_RATIO"], p.percentile_window)
    return df


# ---------------------------------------------------------------------------
# 3.6  Golden Cross (50MA > 200MA)
# ---------------------------------------------------------------------------

def compute_golden_cross(df: pd.DataFrame) -> pd.DataFrame:
    """Golden cross signal: 50-day MA above 200-day MA."""
    df = df.copy()
    ma50 = df["Close"].rolling(50, min_periods=50).mean()
    df["MA50"] = ma50
    df["GOLDEN_CROSS"] = (ma50 > df["MA200"]).astype(float)
    return df


# ---------------------------------------------------------------------------
# 3.7  MACD Histogram
# ---------------------------------------------------------------------------

def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD histogram — momentum confirmation, normalized by price."""
    df = df.copy()
    close = df["Close"]
    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, min_periods=9).mean()
    # Normalize by close for cross-asset comparability
    df["MACD_HIST"] = (macd_line - signal_line) / close
    return df


# ---------------------------------------------------------------------------
# 3.8  VIX Fear Gauge
# ---------------------------------------------------------------------------

def compute_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    """VIX percentile — market-wide fear/greed gauge.
    Requires VIX_Close column (merged via data.merge_vix before calling).
    Falls back to neutral 0.5 if not available.
    """
    p = PARAMS
    df = df.copy()
    if "VIX_Close" in df.columns and df["VIX_Close"].notna().sum() > p.percentile_window:
        df["VIX_PCTILE"] = _rolling_percentile_rank(df["VIX_Close"], p.percentile_window)
    else:
        df["VIX_PCTILE"] = 0.5
    return df


# ---------------------------------------------------------------------------
# 3.9  Assemble final feature vector
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, live: bool = False) -> pd.DataFrame:
    """Run full pipeline and return df with feature columns attached.

    Args:
        live: If True, use lagged FAIL_DN_SCORE (no forward-looking data).
    """
    df = compute_compression(df)
    df = compute_range(df)
    df = compute_failed_breakdown(df, live=live)
    df = compute_trend(df)
    df = compute_volume_features(df)
    df = compute_golden_cross(df)
    df = compute_macd(df)
    df = compute_vix_features(df)

    # Interaction term
    df["COMP_x_FAIL"] = df["COMP_SCORE"] * df["FAIL_DN_SCORE"]

    return df


FEATURE_COLS = [
    "COMP_SCORE",
    "RangeWidth",
    "RangeTightness",
    "FAIL_DN_SCORE",
    "Trend",
    "COMP_x_FAIL",
    # Volume confirmation
    "p_VOL",
    # Golden cross
    "GOLDEN_CROSS",
    # MACD momentum
    "MACD_HIST",
    # VIX fear gauge
    "VIX_PCTILE",
]
