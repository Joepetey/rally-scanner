"""Per-ticker scanning logic — must be top-level for ProcessPoolExecutor pickling."""

import logging
from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

from config import AssetConfig
from core.data import fetch_daily, merge_vix
from core.features import build_features
from core.hmm import predict_hmm_probs
from core.persistence import load_model
from trading.signals import compute_position_size, generate_signals

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 500  # enough for 252-bar percentile window + buffer


class ScanTask(NamedTuple):
    ticker: str
    vix_data: "pd.Series | None"
    ohlcv_df: "pd.DataFrame | None"


def scan_single(
    ticker: str, artifacts: dict,
    vix_data: pd.Series | None = None,
    ohlcv_data: pd.DataFrame | None = None,
) -> dict:
    """Scan a single asset. Returns dict with signal info."""
    # Reconstruct AssetConfig
    ac = artifacts["asset_config"]
    asset = AssetConfig(**ac)

    # Use pre-fetched data if available, otherwise fetch individually
    if ohlcv_data is not None:
        df = ohlcv_data.copy()
    else:
        start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        df = fetch_daily(asset, start=start)

    if len(df) < 300:
        return {"ticker": ticker, "status": "insufficient_data", "bars": len(df)}

    # Merge VIX data if available
    if vix_data is not None:
        df = merge_vix(df, vix_data)

    # Build features (live mode — lagged FAIL_DN_SCORE)
    df = build_features(df, live=True)

    # HMM predictions (predict_hmm_probs handles model=None gracefully)
    hmm_probs = predict_hmm_probs(
        artifacts.get("hmm_model"),
        artifacts.get("hmm_scaler"),
        artifacts.get("state_order"),
        df,
    )
    df = df.join(hmm_probs)

    # Get the latest bar
    feature_cols = artifacts["feature_cols"]
    latest = df.iloc[[-1]]

    # Check features are available
    missing = latest[feature_cols].isna().any(axis=1).iloc[0]
    has_inf = np.isinf(latest[feature_cols].values).any()
    if missing or has_inf:
        return {"ticker": ticker, "status": "features_incomplete"}

    # Predict rally probability
    lr = artifacts["lr_model"]
    scaler = artifacts["lr_scaler"]
    iso = artifacts["iso_calibrator"]

    X = latest[feature_cols].values
    X_s = scaler.transform(X)
    raw_prob = lr.predict_proba(X_s)[:, 1][0]
    cal_prob = float(iso.predict([raw_prob])[0])

    # Build prediction row for signal generation
    latest_pred = latest.copy()
    latest_pred["P_RALLY"] = cal_prob

    signal = generate_signals(latest_pred, require_trend=True)
    is_signal = bool(signal.iloc[0])

    size = 0.0
    if is_signal:
        size = float(compute_position_size(latest_pred).iloc[0])
        if size == 0.0:
            is_signal = False  # below minimum position size

    # Extract diagnostics
    row = df.iloc[-1]
    return {
        "ticker": ticker,
        "status": "ok",
        "date": str(row.name.date()),
        "close": round(float(row["Close"]), 2),
        "p_rally": round(cal_prob, 3),
        "p_rally_raw": round(float(raw_prob), 3),
        "comp_score": round(float(row.get("COMP_SCORE", 0)), 3),
        "fail_dn": round(float(row.get("FAIL_DN_SCORE", 0)), 3),
        "trend": int(row.get("Trend", 0)),
        "hmm_compressed": round(float(row.get("P_compressed", 0)), 3),
        "rv_pctile": round(float(row.get("p_RV", 0)), 3),
        "atr_pct": round(float(row.get("ATR_pct", 0)), 4),
        "atr": round(float(row.get("ATR", 0)), 4),
        "signal": is_signal,
        "size": round(size, 3),
        "r_up": asset.r_up,
        "d_dn": asset.d_dn,
        "range_high": round(float(row.get("RangeHigh", 0)), 2),
        "range_low": round(float(row.get("RangeLow", 0)), 2),
        "rsi": round(float(row.get("RSI", 0)), 1),
        # New features
        "golden_cross": int(row.get("GOLDEN_CROSS", 0)),
        "macd_hist": round(float(row.get("MACD_HIST", 0)), 5),
        "vol_ratio": round(float(row.get("VOL_RATIO", 1)), 2),
        "vix_pctile": round(float(row.get("VIX_PCTILE", 0.5)), 3),
        "ma50": round(float(row.get("MA50", 0)), 2),
    }


def _scan_one(task: ScanTask) -> dict:
    """Worker function for parallel scanning (must be top-level for pickling)."""
    ticker, vix_data, ohlcv_df = task
    try:
        artifacts = load_model(ticker)
        return scan_single(
            ticker, artifacts,
            vix_data=vix_data,
            ohlcv_data=ohlcv_df,
        )
    except Exception as e:
        return {"ticker": ticker, "status": f"error: {e}"}
