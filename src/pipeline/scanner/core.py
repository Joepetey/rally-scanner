"""Single-asset ML scanning — load artifacts, predict, generate signals."""

import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from rally_ml.config import AssetConfig
from rally_ml.config.trading import TradingConfig
from rally_ml.core.data import fetch_daily, merge_vix
from rally_ml.core.features import build_features
from rally_ml.core.hmm import predict_hmm_probs

from trading.signals import compute_position_size, generate_signals

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 500  # enough for 252-bar percentile window + buffer


def scan_single(
    ticker: str, artifacts: dict,
    vix_data: pd.Series | None = None,
    ohlcv_data: pd.DataFrame | None = None,
    config: TradingConfig | None = None,
) -> dict:
    """Scan a single asset. Returns dict with signal info."""
    ac = artifacts["asset_config"]
    asset = AssetConfig(**ac)

    if ohlcv_data is not None:
        df = ohlcv_data.copy()
    else:
        start = (
            datetime.now() - timedelta(days=LOOKBACK_DAYS)
        ).strftime("%Y-%m-%d")
        df = fetch_daily(asset, start=start)

    if len(df) < 300:
        return {"ticker": ticker, "status": "insufficient_data", "bars": len(df)}

    if vix_data is not None:
        df = merge_vix(df, vix_data)

    df = build_features(df, live=True)

    hmm_probs = predict_hmm_probs(
        artifacts.get("hmm_model"),
        artifacts.get("hmm_scaler"),
        artifacts.get("state_order"),
        df,
    )
    df = df.join(hmm_probs)

    feature_cols = artifacts["feature_cols"]
    latest = df.iloc[[-1]]

    missing = latest[feature_cols].isna().any(axis=1).iloc[0]
    has_inf = np.isinf(latest[feature_cols].values).any()
    if missing or has_inf:
        return {"ticker": ticker, "status": "features_incomplete"}

    lr = artifacts["lr_model"]
    scaler = artifacts["lr_scaler"]
    iso = artifacts["iso_calibrator"]

    X = latest[feature_cols].values
    X_s = scaler.transform(X)
    raw_prob = lr.predict_proba(X_s)[:, 1][0]
    cal_prob = float(iso.predict([raw_prob])[0])

    latest_pred = latest.copy()
    latest_pred["P_RALLY"] = cal_prob

    signal = generate_signals(latest_pred, require_trend=True, config=config)
    is_signal = bool(signal.iloc[0])

    size = 0.0
    if is_signal:
        size = float(compute_position_size(latest_pred, config=config).iloc[0])
        if size == 0.0:
            is_signal = False

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
        "golden_cross": int(row.get("GOLDEN_CROSS", 0)),
        "macd_hist": round(float(row.get("MACD_HIST", 0)), 5),
        "vol_ratio": round(float(row.get("VOL_RATIO", 1)), 2),
        "vix_pctile": round(float(row.get("VIX_PCTILE", 0.5)), 3),
        "ma50": round(float(row.get("MA50", 0)), 2),
    }
