"""
Daily market rally scanner — load models, scan assets, return signals.
"""

import logging
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd
from rally_ml.config import CONFIGS_BY_NAME, PARAMS, AssetConfig

logger = logging.getLogger(__name__)
from rally_ml.core.data import (
    fetch_daily,
    fetch_daily_batch,
    fetch_quotes,
    fetch_vix_safe,
    merge_vix,
)
from rally_ml.core.features import build_features
from rally_ml.core.hmm import predict_hmm_probs
from rally_ml.core.persistence import load_manifest, load_model

from trading.signals import compute_position_size, generate_signals

warnings.filterwarnings("ignore")

LOOKBACK_DAYS = 500  # enough for 252-bar percentile window + buffer
_MAX_SCAN_WORKERS = 8
_MAX_WATCHLIST_WORKERS = 4


class ScanTask(NamedTuple):
    ticker: str
    vix_data: "pd.Series | None"
    ohlcv_df: "pd.DataFrame | None"


def apply_config(config_name: str) -> None:
    """Override PARAMS with a named configuration."""
    cfg = CONFIGS_BY_NAME.get(config_name)
    if cfg is None:
        logger.error("Unknown config '%s'. Available: %s",
                      config_name, ", ".join(CONFIGS_BY_NAME.keys()))
        raise SystemExit(1)
    PARAMS.p_rally_threshold = cfg.p_rally
    PARAMS.comp_score_threshold = cfg.comp_score
    PARAMS.vol_target_k = cfg.vol_k
    PARAMS.max_risk_frac = cfg.max_risk
    PARAMS.profit_atr_mult = cfg.profit_atr
    PARAMS.time_stop_bars = cfg.time_stop


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


def _fetch_scan_data(
    scan_tickers: list[str], start: str, verbose: bool = True,
) -> tuple["pd.Series | None", dict]:
    """Fetch VIX and batch OHLCV data for all tickers. Returns (vix_data, ohlcv_cache)."""
    logger.info("Fetching VIX data...")
    vix_data = fetch_vix_safe(start=start, verbose=verbose)

    logger.info("Batch-fetching OHLCV data...")
    try:
        ohlcv_cache = fetch_daily_batch(scan_tickers, start=start)
    except Exception as e:
        logger.warning("Batch fetch failed (%s), falling back to individual fetches", e)
        ohlcv_cache = {}

    return vix_data, ohlcv_cache


def _run_parallel_scan(
    scan_tickers: list[str],
    vix_data: "pd.Series | None",
    ohlcv_cache: dict,
    max_workers: int,
) -> list[dict]:
    """Run scan across tickers using a process pool."""
    n_workers = min(max_workers, len(scan_tickers))
    work_items = [
        ScanTask(ticker, vix_data, ohlcv_cache.get(ticker))
        for ticker in scan_tickers
    ]

    results = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("forkserver"),
    ) as pool:
        futures = {
            pool.submit(_scan_one, item): item[0]
            for item in work_items
        }
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            logger.debug("Scanned %s (%d/%d)", ticker, i, len(scan_tickers))
            results.append(future.result())

    return results


def _update_signals_with_live_prices(signals: list[dict]) -> None:
    """Replace daily close with live quote price for signal tickers (in-place)."""
    if not signals:
        return
    live_tickers = [s["ticker"] for s in signals]
    quotes = fetch_quotes(live_tickers)
    for sig in signals:
        q = quotes.get(sig["ticker"], {})
        if "price" in q:
            daily_close = sig["close"]
            sig["close"] = q["price"]
            logger.info(
                "%s: live price $%.2f (daily close was $%.2f)",
                sig["ticker"], q["price"], daily_close,
            )


def scan_all(
    tickers: list[str] | None = None,
    config_name: str = "conservative",
) -> list[dict]:
    """Scan all trained assets (or specific tickers) and return results."""
    apply_config(config_name)

    manifest = load_manifest()
    if not manifest:
        logger.error("No trained models found. Run retrain.py first.")
        return []

    if tickers:
        scan_tickers = [t for t in tickers if t in manifest]
        missing = [t for t in tickers if t not in manifest]
        if missing:
            logger.warning("No models for: %s", ", ".join(missing))
    else:
        scan_tickers = sorted(manifest.keys())

    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    vix_data, ohlcv_cache = _fetch_scan_data(scan_tickers, start)

    logger.info(
        "Scanning %d assets [%s] P(rally)>%.0f%% Comp>%s",
        len(scan_tickers), config_name.upper(),
        PARAMS.p_rally_threshold * 100, PARAMS.comp_score_threshold,
    )

    results = _run_parallel_scan(scan_tickers, vix_data, ohlcv_cache, _MAX_SCAN_WORKERS)

    signals = [r for r in results if r.get("signal")]
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    err_count = sum(1 for r in results if r.get("status") != "ok")

    _update_signals_with_live_prices(signals)

    logger.info(
        "Scan complete — %d/%d ok, %d signals, %d errors",
        ok_count, len(scan_tickers), len(signals), err_count,
    )

    return results


def scan_watchlist(
    tickers: list[str],
    config_name: str = "conservative",
) -> list[dict]:
    """Scan a specific set of tickers (e.g. near-threshold watchlist).

    Lighter than scan_all: fewer workers.
    Returns list of result dicts (same format as scan_all).
    """
    apply_config(config_name)

    manifest = load_manifest()
    if not manifest:
        return []

    scan_tickers = [t for t in tickers if t in manifest]
    if not scan_tickers:
        return []

    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    vix_data, ohlcv_cache = _fetch_scan_data(scan_tickers, start, verbose=False)

    results = _run_parallel_scan(scan_tickers, vix_data, ohlcv_cache, _MAX_WATCHLIST_WORKERS)

    signals = [r for r in results if r.get("signal")]
    _update_signals_with_live_prices(signals)

    return results
