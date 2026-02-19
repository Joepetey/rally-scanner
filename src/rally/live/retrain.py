"""
Weekly model retraining — batch fetch + parallel training for all assets.

Usage:
    python retrain.py                          # retrain all S&P 500 + Nasdaq 100
    python retrain.py --tickers AAPL MSFT SPY  # retrain specific tickers
    python retrain.py --validate               # compare auto-cal vs hand-tuned
    python retrain.py --workers 4              # limit parallel workers
    python retrain.py --no-cache               # disable OHLCV disk cache
"""

import argparse
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Limit per-worker thread count BEFORE importing numpy/sklearn/hmmlearn.
# Without this, each subprocess spawns os.cpu_count() threads → 2 workers × 16
# threads = 32 threads thrashing 16 cores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..config import ASSETS, PARAMS, PIPELINE, AssetConfig
from ..core.calibrate import calibrate_thresholds
from ..core.data import fetch_daily_batch, fetch_vix_safe, merge_vix
from ..core.features import build_features
from ..core.hmm import fit_hmm, predict_hmm_probs
from ..core.labels import compute_labels
from ..core.model import ALL_FEATURE_COLS
from ..core.persistence import load_manifest, save_model
from ..core.universe import get_universe

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


def train_last_fold(
    df: pd.DataFrame, asset: AssetConfig, live_features: bool = True,
) -> dict | None:
    """
    Train only the most recent fold (last 5 years of data).
    Returns dict of model artifacts ready for save_model(), or None on failure.
    """
    p = PARAMS
    train_yrs = p.walk_forward_train_years
    target_col = "RALLY_ST"
    feature_cols = ALL_FEATURE_COLS

    # Build features
    df = build_features(df, live=live_features)
    df["RALLY_ST"] = compute_labels(df, asset)

    years = df.index.year.unique().sort_values()
    max_year = int(years.max())
    train_start_year = max_year - train_yrs + 1

    # Use last train_yrs years
    train_mask = df.index.year >= train_start_year
    df_train_raw = df.loc[train_mask]

    if len(df_train_raw) < 200:
        return None

    # Fit HMM on training data
    try:
        hmm_model, hmm_scaler, state_order = fit_hmm(df_train_raw)
    except (ValueError, np.linalg.LinAlgError):
        # HMM failed (convergence or singular covariance) — train without HMM features
        hmm_model, hmm_scaler, state_order = None, None, None
    hmm_probs = predict_hmm_probs(hmm_model, hmm_scaler, state_order, df_train_raw)
    df_train_full = df_train_raw.join(hmm_probs)

    # Filter valid rows
    valid = df_train_full[feature_cols + [target_col]].notna().all(axis=1)
    df_train = df_train_full.loc[valid]

    if len(df_train) < 200:
        return None

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    # Check for sufficient positive labels
    n_pos = y_train.sum()
    if n_pos < 10:
        return None

    # Standardize
    lr_scaler = StandardScaler()
    X_train_s = lr_scaler.fit_transform(X_train)

    # Fit logistic regression
    lr = LogisticRegression(
        C=1.0, l1_ratio=0, solver="lbfgs",
        max_iter=1000, class_weight="balanced",
    )
    lr.fit(X_train_s, y_train)

    # Isotonic calibration on last 20% as validation
    val_split = int(len(X_train_s) * 0.8)
    X_val_s = X_train_s[val_split:]
    y_val = y_train[val_split:]
    raw_val_probs = lr.predict_proba(X_val_s)[:, 1]

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    if y_val.sum() >= 3:
        iso.fit(raw_val_probs, y_val)
    else:
        # Not enough positives for calibration — use raw probs
        iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    return {
        "lr_model": lr,
        "lr_scaler": lr_scaler,
        "iso_calibrator": iso,
        "hmm_model": hmm_model,
        "hmm_scaler": hmm_scaler,
        "state_order": state_order,
        "feature_cols": feature_cols,
        "train_start": str(train_start_year),
        "train_end": str(max_year),
    }


# ---------------------------------------------------------------------------
# Worker function for parallel training (runs in subprocess)
# ---------------------------------------------------------------------------

def _train_single_worker(
    ticker: str,
    df: pd.DataFrame,
    vix_data: pd.Series | None,
    live_features: bool,
) -> tuple[str, dict | None, AssetConfig | None, str]:
    """
    Train a single ticker in a subprocess.
    Returns (ticker, artifacts_or_None, asset_config_or_None, status_message).
    """
    try:
        if vix_data is not None:
            df = merge_vix(df, vix_data)

        if len(df) < 500:
            return (ticker, None, None, f"SKIP ({len(df)} bars)")

        asset = calibrate_thresholds(df, ticker)
        artifacts = train_last_fold(df, asset, live_features=live_features)

        if artifacts is None:
            return (ticker, None, None, "FAIL (training returned None)")

        return (ticker, artifacts, asset,
                f"OK r_up={asset.r_up:.3f} d_dn={asset.d_dn:.3f}")

    except Exception as e:
        return (ticker, None, None, f"ERROR: {e}")


def _is_fresh(manifest_entry: dict, max_age_days: int) -> bool:
    """Check if a model was trained within the last N days."""
    try:
        saved_at = datetime.fromisoformat(manifest_entry["saved_at"])
        return (datetime.now() - saved_at).days < max_age_days
    except (KeyError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Main retraining pipeline
# ---------------------------------------------------------------------------

def retrain_all(tickers: list[str] | None = None, validate: bool = False) -> None:
    if tickers is None:
        logger.info("Fetching universe...")
        tickers = get_universe()

    n_workers = min(PIPELINE.n_workers, os.cpu_count() or 4)
    t0_total = time.time()

    # --- Phase 0: Fetch VIX ---
    logger.info("Fetching VIX data...")
    vix_data = fetch_vix_safe()
    if vix_data is not None:
        logger.info("VIX: %d bars loaded", len(vix_data))

    # --- Phase 1: Batch fetch all OHLCV data ---
    t0_fetch = time.time()
    logger.info("Phase 1: Batch fetching %d tickers...", len(tickers))
    ohlcv_data = fetch_daily_batch(tickers)
    fetched = [t for t in tickers if t in ohlcv_data]
    fetch_time = time.time() - t0_fetch
    logger.info("Fetched %d/%d tickers (%.1fs)", len(fetched), len(tickers), fetch_time)

    # --- Phase 1.5: Skip fresh models ---
    if PIPELINE.skip_fresh_enabled:
        manifest = load_manifest()
        to_train = []
        skipped = 0
        for ticker in fetched:
            entry = manifest.get(ticker)
            if entry and _is_fresh(entry, PIPELINE.skip_fresh_days):
                skipped += 1
            else:
                to_train.append(ticker)
        if skipped:
            logger.info("Skipping %d fresh models (< %d days old)",
                        skipped, PIPELINE.skip_fresh_days)
        fetched = to_train

    # --- Phase 2: Parallel model training ---
    logger.info("=" * 70)
    logger.info("RALLY DETECTOR — WEEKLY RETRAIN")
    logger.info("%s  |  %d assets  |  %d workers",
                datetime.now().strftime("%Y-%m-%d %H:%M"), len(fetched), n_workers)
    logger.info("=" * 70)

    success, failed = [], []
    t0_train = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for ticker in fetched:
            future = executor.submit(
                _train_single_worker,
                ticker, ohlcv_data[ticker], vix_data, PARAMS.train_live_features,
            )
            futures[future] = ticker

        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                t_name, artifacts, asset, msg = future.result(timeout=120)
                logger.info("[%d/%d] %s: %s", i, len(fetched), t_name, msg)

                if artifacts is not None and asset is not None:
                    save_model(t_name, artifacts, asset)
                    success.append(t_name)

                    if validate and t_name in ASSETS:
                        hand = ASSETS[t_name]
                        logger.info("  [hand: r_up=%.3f d_dn=%.3f]",
                                    hand.r_up, hand.d_dn)
                else:
                    failed.append((t_name, msg))

            except Exception as e:
                logger.error("[%d/%d] %s: TIMEOUT/ERROR: %s",
                             i, len(fetched), ticker, e)
                failed.append((ticker, str(e)))

    train_time = time.time() - t0_train
    total_time = time.time() - t0_total

    # Summary
    logger.info("=" * 70)
    logger.info("RETRAIN COMPLETE: %d success, %d failed", len(success), len(failed))
    logger.info("Fetch: %.1fs  |  Train: %.1fs  |  Total: %.1fs",
                fetch_time, train_time, total_time)
    if failed:
        logger.warning("Failed:")
        for t, reason in failed[:20]:
            logger.warning("  %s: %s", t, reason)
        if len(failed) > 20:
            logger.warning("  ... and %d more", len(failed) - 20)
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly model retraining")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers to retrain (default: full universe)")
    parser.add_argument("--validate", action="store_true",
                        help="Compare auto-calibrated vs hand-tuned for original 14 assets")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: from config)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable OHLCV disk cache")
    args = parser.parse_args()

    if args.workers:
        PIPELINE.n_workers = args.workers
    if args.no_cache:
        PIPELINE.cache_enabled = False

    retrain_all(tickers=args.tickers, validate=args.validate)


if __name__ == "__main__":
    main()
