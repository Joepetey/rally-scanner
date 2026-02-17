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

from .calibrate import calibrate_thresholds
from .config import ASSETS, PARAMS, PIPELINE, AssetConfig
from .data import fetch_daily_batch, fetch_vix, merge_vix
from .features import build_features
from .hmm import fit_hmm, predict_hmm_probs
from .labels import compute_labels
from .model import ALL_FEATURE_COLS
from .persistence import load_manifest, save_model
from .universe import get_universe

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
        hmm_probs = predict_hmm_probs(hmm_model, hmm_scaler, state_order, df_train_raw)
        df_train_full = df_train_raw.join(hmm_probs)
    except (ValueError, np.linalg.LinAlgError):
        # HMM failed (convergence or singular covariance) — train without HMM features
        df_train_full = df_train_raw.copy()
        for col in ["P_compressed", "P_expanding", "HMM_transition_signal"]:
            df_train_full[col] = 0.0
        hmm_model, hmm_scaler, state_order = None, None, None

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
        print("  Fetching universe...")
        tickers = get_universe()

    n_workers = min(PIPELINE.n_workers, os.cpu_count() or 4)
    t0_total = time.time()

    # --- Phase 0: Fetch VIX ---
    print("  Fetching VIX data...")
    vix_data = None
    try:
        vix_data = fetch_vix()
        print(f"  VIX: {len(vix_data)} bars loaded")
    except Exception as e:
        print(f"  WARNING: Could not fetch VIX: {e}")

    # --- Phase 1: Batch fetch all OHLCV data ---
    t0_fetch = time.time()
    print(f"\n  Phase 1: Batch fetching {len(tickers)} tickers...")
    ohlcv_data = fetch_daily_batch(tickers)
    fetched = [t for t in tickers if t in ohlcv_data]
    fetch_time = time.time() - t0_fetch
    print(f"  Fetched {len(fetched)}/{len(tickers)} tickers ({fetch_time:.1f}s)")

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
            print(f"  Skipping {skipped} fresh models (< {PIPELINE.skip_fresh_days} days old)")
        fetched = to_train

    # --- Phase 2: Parallel model training ---
    print(f"\n{'='*70}")
    print("  RALLY DETECTOR — WEEKLY RETRAIN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
          f"{len(fetched)} assets  |  {n_workers} workers")
    print(f"{'='*70}")

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
                print(f"  [{i}/{len(fetched)}] {t_name}: {msg}")

                if artifacts is not None and asset is not None:
                    save_model(t_name, artifacts, asset)
                    success.append(t_name)

                    if validate and t_name in ASSETS:
                        hand = ASSETS[t_name]
                        print(f"         [hand: r_up={hand.r_up:.3f} d_dn={hand.d_dn:.3f}]")
                else:
                    failed.append((t_name, msg))

            except Exception as e:
                print(f"  [{i}/{len(fetched)}] {ticker}: TIMEOUT/ERROR: {e}")
                failed.append((ticker, str(e)))

    train_time = time.time() - t0_train
    total_time = time.time() - t0_total

    # Summary
    print(f"\n{'='*70}")
    print(f"  RETRAIN COMPLETE: {len(success)} success, {len(failed)} failed")
    print(f"  Fetch: {fetch_time:.1f}s  |  Train: {train_time:.1f}s  |  "
          f"Total: {total_time:.1f}s")
    if failed:
        print("  Failed:")
        for t, reason in failed[:20]:
            print(f"    {t}: {reason}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more")
    print(f"{'='*70}")


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
