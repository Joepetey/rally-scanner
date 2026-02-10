"""
Weekly model retraining — train last fold for all assets, save to disk.

Usage:
    python retrain.py                          # retrain all S&P 500 + Nasdaq 100
    python retrain.py --tickers AAPL MSFT SPY  # retrain specific tickers
    python retrain.py --validate               # compare auto-cal vs hand-tuned
"""

import argparse
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import PARAMS, ASSETS, AssetConfig
from data import fetch_daily, fetch_vix, merge_vix
from features import build_features
from labels import compute_labels
from hmm import fit_hmm, predict_hmm_probs
from model import ALL_FEATURE_COLS
from persistence import save_model
from calibrate import calibrate_thresholds
from universe import get_universe

warnings.filterwarnings("ignore")


def train_last_fold(df: pd.DataFrame, asset: AssetConfig, live_features: bool = True):
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
    except Exception:
        # HMM failed — train without HMM features
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


def retrain_all(tickers=None, validate=False):
    if tickers is None:
        print("  Fetching universe...")
        tickers = get_universe()

    # Fetch VIX data once for all tickers
    print("  Fetching VIX data...")
    try:
        vix_data = fetch_vix()
        print(f"  VIX: {len(vix_data)} bars loaded")
    except Exception as e:
        print(f"  WARNING: Could not fetch VIX: {e}")
        vix_data = None

    print(f"{'='*70}")
    print(f"  RALLY DETECTOR — WEEKLY RETRAIN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {len(tickers)} assets")
    print(f"{'='*70}")

    success, failed = [], []

    for i, ticker in enumerate(tickers, 1):
        t0 = time.time()
        print(f"\n  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)

        try:
            # Fetch data (use ticker directly for most equities)
            tmp_asset = AssetConfig(ticker=ticker, asset_class="equity",
                                    r_up=0.03, d_dn=0.015)
            df = fetch_daily(tmp_asset)

            # Merge VIX data
            if vix_data is not None:
                df = merge_vix(df, vix_data)

            if len(df) < 500:
                print(f"SKIP ({len(df)} bars, need 500+)")
                failed.append((ticker, "insufficient data"))
                continue

            # Auto-calibrate thresholds
            asset = calibrate_thresholds(df, ticker)
            print(f"r_up={asset.r_up:.3f} d_dn={asset.d_dn:.3f}", end=" ", flush=True)

            if validate and ticker in ASSETS:
                hand = ASSETS[ticker]
                print(f"[hand: r_up={hand.r_up:.3f} d_dn={hand.d_dn:.3f}]", end=" ")

            # Train
            artifacts = train_last_fold(df, asset, live_features=PARAMS.train_live_features)
            if artifacts is None:
                print("FAIL (training returned None)")
                failed.append((ticker, "training failed"))
                continue

            # Save
            save_model(ticker, artifacts, asset)
            elapsed = time.time() - t0
            print(f"OK ({elapsed:.1f}s)")
            success.append(ticker)

            # Brief pause to avoid yfinance rate limiting
            time.sleep(0.3)

        except Exception as e:
            print(f"ERROR: {e}")
            failed.append((ticker, str(e)))

    # Summary
    print(f"\n{'='*70}")
    print(f"  RETRAIN COMPLETE: {len(success)} success, {len(failed)} failed")
    if failed:
        print(f"  Failed:")
        for t, reason in failed:
            print(f"    {t}: {reason}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly model retraining")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers to retrain (default: all S&P 100)")
    parser.add_argument("--validate", action="store_true",
                        help="Compare auto-calibrated vs hand-tuned for original 14 assets")
    args = parser.parse_args()
    retrain_all(tickers=args.tickers, validate=args.validate)
