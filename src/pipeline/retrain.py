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
import multiprocessing
import os
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Limit per-worker thread count BEFORE importing numpy/sklearn/hmmlearn.
# Without this, each subprocess spawns os.cpu_count() threads → 2 workers × 16
# threads = 32 threads thrashing 16 cores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import pandas as pd

from config import ASSETS, PARAMS, PIPELINE, AssetConfig
from core.calibrate import calibrate_thresholds
from core.data import fetch_daily_batch, fetch_vix_safe, merge_vix
from core.features import ALL_FEATURE_COLS, build_features
from core.labels import compute_labels
from core.persistence import load_manifest, save_model
from core.train import fit_model
from core.universe import get_universe

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


def train_last_fold(
    df: pd.DataFrame, asset: AssetConfig, live_features: bool = True,
) -> dict | None:
    """Train only the most recent fold (last 5 years of data).

    Returns dict of model artifacts ready for save_model(), or None on failure.
    """
    df = build_features(df, live=live_features)
    df["RALLY_ST"] = compute_labels(df, asset)

    max_year = int(df.index.year.max())
    train_start_year = max_year - PARAMS.walk_forward_train_years + 1

    df_train = df.loc[df.index.year >= train_start_year]
    if len(df_train) < 200:
        return None

    artifacts = fit_model(df_train, ALL_FEATURE_COLS, min_train=200, min_positives=10)
    if artifacts is None:
        return None

    artifacts["train_start"] = str(train_start_year)
    artifacts["train_end"] = str(max_year)
    return artifacts


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

def retrain_all(
    tickers: list[str] | None = None,
    validate: bool = False,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> None:
    """Retrain all models.

    progress_callback is called after each ticker completes:
        callback(done, total, success_count, fail_count)
    """
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

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("forkserver"),
    ) as executor:
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

                if progress_callback is not None:
                    progress_callback(i, len(fetched), len(success), len(failed))

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
