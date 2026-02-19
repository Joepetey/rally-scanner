"""
Full walk-forward backtest across the entire S&P 500 + Nasdaq 100 universe.

Runs the complete model pipeline (features → labels → walk-forward LR+HMM → OOS predictions)
for every ticker, then simulates portfolio performance across multiple trading configurations.

Saves intermediate results to disk so a crash doesn't lose all progress.

Usage:
    python backtest_universe.py                           # full universe (~512 assets)
    python backtest_universe.py --tickers AAPL MSFT SPY   # specific tickers
    python backtest_universe.py --resume                  # resume from saved predictions cache
    python backtest_universe.py --portfolio-only  # skip training, re-run portfolio sims
    python backtest_universe.py --workers 4               # limit parallel workers
    python backtest_universe.py --no-cache                # disable OHLCV disk cache
"""

import argparse
import json
import logging
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Limit per-worker thread count BEFORE importing numpy/sklearn/hmmlearn.
# Without this, each subprocess spawns os.cpu_count() threads → 2 workers × 16
# threads = 32 threads thrashing 16 cores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from .common import (  # noqa: E402
    CONFIGS,
    generate_signals_fast,
    simulate_portfolio,
    simulate_trades_fast,
)
from ..core.calibrate import calibrate_thresholds
from ..config import PIPELINE
from ..core.data import fetch_daily_batch, fetch_vix_safe, merge_vix
from ..core.features import build_features
from ..core.labels import compute_labels
from ..core.model import combine_predictions, walk_forward_train
from ..core.universe import get_universe

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
CACHE_DIR = PROJECT_ROOT / "backtest_cache"
CACHE_DIR.mkdir(exist_ok=True)
PREDICTIONS_CACHE = CACHE_DIR / "predictions.pkl"
RESULTS_FILE = CACHE_DIR / "results.json"


# ---------------------------------------------------------------------------
# Step 1: Train all assets (slow — cached to disk)
# ---------------------------------------------------------------------------

def _backtest_single_worker(
    ticker: str,
    df: pd.DataFrame,
    vix_data: pd.Series | None,
) -> tuple[str, pd.DataFrame | None, str]:
    """
    Run full walk-forward backtest for a single ticker in a subprocess.
    Returns (ticker, predictions_or_None, status_message).
    """
    try:
        if vix_data is not None:
            df = merge_vix(df, vix_data)

        if len(df) < 500:
            return (ticker, None, f"SKIP ({len(df)} bars)")

        asset = calibrate_thresholds(df, ticker)
        df = build_features(df)
        df["RALLY_ST"] = compute_labels(df, asset)

        folds = walk_forward_train(df)
        if not folds:
            return (ticker, None, "SKIP (no valid folds)")

        preds = combine_predictions(folds)
        if preds is None or len(preds) == 0:
            return (ticker, None, "SKIP (no predictions)")

        return (ticker, preds, f"OK {len(preds)} OOS bars")

    except Exception as e:
        return (ticker, None, f"ERROR: {e}")


def cache_all_predictions(tickers: list[str], resume: bool = False) -> dict[str, pd.DataFrame]:
    """Train all tickers with walk-forward, cache results to disk."""
    cached: dict[str, pd.DataFrame] = {}
    n_workers = min(PIPELINE.n_workers, os.cpu_count() or 4)

    # Load existing cache if resuming
    if resume and PREDICTIONS_CACHE.exists():
        logger.info("Loading cached predictions...")
        with open(PREDICTIONS_CACHE, "rb") as f:
            cached = pickle.load(f)
        logger.info("Loaded %d cached assets", len(cached))

    # Fetch VIX data once for all tickers
    logger.info("Fetching VIX data...")
    vix_data = fetch_vix_safe()
    if vix_data is not None:
        logger.info("VIX: %d bars loaded", len(vix_data))

    remaining = [t for t in tickers if t not in cached]
    logger.info("Training %d assets (%d already cached)", len(remaining), len(cached))

    if not remaining:
        return cached

    # --- Phase 1: Batch fetch all OHLCV data ---
    t0_fetch = time.time()
    logger.info("Phase 1: Batch fetching %d tickers...", len(remaining))
    ohlcv_data = fetch_daily_batch(remaining)
    fetched = [t for t in remaining if t in ohlcv_data]
    fetch_time = time.time() - t0_fetch
    logger.info("Fetched %d/%d tickers (%.1fs)", len(fetched), len(remaining), fetch_time)

    # --- Phase 2: Parallel walk-forward training ---
    logger.info("Phase 2: Walk-forward training (%d assets, %d workers)...",
                len(fetched), n_workers)
    success, failed = 0, 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for ticker in fetched:
            future = executor.submit(
                _backtest_single_worker,
                ticker, ohlcv_data[ticker], vix_data,
            )
            futures[future] = ticker

        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                t_name, preds, msg = future.result(timeout=300)
                logger.info("[%d/%d] %s: %s", i, len(fetched), t_name, msg)

                if preds is not None:
                    cached[t_name] = preds
                    success += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error("[%d/%d] %s: TIMEOUT/ERROR: %s", i, len(fetched), ticker, e)
                failed += 1

            # Save checkpoint every 25 completions
            if i % 25 == 0:
                logger.info("--- Checkpoint: saving %d assets to disk ---", len(cached))
                with open(PREDICTIONS_CACHE, "wb") as f:
                    pickle.dump(cached, f)

    # Final save
    with open(PREDICTIONS_CACHE, "wb") as f:
        pickle.dump(cached, f)

    total_time = time.time() - t_start
    logger.info("Training complete: %d success, %d failed (fetch %.1fs + train %.1fs)",
                success, failed, fetch_time, total_time)
    logger.info("Total cached: %d assets", len(cached))

    return cached


# ---------------------------------------------------------------------------
# Per-asset stats
# ---------------------------------------------------------------------------

def compute_per_asset_stats(cached: dict[str, pd.DataFrame], cfg) -> pd.DataFrame:
    """Compute per-asset backtest metrics for a given config."""
    rows = []
    for ticker, preds in cached.items():
        signal = generate_signals_fast(preds, cfg, require_trend=True)
        trades = simulate_trades_fast(preds, signal, cfg)
        if trades.empty:
            continue

        pnl = trades["pnl_pct"]
        sized = trades["pnl_sized"]
        gross_p = sized[sized > 0].sum()
        gross_l = -sized[sized < 0].sum()

        n_years = (preds.index[-1] - preds.index[0]).days / 365.25
        rows.append({
            "ticker": ticker,
            "n_trades": len(trades),
            "trades_yr": len(trades) / max(n_years, 1),
            "win_rate": (pnl > 0).mean(),
            "avg_pnl": pnl.mean(),
            "total_sized_pnl": sized.sum(),
            "pf": gross_p / gross_l if gross_l > 0 else np.inf,
            "avg_bars": trades["bars_held"].mean(),
            "oos_years": round(n_years, 1),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("total_sized_pnl", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(results: list[dict], per_asset: pd.DataFrame) -> None:
    """Generate frontier plot, equity curves, and per-asset distribution."""
    # --- Efficient frontier ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for r in results:
        cfg = r["config"]
        dd = abs(r["max_dd"]) * 100
        cagr = r["cagr"] * 100
        color = "red" if cfg.leverage > 1 else "blue"
        marker = "o" if cfg.cash_yield == 0 else "s"
        ax.scatter(dd, cagr, c=color, marker=marker, s=100, zorder=5)
        ax.annotate(cfg.name, (dd, cagr), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)

    ax.scatter(55, 10.5, c="green", marker="*", s=200, zorder=5)
    ax.annotate("SPY B&H", (55, 10.5), textcoords="offset points",
                xytext=(8, 4), fontsize=9, fontweight="bold", color="green")
    ax.set_xlabel("Max Drawdown (%)", fontsize=12)
    ax.set_ylabel("CAGR (%)", fontsize=12)
    ax.set_title(f"Rally Detector — Universe Backtest ({len(per_asset)} assets)", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.savefig(PLOTS_DIR / "universe_frontier.png", dpi=150)
    plt.close()
    logger.info("Saved: %s", PLOTS_DIR / "universe_frontier.png")

    # --- Top equity curves ---
    top3 = sorted(results, key=lambda x: x["sharpe"], reverse=True)[:3]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    for r in top3:
        cfg = r["config"]
        if "equity_series" in r:
            eq = r["equity_series"]
            ax.plot(eq.index, eq.values, linewidth=1.2,
                    label=f"{cfg.name} (CAGR {r['cagr']:+.1%}, Sharpe {r['sharpe']:.2f})")
    ax.set_title("Rally Detector — Top Configurations by Sharpe (Universe)", fontsize=14)
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(100_000, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "universe_equity_curves.png", dpi=150)
    plt.close()
    logger.info("Saved: %s", PLOTS_DIR / "universe_equity_curves.png")

    # --- Per-asset PnL distribution ---
    if not per_asset.empty and len(per_asset) > 5:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram of profit factors
        pf_vals = per_asset["pf"].clip(upper=10)  # clip for display
        axes[0].hist(pf_vals, bins=30, edgecolor="black", alpha=0.7)
        axes[0].axvline(1.0, color="red", linestyle="--", linewidth=2, label="Breakeven (PF=1)")
        axes[0].set_xlabel("Profit Factor")
        axes[0].set_ylabel("# Assets")
        axes[0].set_title("Per-Asset Profit Factor Distribution")
        axes[0].legend()

        n_profitable = (per_asset["pf"] > 1.0).sum()
        n_total = len(per_asset)
        pct = n_profitable / n_total
        axes[0].text(
            0.95, 0.95, f"{n_profitable}/{n_total} profitable ({pct:.0%})",
            transform=axes[0].transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow"),
        )

        # Win rate distribution
        axes[1].hist(
            per_asset["win_rate"] * 100, bins=30,
            edgecolor="black", alpha=0.7, color="green",
        )
        axes[1].axvline(50, color="red", linestyle="--", linewidth=2, label="50% line")
        axes[1].set_xlabel("Win Rate (%)")
        axes[1].set_ylabel("# Assets")
        axes[1].set_title("Per-Asset Win Rate Distribution")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "universe_asset_distributions.png", dpi=150)
        plt.close()
        logger.info("Saved: %s", PLOTS_DIR / "universe_asset_distributions.png")

    # --- Top/Bottom 20 assets bar chart ---
    if not per_asset.empty and len(per_asset) > 20:
        top20 = per_asset.head(20)
        bottom20 = per_asset.tail(20)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        colors = ["green" if x > 0 else "red" for x in top20["total_sized_pnl"]]
        axes[0].barh(top20["ticker"], top20["total_sized_pnl"], color=colors)
        axes[0].set_xlabel("Total Sized PnL")
        axes[0].set_title("Top 20 Assets by Total PnL (OOS)")
        axes[0].invert_yaxis()

        colors = ["green" if x > 0 else "red" for x in bottom20["total_sized_pnl"]]
        axes[1].barh(bottom20["ticker"], bottom20["total_sized_pnl"], color=colors)
        axes[1].set_xlabel("Total Sized PnL")
        axes[1].set_title("Bottom 20 Assets by Total PnL (OOS)")
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "universe_top_bottom_assets.png", dpi=150)
        plt.close()
        logger.info("Saved: %s", PLOTS_DIR / "universe_top_bottom_assets.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Universe walk-forward backtest")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers (default: full S&P 500 + Nasdaq 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from cached predictions")
    parser.add_argument("--portfolio-only", action="store_true",
                        help="Skip training, re-run portfolio sims from cache")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: from config)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable OHLCV disk cache")
    args = parser.parse_args()

    if args.workers:
        PIPELINE.n_workers = args.workers
    if args.no_cache:
        PIPELINE.cache_enabled = False

    print("=" * 90)
    print("  RALLY DETECTOR — UNIVERSE WALK-FORWARD BACKTEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    # Get tickers
    if args.tickers:
        tickers = args.tickers
    else:
        print("\n[0/4] Fetching universe...")
        tickers = get_universe()

    print(f"  Universe: {len(tickers)} tickers")

    # Step 1: Train
    if args.portfolio_only:
        print("\n[1/4] Loading cached predictions (--portfolio-only)...")
        if not PREDICTIONS_CACHE.exists():
            print("  ERROR: No predictions cache found. Run without --portfolio-only first.")
            return
        with open(PREDICTIONS_CACHE, "rb") as f:
            cached = pickle.load(f)
        print(f"  Loaded {len(cached)} assets")
    else:
        print(f"\n[1/4] Walk-forward training ({len(tickers)} assets)...")
        cached = cache_all_predictions(tickers, resume=args.resume)

    # Step 2: Sweep configs
    print(f"\n[2/4] Portfolio simulation — {len(CONFIGS)} configurations...")
    results = []

    for cfg in CONFIGS:
        all_trades = []
        for ticker, preds in cached.items():
            signal = generate_signals_fast(preds, cfg, require_trend=True)
            trades = simulate_trades_fast(preds, signal, cfg)
            if not trades.empty:
                trades["asset"] = ticker
                all_trades.append(trades)

        if all_trades:
            portfolio_trades = pd.concat(all_trades, ignore_index=True)
        else:
            portfolio_trades = pd.DataFrame()

        metrics = simulate_portfolio(portfolio_trades, cfg)
        metrics["config"] = cfg
        results.append(metrics)

        n = metrics["n_trades"]
        tr_yr = n / metrics["n_years"] if metrics.get("n_years", 0) > 0 else 0
        print(f"    {cfg.name:<16s}  trades={n:>5d}  CAGR={metrics['cagr']:+6.1%}  "
              f"MaxDD={metrics['max_dd']:+6.1%}  Sharpe={metrics['sharpe']:.2f}  "
              f"PF={metrics['pf']:.2f}  Tr/yr={tr_yr:.0f}")

    # Step 3: Per-asset stats (using Baseline config)
    print("\n[3/4] Computing per-asset statistics (Baseline config)...")
    baseline_cfg = CONFIGS[0]
    per_asset = compute_per_asset_stats(cached, baseline_cfg)

    # Summary table
    print(f"\n{'='*110}")
    print(f"  PORTFOLIO RESULTS — {len(cached)} assets, {len(CONFIGS)} configs")
    print(f"{'='*110}")
    print(f"  {'Config':<16s} {'CAGR':>7} {'TotalRet':>9} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'PF':>6} {'WinRate':>8} {'Trades':>7} {'Tr/Yr':>6} {'Lev':>4}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: x["cagr"], reverse=True):
        cfg = r["config"]
        n_years = r.get("n_years", 1)
        tr_yr = r["n_trades"] / n_years if n_years > 0 else 0
        print(f"  {cfg.name:<16s} {r['cagr']:>+6.1%} {r['total_return']:>+8.0%} "
              f"{r['max_dd']:>+6.1%} {r['sharpe']:>7.2f} "
              f"{r['pf']:>6.2f} {r['win_rate']:>7.0%} "
              f"{r['n_trades']:>7d} {tr_yr:>5.0f}  {cfg.leverage:.0f}x")

    print(f"{'='*110}")

    # Per-asset summary
    if not per_asset.empty:
        n_profitable = (per_asset["pf"] > 1.0).sum()
        n_total = len(per_asset)
        n_no_trades = len(cached) - n_total

        print("\n  PER-ASSET SUMMARY (Baseline config):")
        print(f"  Assets with trades: {n_total}/{len(cached)}")
        print(f"  Profitable (PF>1):  {n_profitable}/{n_total} ({n_profitable/n_total:.0%})")
        print(f"  No trades:          {n_no_trades}")
        print(f"  Median PF:          {per_asset['pf'].clip(upper=20).median():.2f}")
        print(f"  Median Win Rate:    {per_asset['win_rate'].median():.1%}")
        print(f"  Median Trades/Yr:   {per_asset['trades_yr'].median():.1f}")

        print("\n  TOP 20 ASSETS:")
        print(f"  {'Ticker':<7} {'Trades':>7} {'Tr/Yr':>6} {'WinRate':>8} "
              f"{'PF':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'Years':>6}")
        print(f"  {'-'*65}")
        for _, row in per_asset.head(20).iterrows():
            print(f"  {row['ticker']:<7} {row['n_trades']:>7.0f} {row['trades_yr']:>6.1f} "
                  f"{row['win_rate']:>7.0%} {min(row['pf'], 99.9):>7.1f} "
                  f"{row['avg_pnl']:>+7.2%} {row['total_sized_pnl']:>+9.3f} "
                  f"{row['oos_years']:>5.1f}")

        print("\n  BOTTOM 20 ASSETS:")
        print(f"  {'Ticker':<7} {'Trades':>7} {'Tr/Yr':>6} {'WinRate':>8} "
              f"{'PF':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'Years':>6}")
        print(f"  {'-'*65}")
        for _, row in per_asset.tail(20).iterrows():
            print(f"  {row['ticker']:<7} {row['n_trades']:>7.0f} {row['trades_yr']:>6.1f} "
                  f"{row['win_rate']:>7.0%} {min(row['pf'], 99.9):>7.1f} "
                  f"{row['avg_pnl']:>+7.2%} {row['total_sized_pnl']:>+9.3f} "
                  f"{row['oos_years']:>5.1f}")

    # Step 4: Plots
    print("\n[4/4] Generating plots...")
    plot_results(results, per_asset)

    # Save summary to JSON
    summary = []
    for r in results:
        cfg = r["config"]
        summary.append({
            "config": cfg.name,
            "cagr": round(r["cagr"], 4),
            "max_dd": round(r["max_dd"], 4),
            "sharpe": round(r["sharpe"], 3),
            "total_return": round(r["total_return"], 4),
            "n_trades": r["n_trades"],
            "win_rate": round(r["win_rate"], 4),
            "pf": round(min(r["pf"], 999), 3),
            "leverage": cfg.leverage,
        })
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {RESULTS_FILE}")

    print(f"\n{'='*90}")
    print(f"  DONE — {len(cached)} assets backtested")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
