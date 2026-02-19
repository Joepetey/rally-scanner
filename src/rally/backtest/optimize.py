"""
Parameter optimizer — sweeps trading rules on cached model predictions.

Separates the slow step (model training) from fast step (trading rule sweep),
then simulates portfolio-level performance for each configuration including
leverage and cash yield.

Usage:
    python optimize.py
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from .common import (  # noqa: E402
    CONFIGS,
    generate_signals_fast,
    simulate_portfolio,
    simulate_trades_fast,
)
from ..config import ASSETS
from ..core.data import fetch_daily, fetch_vix_safe, merge_vix
from ..core.features import build_features
from ..core.labels import compute_labels
from ..core.model import combine_predictions, walk_forward_train

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Step 1: Cache model predictions (slow — runs once)
# ---------------------------------------------------------------------------

def cache_all_predictions() -> dict[str, pd.DataFrame]:
    """Run model training for all assets, return cached OOS predictions."""
    vix_data = fetch_vix_safe()

    cached = {}
    for name, asset in ASSETS.items():
        logger.info("Training %s...", name)
        try:
            df = fetch_daily(asset)
            if len(df) < 500:
                logger.info("  %s: SKIP (insufficient data)", name)
                continue
            if vix_data is not None:
                df = merge_vix(df, vix_data)
            df = build_features(df)
            df["RALLY_ST"] = compute_labels(df, asset)
            folds = walk_forward_train(df)
            if not folds:
                logger.info("  %s: SKIP (no folds)", name)
                continue
            preds = combine_predictions(folds)
            cached[name] = preds
            logger.info("  %s: %d OOS bars, %d folds", name, len(preds), len(folds))
        except Exception as e:
            logger.error("  %s: ERROR: %s", name, e)
    return cached


# ---------------------------------------------------------------------------
# Main: Define configurations and run sweep
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  RALLY DETECTOR — PARAMETER OPTIMIZATION")
    print("=" * 70)

    # Cache predictions (slow)
    print("\n[1/3] Training models for all assets...")
    cached = cache_all_predictions()
    print(f"       Cached {len(cached)} assets")

    # Sweep configurations (fast)
    print(f"\n[2/3] Sweeping {len(CONFIGS)} configurations...")
    results = []

    for cfg in CONFIGS:
        all_trades = []
        for name, preds in cached.items():
            asset = ASSETS[name]
            require_trend = asset.asset_class == "equity"
            signal = generate_signals_fast(preds, cfg, require_trend)
            trades = simulate_trades_fast(preds, signal, cfg)
            if not trades.empty:
                trades["asset"] = name
                all_trades.append(trades)

        if all_trades:
            portfolio_trades = pd.concat(all_trades, ignore_index=True)
        else:
            portfolio_trades = pd.DataFrame()

        metrics = simulate_portfolio(portfolio_trades, cfg)
        metrics["config"] = cfg
        results.append(metrics)

        n = metrics["n_trades"]
        tr_yr = n / metrics["n_years"] if metrics["n_years"] > 0 else 0
        print(f"    {cfg.name:<16s}  trades={n:>4d}  CAGR={metrics['cagr']:+6.1%}  "
              f"MaxDD={metrics['max_dd']:+6.1%}  Sharpe={metrics['sharpe']:.2f}  "
              f"PF={metrics['pf']:.2f}  Tr/yr={tr_yr:.0f}")

    # Summary table
    print("\n[3/3] Results")
    print(f"\n{'='*110}")
    print(f"  {'Config':<16s} {'CAGR':>7} {'TotalRet':>9} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'PF':>6} {'WinRate':>8} {'Trades':>7} {'Tr/Yr':>6} {'Leverage':>4}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: x["cagr"], reverse=True):
        cfg = r["config"]
        n_years = r["n_years"]
        tr_yr = r["n_trades"] / n_years if n_years > 0 else 0
        print(f"  {cfg.name:<16s} {r['cagr']:>+6.1%} {r['total_return']:>+8.0%} "
              f"{r['max_dd']:>+6.1%} {r['sharpe']:>7.2f} "
              f"{r['pf']:>6.2f} {r['win_rate']:>7.0%} "
              f"{r['n_trades']:>7d} {tr_yr:>5.0f}  {cfg.leverage:.0f}x")

    print(f"{'='*110}")

    # Efficient frontier plot
    print("\n  Plotting efficient frontier...")
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

    # Add SPY buy-and-hold reference
    ax.scatter(55, 10.5, c="green", marker="*", s=200, zorder=5)
    ax.annotate("SPY B&H", (55, 10.5), textcoords="offset points",
                xytext=(8, 4), fontsize=9, fontweight="bold", color="green")

    ax.set_xlabel("Max Drawdown (%)", fontsize=12)
    ax.set_ylabel("CAGR (%)", fontsize=12)
    ax.set_title("Rally Detector — Return vs Drawdown Frontier", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(["Unleveraged (blue)", "Leveraged (red)", "SPY B&H (green)"],
              loc="lower right")

    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.savefig(PLOTS_DIR / "frontier.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'frontier.png'}")

    # Best equity curves
    print("\n  Plotting top equity curves...")
    top3 = sorted(results, key=lambda x: x["sharpe"], reverse=True)[:3]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    for r in top3:
        cfg = r["config"]
        if "equity_series" in r:
            eq = r["equity_series"]
            ax.plot(eq.index, eq.values, linewidth=1.2, label=f"{cfg.name} (CAGR {r['cagr']:+.1%})")

    ax.set_title("Rally Detector — Top Configurations by Sharpe", fontsize=14)
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(100_000, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_equity_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'top_equity_curves.png'}")


if __name__ == "__main__":
    main()
