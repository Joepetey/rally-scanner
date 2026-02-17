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
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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

from .calibrate import calibrate_thresholds  # noqa: E402
from .config import PIPELINE
from .data import fetch_daily_batch, fetch_vix, merge_vix
from .features import build_features
from .labels import compute_labels
from .model import combine_predictions, walk_forward_train
from .universe import get_universe

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
CACHE_DIR = PROJECT_ROOT / "backtest_cache"
CACHE_DIR.mkdir(exist_ok=True)
PREDICTIONS_CACHE = CACHE_DIR / "predictions.pkl"
RESULTS_FILE = CACHE_DIR / "results.json"


# ---------------------------------------------------------------------------
# Configs (same as optimize.py)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    name: str
    p_rally: float
    comp_score: float
    max_risk: float
    vol_k: float
    profit_atr: float
    time_stop: int
    leverage: float
    cash_yield: float


CONFIGS = [
    Config("Baseline", p_rally=0.50, comp_score=0.55, max_risk=0.25,
           vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=1.0, cash_yield=0.0),
    Config("Base+Cash4%", p_rally=0.50, comp_score=0.55, max_risk=0.25,
           vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=1.0, cash_yield=0.04),
    Config("Aggressive", p_rally=0.40, comp_score=0.45, max_risk=0.30,
           vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=1.0, cash_yield=0.0),
    Config("Aggr+Cash4%", p_rally=0.40, comp_score=0.45, max_risk=0.30,
           vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=1.0, cash_yield=0.04),
    Config("Concentrated", p_rally=0.55, comp_score=0.60, max_risk=0.40,
           vol_k=0.15, profit_atr=2.5, time_stop=12, leverage=1.0, cash_yield=0.0),
    Config("Base 2x Lev", p_rally=0.50, comp_score=0.55, max_risk=0.25,
           vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=2.0, cash_yield=0.04),
    Config("Base 3x Lev", p_rally=0.50, comp_score=0.55, max_risk=0.25,
           vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=3.0, cash_yield=0.04),
    Config("Aggr 2x Lev", p_rally=0.40, comp_score=0.45, max_risk=0.30,
           vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=2.0, cash_yield=0.04),
    Config("Aggr 3x Lev", p_rally=0.40, comp_score=0.45, max_risk=0.30,
           vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=3.0, cash_yield=0.04),
    Config("Conservative", p_rally=0.55, comp_score=0.60, max_risk=0.15,
           vol_k=0.08, profit_atr=2.0, time_stop=8, leverage=1.0, cash_yield=0.05),
    Config("Cons 2x Lev", p_rally=0.55, comp_score=0.60, max_risk=0.15,
           vol_k=0.08, profit_atr=2.0, time_stop=8, leverage=2.0, cash_yield=0.05),
]


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
        print("  Loading cached predictions...")
        with open(PREDICTIONS_CACHE, "rb") as f:
            cached = pickle.load(f)
        print(f"  Loaded {len(cached)} cached assets")

    # Fetch VIX data once for all tickers
    print("  Fetching VIX data...")
    try:
        vix_data = fetch_vix()
        print(f"  VIX: {len(vix_data)} bars loaded")
    except Exception as e:
        print(f"  WARNING: Could not fetch VIX: {e}")
        vix_data = None

    remaining = [t for t in tickers if t not in cached]
    print(f"  Training {len(remaining)} assets ({len(cached)} already cached)")

    if not remaining:
        return cached

    # --- Phase 1: Batch fetch all OHLCV data ---
    t0_fetch = time.time()
    print(f"\n  Phase 1: Batch fetching {len(remaining)} tickers...")
    ohlcv_data = fetch_daily_batch(remaining)
    fetched = [t for t in remaining if t in ohlcv_data]
    fetch_time = time.time() - t0_fetch
    print(f"  Fetched {len(fetched)}/{len(remaining)} tickers ({fetch_time:.1f}s)")

    # --- Phase 2: Parallel walk-forward training ---
    print(f"\n  Phase 2: Walk-forward training ({len(fetched)} assets, {n_workers} workers)...")
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
                print(f"  [{i}/{len(fetched)}] {t_name}: {msg}")

                if preds is not None:
                    cached[t_name] = preds
                    success += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"  [{i}/{len(fetched)}] {ticker}: TIMEOUT/ERROR: {e}")
                failed += 1

            # Save checkpoint every 25 completions
            if i % 25 == 0:
                print(f"  --- Checkpoint: saving {len(cached)} assets to disk ---")
                with open(PREDICTIONS_CACHE, "wb") as f:
                    pickle.dump(cached, f)

    # Final save
    with open(PREDICTIONS_CACHE, "wb") as f:
        pickle.dump(cached, f)

    total_time = time.time() - t_start
    print(f"\n  Training complete: {success} success, {failed} failed "
          f"(fetch {fetch_time:.1f}s + train {total_time:.1f}s)")
    print(f"  Total cached: {len(cached)} assets")

    return cached


# ---------------------------------------------------------------------------
# Step 2: Fast signal generation + trade simulation
# ---------------------------------------------------------------------------

def generate_signals_fast(preds: pd.DataFrame, cfg: Config,
                          require_trend: bool) -> pd.Series:
    signal = (
        (preds["P_RALLY"] > cfg.p_rally)
        & (preds["COMP_SCORE"] > cfg.comp_score)
    )
    if require_trend:
        signal = signal & (preds["Trend"] == 1.0)
    return signal


def simulate_trades_fast(preds: pd.DataFrame, signal: pd.Series,
                         cfg: Config) -> pd.DataFrame:
    """Simulate trades with config-specific parameters."""
    n = len(preds)
    close = preds["Close"].values
    high = preds["High"].values
    low = preds["Low"].values
    atr = preds["ATR"].values
    rv_pct = preds["p_RV"].values if "p_RV" in preds.columns else np.full(n, 0.5)

    trades = []
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    stop_price = 0.0
    trailing_stop = 0.0
    size = 0.0
    bars_held = 0
    highest_close = 0.0

    for i in range(n):
        if in_trade:
            bars_held += 1
            if close[i] > highest_close:
                highest_close = close[i]
                new_trail = highest_close - 1.5 * atr[entry_idx]
                trailing_stop = max(trailing_stop, new_trail)

            exit_reason = None
            exit_price = close[i]

            if low[i] <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
            elif high[i] >= entry_price + cfg.profit_atr * atr[entry_idx]:
                exit_price = entry_price + cfg.profit_atr * atr[entry_idx]
                exit_reason = "profit_target"
            elif bars_held >= 2 and close[i] < trailing_stop:
                exit_reason = "trail_stop"
            elif bars_held >= cfg.time_stop:
                exit_reason = "time_stop"
            elif rv_pct[i] > 0.80 and close[i] < close[i - 1]:
                exit_reason = "vol_exhaustion"

            if exit_reason:
                pnl_pct = exit_price / entry_price - 1
                trades.append({
                    "entry_date": preds.index[entry_idx],
                    "exit_date": preds.index[i],
                    "pnl_pct": pnl_pct,
                    "pnl_sized": pnl_pct * size,
                    "size": size,
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                    "asset": "",  # filled later
                })
                in_trade = False

        if not in_trade and signal.iloc[i]:
            entry_idx = i
            entry_price = close[i]
            stop_price = preds["RangeLow"].iloc[i]
            if np.isnan(stop_price) or stop_price >= entry_price:
                stop_price = entry_price * 0.97

            p_rally = preds["P_RALLY"].iloc[i]
            atr_pct = preds["ATR_pct"].iloc[i]
            raw_size = cfg.vol_k * (p_rally - cfg.p_rally) / atr_pct if atr_pct > 0 else 0
            size = max(0.0, min(raw_size, cfg.max_risk))
            if size >= 0.01:
                in_trade = True
                bars_held = 0
                highest_close = entry_price
                trailing_stop = entry_price - 1.5 * atr[i]

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 3: Portfolio simulation
# ---------------------------------------------------------------------------

def simulate_portfolio(all_trades: pd.DataFrame, cfg: Config,
                       initial_capital: float = 100_000) -> dict:
    """Simulate portfolio with leverage and cash yield on idle capital."""
    if all_trades.empty:
        return {"cagr": 0, "max_dd": 0, "sharpe": 0, "total_return": 0,
                "n_trades": 0, "win_rate": 0, "pf": 0, "n_years": 0}

    trades = all_trades.sort_values("entry_date").reset_index(drop=True)
    min_date = trades["entry_date"].min()
    max_date = trades["exit_date"].max()
    dates = pd.date_range(min_date, max_date, freq="B")

    equity = initial_capital
    equity_series = pd.Series(np.nan, index=dates)
    open_positions = []
    daily_pnl = pd.Series(0.0, index=dates)
    trade_queue = list(trades.itertuples(index=False))
    trade_idx = 0
    daily_cash_yield = (1 + cfg.cash_yield) ** (1 / 252) - 1

    for date in dates:
        # Close positions
        still_open = []
        for pos in open_positions:
            if date >= pos["exit_date"]:
                sized_pnl = pos["pnl_pct"] * pos["allocated"] * cfg.leverage
                equity += sized_pnl
                daily_pnl[date] += sized_pnl
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open new positions
        total_allocated = sum(p["allocated"] for p in open_positions)
        exposure = total_allocated / equity if equity > 0 else 0
        while trade_idx < len(trade_queue):
            t = trade_queue[trade_idx]
            if t.entry_date > date:
                break
            if t.entry_date == date:
                alloc = t.size * equity
                new_exposure = exposure + alloc / equity
                if new_exposure <= 1.0:
                    open_positions.append({
                        "entry_date": t.entry_date,
                        "exit_date": t.exit_date,
                        "pnl_pct": t.pnl_pct,
                        "allocated": alloc,
                    })
                    exposure = new_exposure
            trade_idx += 1

        # Cash yield on idle capital
        idle = max(0, equity - total_allocated)
        cash_income = idle * daily_cash_yield
        equity += cash_income
        daily_pnl[date] += cash_income

        equity_series[date] = equity

    equity_series = equity_series.ffill().bfill()
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak

    total_return = equity_series.iloc[-1] / initial_capital - 1
    total_days = (dates[-1] - dates[0]).days
    n_years = total_days / 365.25
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    max_dd = drawdown.min()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    sized_pnl = trades["pnl_sized"] * cfg.leverage
    gross_profit = sized_pnl[sized_pnl > 0].sum()
    gross_loss = -sized_pnl[sized_pnl < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_return": total_return,
        "n_trades": len(trades),
        "n_years": n_years,
        "win_rate": (trades["pnl_pct"] > 0).mean(),
        "pf": pf,
        "equity_series": equity_series,
        "drawdown": drawdown,
    }


# ---------------------------------------------------------------------------
# Step 4: Per-asset stats
# ---------------------------------------------------------------------------

def compute_per_asset_stats(cached: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
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
# Step 5: Plots
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
    print(f"  Saved: {PLOTS_DIR / 'universe_frontier.png'}")

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
    print(f"  Saved: {PLOTS_DIR / 'universe_equity_curves.png'}")

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
        print(f"  Saved: {PLOTS_DIR / 'universe_asset_distributions.png'}")

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
        print(f"  Saved: {PLOTS_DIR / 'universe_top_bottom_assets.png'}")


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
