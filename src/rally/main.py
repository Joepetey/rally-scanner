"""
Rally Phase-Transition Detector — Main runner.

Usage:
    python main.py [--asset SPY] [--require-trend] [--plot]
    python main.py --run-all [--plot]
"""

import argparse
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from .config import ASSETS, PARAMS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
from .data import fetch_daily, fetch_vix, merge_vix
from .features import build_features, FEATURE_COLS
from .labels import compute_labels
from .model import walk_forward_train, combine_predictions
from .trading import generate_signals, simulate_trades
from .backtest import (compute_equity_curve, print_report, compute_metrics,
                      performance_by_year)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run(asset_name: str, require_trend: bool = False, plot: bool = False, verbose: bool = True) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    asset = ASSETS[asset_name]
    _p = print if verbose else (lambda *a, **k: None)

    _p(f"[1/6] Fetching daily data for {asset.ticker} ...")
    try:
        df = fetch_daily(asset)
    except Exception as e:
        _p(f"       ERROR fetching {asset.ticker}: {e}")
        return None, None, {"error": str(e)}
    _p(f"       {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    if len(df) < 500:
        _p(f"       SKIP: Not enough data ({len(df)} bars, need 500+)")
        return None, None, {"error": "insufficient data"}

    _p(f"[2/6] Building features ...")
    try:
        vix_data = fetch_vix()
        df = merge_vix(df, vix_data)
    except Exception as e:
        _p(f"       WARNING: VIX data unavailable: {e}")
    df = build_features(df)

    _p(f"[3/6] Computing labels (H={PARAMS.forward_horizon}, "
       f"r_up={asset.r_up:.1%}, d_dn={asset.d_dn:.1%}) ...")
    df["RALLY_ST"] = compute_labels(df, asset)

    valid = df["RALLY_ST"].dropna()
    pos_rate = valid.mean()
    _p(f"       Label balance: {pos_rate:.1%} positive  "
       f"({int(valid.sum())}/{len(valid)} bars)")

    _p(f"[4/6] Walk-forward training (with HMM regime features) ...")
    folds = walk_forward_train(df)
    if not folds:
        _p("       ERROR: No folds produced. Need more data.")
        return None, None, {"error": "no folds"}
    _p(f"       {len(folds)} folds")
    for f in folds:
        _p(f"       train {f.train_start}–{f.train_end}  "
           f"test {f.test_start}–{f.test_end}")
        if verbose:
            _p(f"         coefs: { {k: round(v, 4) for k, v in f.coefs.items()} }")

    preds = combine_predictions(folds)
    _p(f"       {len(preds)} out-of-sample bars")

    _p(f"[5/6] Generating signals & simulating trades ...")
    use_trend = require_trend or asset.asset_class == "equity"
    signal = generate_signals(preds, require_trend=use_trend)
    _p(f"       Signal fires on {signal.sum()} bars "
       f"({signal.mean():.1%} of OOS bars)")

    trades = simulate_trades(preds, signal)
    _p(f"       {len(trades)} completed trades")

    _p(f"[6/6] Computing metrics ...")
    equity = compute_equity_curve(trades)
    metrics = print_report(trades, equity, asset_name) if verbose else compute_metrics(trades, equity)

    if verbose:
        print("--- SANITY CHECKS ---")
        n_trades = len(trades)
        if n_trades > 0:
            oos_bars = len(preds)
            trade_freq = n_trades / oos_bars * 252 if oos_bars > 0 else 0
            print(f"  Trades/year (approx): {trade_freq:.1f}")
            if trade_freq > 50:
                print("  WARNING: Trading too frequently")
            print("  Expect: rare trades, 40-55% win rate, winners >> losers")
        else:
            print("  No trades generated")

    if plot and not equity.empty:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(equity.index, equity["equity"], linewidth=1)
        axes[0].set_title(f"Rally Detector Equity Curve — {asset_name}")
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(True, alpha=0.3)

        axes[1].fill_between(equity.index, equity["drawdown"] * 100, 0,
                             alpha=0.4, color="red")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        PLOTS_DIR.mkdir(exist_ok=True)
        fname = PLOTS_DIR / f"equity_{asset_name}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        _p(f"  Plot saved: {fname}")

    return trades, equity, metrics


def run_all(plot: bool = False) -> dict:
    """Run backtest across all assets and print a summary table."""
    results = {}
    all_trades = []

    for name in ASSETS:
        print(f"\n{'#'*60}")
        print(f"  {name}")
        print(f"{'#'*60}")
        trades, equity, metrics = run(name, plot=plot, verbose=True)
        results[name] = metrics
        if trades is not None and not trades.empty:
            t = trades.copy()
            t["asset"] = name
            all_trades.append(t)

    # Per-asset summary
    print(f"\n{'='*90}")
    print(f"  CROSS-ASSET SUMMARY (per-asset)")
    print(f"{'='*90}")
    header = f"{'Asset':<8} {'Trades':>7} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>8} {'PF':>7} {'CAGR':>7} {'MaxDD':>7} {'Tr/Yr':>6}"
    print(header)
    print("-" * 90)

    for name, m in results.items():
        if "error" in m:
            print(f"{name:<8} {'ERROR':>7}  {m.get('error', '')}")
            continue
        n = m.get("total_trades", 0)
        if n == 0:
            print(f"{name:<8} {'0':>7}  no trades")
            continue
        print(f"{name:<8} {n:>7} {m['win_rate']:>7.0%} "
              f"{m['avg_win_pct']:>+7.2f}% {m['avg_loss_pct']:>+7.2f}% "
              f"{m['profit_factor']:>7.2f} {m['cagr']:>+6.2f}% "
              f"{m['max_drawdown']:>6.2f}%  "
              f"{n / 5:.1f}")

    print(f"{'='*90}")

    # Portfolio-level backtest
    if all_trades:
        portfolio_trades = pd.concat(all_trades, ignore_index=True)
        _run_portfolio(portfolio_trades, plot=plot)

    return results


def _run_portfolio(trades: pd.DataFrame, initial_capital: float = 100_000,
                   max_total_exposure: float = 1.0, plot: bool = False) -> None:
    """
    Portfolio-level backtest: all 14 assets share one equity pool.
    Sizes each position from current equity, caps total exposure.
    """
    trades = trades.sort_values("entry_date").reset_index(drop=True)

    # Build daily timeline
    min_date = trades["entry_date"].min()
    max_date = trades["exit_date"].max()
    dates = pd.date_range(min_date, max_date, freq="B")

    equity = initial_capital
    equity_series = pd.Series(np.nan, index=dates)
    open_positions = []  # list of dicts

    daily_pnl = pd.Series(0.0, index=dates)
    trade_queue = list(trades.itertuples(index=False))
    trade_idx = 0

    for date in dates:
        # Close positions that exit today
        still_open = []
        for pos in open_positions:
            if date >= pos["exit_date"]:
                # Realized PnL
                sized_pnl = pos["pnl_pct"] * pos["allocated"]
                equity += sized_pnl
                daily_pnl[date] += sized_pnl
            else:
                # Mark-to-market (approximate: distribute PnL linearly)
                still_open.append(pos)
        open_positions = still_open

        # Open new positions
        current_exposure = sum(p["allocated"] for p in open_positions) / equity if equity > 0 else 0
        while trade_idx < len(trade_queue):
            t = trade_queue[trade_idx]
            if t.entry_date > date:
                break
            if t.entry_date == date:
                # Size from current equity
                alloc = t.size * equity
                new_exposure = current_exposure + alloc / equity
                if new_exposure <= max_total_exposure:
                    open_positions.append({
                        "asset": t.asset,
                        "entry_date": t.entry_date,
                        "exit_date": t.exit_date,
                        "pnl_pct": t.pnl_pct,
                        "allocated": alloc,
                        "size": t.size,
                    })
                    current_exposure = new_exposure
            trade_idx += 1

        equity_series[date] = equity

    # Forward-fill any NaN
    equity_series = equity_series.ffill().bfill()
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak

    equity_df = pd.DataFrame({"equity": equity_series, "drawdown": drawdown})

    # Metrics
    total_return = equity_series.iloc[-1] / initial_capital - 1
    total_days = (dates[-1] - dates[0]).days
    cagr = (1 + total_return) ** (365.25 / max(total_days, 1)) - 1
    max_dd = drawdown.min()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    n_trades = len(trades)
    sized_pnl = trades["pnl_sized"]
    gross_profit = sized_pnl[sized_pnl > 0].sum()
    gross_loss = -sized_pnl[sized_pnl < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    n_years = total_days / 365.25

    print(f"\n{'='*90}")
    print(f"  PORTFOLIO BACKTEST — ALL {len(trades['asset'].unique())} ASSETS COMBINED")
    print(f"{'='*90}")
    print(f"  Initial capital:   ${initial_capital:,.0f}")
    print(f"  Final equity:      ${equity_series.iloc[-1]:,.0f}")
    print(f"  Total return:      {total_return:+.1%}")
    print(f"  CAGR:              {cagr:+.2%}")
    print(f"  Max drawdown:      {max_dd:.2%}")
    print(f"  Sharpe ratio:      {sharpe:.2f}")
    print(f"  Profit factor:     {pf:.2f}")
    print(f"  Total trades:      {n_trades}")
    print(f"  Trades/year:       {n_trades / n_years:.1f}")
    print(f"  Win rate:          {(trades['pnl_pct'] > 0).mean():.1%}")
    print(f"  Avg win:           {trades.loc[trades['pnl_pct'] > 0, 'pnl_pct'].mean():+.2%}")
    print(f"  Avg loss:          {trades.loc[trades['pnl_pct'] <= 0, 'pnl_pct'].mean():+.2%}")
    print(f"  Period:            {dates[0].date()} → {dates[-1].date()} ({n_years:.1f} years)")

    # Per-asset contribution
    print(f"\n  --- Contribution by asset ---")
    asset_contrib = trades.groupby("asset").agg(
        trades=("pnl_pct", "count"),
        total_sized_pnl=("pnl_sized", "sum"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean()),
    ).round(4)
    asset_contrib = asset_contrib.sort_values("total_sized_pnl", ascending=False)
    print(asset_contrib.to_string())

    # Per-year
    print(f"\n  --- Portfolio performance by YEAR ---")
    yearly = performance_by_year(trades)
    if not yearly.empty:
        print(yearly.to_string())

    print(f"{'='*90}")

    # Plot
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1]})

        axes[0].plot(equity_series.index, equity_series.values, linewidth=1.2, color="navy")
        axes[0].set_title("Rally Detector — 14-Asset Portfolio Equity Curve", fontsize=14)
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(initial_capital, color="gray", linestyle="--", alpha=0.5)

        axes[1].fill_between(drawdown.index, drawdown.values * 100, 0,
                             alpha=0.5, color="red")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)

        # Trade scatter
        for _, t in trades.iterrows():
            color = "green" if t["pnl_pct"] > 0 else "red"
            axes[2].scatter(t["entry_date"], t["pnl_pct"] * 100,
                           c=color, s=15, alpha=0.6)
        axes[2].axhline(0, color="gray", linewidth=0.5)
        axes[2].set_ylabel("Trade PnL (%)")
        axes[2].set_xlabel("Date")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        PLOTS_DIR.mkdir(exist_ok=True)
        plt.savefig(PLOTS_DIR / "equity_PORTFOLIO.png", dpi=150)
        plt.close()
        print(f"  Plot saved: {PLOTS_DIR / 'equity_PORTFOLIO.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rally Phase-Transition Detector")
    parser.add_argument("--asset", default="SPY", choices=list(ASSETS.keys()))
    parser.add_argument("--require-trend", action="store_true",
                        help="Require Trend==1 for entry (auto-enabled for equities)")
    parser.add_argument("--plot", action="store_true", help="Save equity curve plots")
    parser.add_argument("--run-all", action="store_true",
                        help="Run backtest across all assets")
    args = parser.parse_args()

    if args.run_all:
        run_all(plot=args.plot)
    else:
        run(args.asset, require_trend=args.require_trend, plot=args.plot)


if __name__ == "__main__":
    main()
