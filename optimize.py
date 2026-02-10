"""
Parameter optimizer — sweeps trading rules on cached model predictions.

Separates the slow step (model training) from fast step (trading rule sweep),
then simulates portfolio-level performance for each configuration including
leverage and cash yield.

Usage:
    python optimize.py
"""

import warnings
import itertools
from dataclasses import dataclass
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ASSETS, PARAMS
from data import fetch_daily, fetch_vix, merge_vix
from features import build_features
from labels import compute_labels
from model import walk_forward_train, combine_predictions
from trading import simulate_trades

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Step 1: Cache model predictions (slow — runs once)
# ---------------------------------------------------------------------------

def cache_all_predictions() -> Dict[str, pd.DataFrame]:
    """Run model training for all assets, return cached OOS predictions."""
    # Fetch VIX once
    try:
        vix_data = fetch_vix()
    except Exception:
        vix_data = None

    cached = {}
    for name, asset in ASSETS.items():
        print(f"  Training {name}...", end="", flush=True)
        try:
            df = fetch_daily(asset)
            if len(df) < 500:
                print(" SKIP (insufficient data)")
                continue
            if vix_data is not None:
                df = merge_vix(df, vix_data)
            df = build_features(df)
            df["RALLY_ST"] = compute_labels(df, asset)
            folds = walk_forward_train(df)
            if not folds:
                print(" SKIP (no folds)")
                continue
            preds = combine_predictions(folds)
            cached[name] = preds
            print(f" {len(preds)} OOS bars, {len(folds)} folds")
        except Exception as e:
            print(f" ERROR: {e}")
    return cached


# ---------------------------------------------------------------------------
# Step 2: Fast signal + trade sweep (runs many times)
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
    cash_yield: float  # annual yield on idle capital


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
            raw_size = cfg.vol_k * (p_rally - 0.5) / atr_pct if atr_pct > 0 else 0
            size = max(0.0, min(raw_size, cfg.max_risk))
            if size > 0:
                in_trade = True
                bars_held = 0
                highest_close = entry_price
                trailing_stop = entry_price - 1.5 * atr[i]

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 3: Portfolio simulation with leverage + cash yield
# ---------------------------------------------------------------------------

def simulate_portfolio(all_trades: pd.DataFrame, cfg: Config,
                       initial_capital: float = 100_000) -> dict:
    """Simulate portfolio with leverage and cash yield on idle capital."""
    if all_trades.empty:
        return {"cagr": 0, "max_dd": 0, "sharpe": 0, "total_return": 0,
                "n_trades": 0, "win_rate": 0, "pf": 0}

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
# Step 4: Define configurations and run sweep
# ---------------------------------------------------------------------------

CONFIGS = [
    # --- Baseline (current) ---
    Config("Baseline", p_rally=0.50, comp_score=0.55, max_risk=0.25,
           vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=1.0, cash_yield=0.0),

    # --- Baseline + cash yield ---
    Config("Base+Cash4%", p_rally=0.50, comp_score=0.55, max_risk=0.25,
           vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=1.0, cash_yield=0.04),

    # --- Aggressive: lower thresholds, longer holds ---
    Config("Aggressive", p_rally=0.40, comp_score=0.45, max_risk=0.30,
           vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=1.0, cash_yield=0.0),

    Config("Aggr+Cash4%", p_rally=0.40, comp_score=0.45, max_risk=0.30,
           vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=1.0, cash_yield=0.04),

    # --- Concentrated: fewer trades, bigger bets ---
    Config("Concentrated", p_rally=0.55, comp_score=0.60, max_risk=0.40,
           vol_k=0.15, profit_atr=2.5, time_stop=12, leverage=1.0, cash_yield=0.0),

    # --- Leveraged variants ---
    Config("Base 2x Lev", p_rally=0.50, comp_score=0.55, max_risk=0.25,
           vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=2.0, cash_yield=0.04),

    Config("Base 3x Lev", p_rally=0.50, comp_score=0.55, max_risk=0.25,
           vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=3.0, cash_yield=0.04),

    Config("Aggr 2x Lev", p_rally=0.40, comp_score=0.45, max_risk=0.30,
           vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=2.0, cash_yield=0.04),

    # --- Max return: aggressive + 3x ---
    Config("Aggr 3x Lev", p_rally=0.40, comp_score=0.45, max_risk=0.30,
           vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=3.0, cash_yield=0.04),

    # --- Conservative: minimize drawdown ---
    Config("Conservative", p_rally=0.55, comp_score=0.60, max_risk=0.15,
           vol_k=0.08, profit_atr=2.0, time_stop=8, leverage=1.0, cash_yield=0.05),

    Config("Cons 2x Lev", p_rally=0.55, comp_score=0.60, max_risk=0.15,
           vol_k=0.08, profit_atr=2.0, time_stop=8, leverage=2.0, cash_yield=0.05),
]


def main():
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
    print(f"\n[3/3] Results")
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
    plt.savefig("frontier.png", dpi=150)
    plt.close()
    print("  Saved: frontier.png")

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
    plt.savefig("top_equity_curves.png", dpi=150)
    plt.close()
    print("  Saved: top_equity_curves.png")


if __name__ == "__main__":
    main()
