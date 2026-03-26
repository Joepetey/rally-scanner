#!/usr/bin/env python
"""Experiment: test each win-rate filter against the Baseline independently.

Run from repo root:
    .venv/bin/python scripts/experiment_filters.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backtest.common import generate_signals_fast, simulate_portfolio, simulate_trades_fast
from config import CONFIGS, CONFIGS_BY_NAME, TradingConfig

CACHE = ROOT / "backtest_cache" / "predictions.pkl"

BASE = CONFIGS_BY_NAME["baseline"]

HIGH_THRESH = TradingConfig(
    name="HighThresh",
    p_rally=0.62, comp_score=0.65,
    max_risk=BASE.max_risk, vol_k=BASE.vol_k,
    profit_atr=BASE.profit_atr, time_stop=BASE.time_stop,
    leverage=BASE.leverage, cash_yield=BASE.cash_yield,
)


def run(cached: dict, cfg: TradingConfig, label: str, **kwargs) -> dict:
    all_trades = []
    for ticker, preds in cached.items():
        signal = generate_signals_fast(preds, cfg, require_trend=True)
        trades = simulate_trades_fast(preds, signal, cfg, **kwargs)
        if not trades.empty:
            trades["asset"] = ticker
            all_trades.append(trades)

    portfolio_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    m = simulate_portfolio(portfolio_trades, cfg)
    n_years = m.get("n_years", 1)
    tr_yr = m["n_trades"] / n_years if n_years > 0 else 0
    print(
        f"  {label:<32s}  trades={m['n_trades']:>6d} ({tr_yr:>4.0f}/yr)"
        f"  WinRate={m['win_rate']:.1%}"
        f"  CAGR={m['cagr']:+7.1%}"
        f"  MaxDD={m['max_dd']:+6.1%}"
        f"  Sharpe={m['sharpe']:.2f}"
        f"  PF={m['pf']:.2f}"
    )
    return m


def build_spy_trend(cached: dict) -> pd.Series:
    """Build date→bool from SPY's own predictions if present, else fetch via yfinance."""
    if "SPY" in cached:
        spy = cached["SPY"]
        return (spy["Close"] > spy["MA200"]).rename("spy_trend")

    try:
        import yfinance as yf
        spy_df = yf.download("SPY", start="2000-01-01", progress=False, auto_adjust=True)
        spy_df.columns = spy_df.columns.get_level_values(0) if isinstance(spy_df.columns, pd.MultiIndex) else spy_df.columns
        close = spy_df["Close"]
        ma200 = close.rolling(200, min_periods=200).mean()
        trend = (close > ma200).rename("spy_trend")
        trend.index = trend.index.tz_localize(None)
        return trend
    except Exception as e:
        print(f"  [warn] Could not fetch SPY: {e} — SPY filter will be skipped")
        return pd.Series(dtype=bool)


def main():
    print(f"\nLoading predictions cache: {CACHE}")
    with open(CACHE, "rb") as f:
        cached = pickle.load(f)
    print(f"Loaded {len(cached)} assets\n")

    header = f"  {'Config':<20s}  {'Variant':<18s}  {'Trades':>12s}  {'WinRate':>8s}  {'CAGR':>8s}  {'MaxDD':>7s}  {'Sharpe':>7s}  {'PF':>5s}"
    sep = "  " + "-" * (len(header) - 2)

    # High-threshold variants for each config
    combined_configs = []
    for cfg in CONFIGS:
        combined_configs.append(TradingConfig(
            name=cfg.name,
            p_rally=cfg.p_rally + 0.12,
            comp_score=cfg.comp_score + 0.10,
            max_risk=cfg.max_risk,
            vol_k=cfg.vol_k,
            profit_atr=cfg.profit_atr,
            time_stop=cfg.time_stop,
            leverage=cfg.leverage,
            cash_yield=cfg.cash_yield,
        ))

    print(header)
    print(sep)

    all_results = []
    for cfg, hi_cfg in zip(CONFIGS, combined_configs):
        base_m  = run(cached, cfg,    f"{cfg.name:<20s}  {'baseline':<18s}")
        fail_m  = run(cached, cfg,    f"{cfg.name:<20s}  {'+ FAIL_DN gate':<18s}", require_fail_dn=True)
        hi_m    = run(cached, hi_cfg, f"{cfg.name:<20s}  {'+ hi thresh':<18s}",   require_fail_dn=True)
        all_results.append((cfg.name, base_m, fail_m, hi_m))
        print(sep)

    print("\nDelta: combined (hi thresh + FAIL_DN) vs baseline per config:")
    print(f"  {'Config':<20s}  {'ΔTrades':>8s}  {'ΔWinRate':>9s}  {'ΔCAGR':>8s}  {'ΔMaxDD':>8s}  {'ΔSharpe':>8s}")
    print("  " + "-" * 72)
    for name, base_m, _, hi_m in all_results:
        print(
            f"  {name:<20s}"
            f"  {hi_m['n_trades'] - base_m['n_trades']:>+8d}"
            f"  {hi_m['win_rate'] - base_m['win_rate']:>+8.1%}"
            f"  {hi_m['cagr'] - base_m['cagr']:>+8.1%}"
            f"  {hi_m['max_dd'] - base_m['max_dd']:>+8.1%}"
            f"  {hi_m['sharpe'] - base_m['sharpe']:>+8.2f}"
        )
    print()


if __name__ == "__main__":
    main()
