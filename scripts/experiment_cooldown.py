#!/usr/bin/env python
"""Experiment: does a cooldown period after closing a position improve returns?

Sweeps cooldown_days=[0, 3, 5, 10] across all configs and compares
CAGR, Sharpe, MaxDD, Win Rate, Avg Trade Duration, and Trade Count.

Also breaks down the impact by asset class (crypto vs equities).

Run from repo root:
    .venv/bin/python scripts/experiment_cooldown.py
    .venv/bin/python scripts/experiment_cooldown.py --tickers AAPL MSFT BTC-USD
    .venv/bin/python scripts/experiment_cooldown.py --cooldowns 0 3 5 10 15 20
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rally_ml.config import CONFIGS, CONFIGS_BY_NAME, TradingConfig  # noqa: E402
from sweep_common import filter_tickers, run_backtest  # noqa: E402

CACHE = ROOT / "backtest_cache" / "predictions.pkl"

CRYPTO_SUFFIXES = ("-USD",)


def is_crypto(ticker: str) -> bool:
    return any(ticker.endswith(s) for s in CRYPTO_SUFFIXES)


def _run_sweep(
    cached: dict[str, pd.DataFrame],
    cfg: TradingConfig,
    cooldown: int,
    tickers: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    subset = {t: cached[t] for t in tickers} if tickers else cached
    return run_backtest(subset, cfg, cooldown_days=cooldown)


def print_comparison_table(results: list[dict]) -> None:
    print(
        f"  {'Config':<16s} {'CD':>3s} {'Trades':>7s} {'Tr/Yr':>6s} "
        f"{'WinRate':>8s} {'AvgBars':>8s} {'CAGR':>8s} {'MaxDD':>7s} "
        f"{'Sharpe':>7s} {'PF':>6s}"
    )
    print("  " + "-" * 100)
    for r in results:
        n_years = r["n_years"] if r["n_years"] > 0 else 1
        tr_yr = r["n_trades"] / n_years
        print(
            f"  {r['config']:<16s} {r['cooldown']:>3d} {r['n_trades']:>7d} {tr_yr:>6.0f} "
            f"{r['win_rate']:>7.1%} {r['avg_bars']:>8.1f} {r['cagr']:>+7.1%} "
            f"{r['max_dd']:>+6.1%} {r['sharpe']:>7.2f} {r['pf']:>6.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Cooldown period experiment")
    parser.add_argument("--tickers", nargs="+", metavar="T", default=None)
    parser.add_argument("--cooldowns", nargs="+", type=int, default=[0, 3, 5, 10])
    parser.add_argument("--configs", nargs="+", default=None)
    args = parser.parse_args()

    print(f"\nLoading predictions cache: {CACHE}")
    with open(CACHE, "rb") as f:
        cached = pickle.load(f)
    print(f"Loaded {len(cached)} assets")

    cached = filter_tickers(cached, args.tickers)

    configs = CONFIGS
    if args.configs:
        configs = [c for c in CONFIGS if c.name in args.configs]
        if not configs:
            print(f"ERROR: no matching configs for {args.configs}")
            return

    cooldowns = args.cooldowns
    crypto_tickers = [t for t in cached if is_crypto(t)]
    equity_tickers = [t for t in cached if not is_crypto(t)]

    print(f"\n{'='*110}")
    print(f"  COOLDOWN EXPERIMENT — {len(cached)} assets, cooldowns={cooldowns}")
    print(f"{'='*110}\n")

    all_results = []
    for cfg in configs:
        for cd in cooldowns:
            metrics, trades = _run_sweep(cached, cfg, cd)
            avg_bars = trades["bars_held"].mean() if not trades.empty else 0
            all_results.append({
                "config": cfg.name,
                "cooldown": cd,
                "n_trades": metrics["n_trades"],
                "n_years": metrics.get("n_years", 1),
                "win_rate": metrics["win_rate"],
                "avg_bars": avg_bars,
                "cagr": metrics["cagr"],
                "max_dd": metrics["max_dd"],
                "sharpe": metrics["sharpe"],
                "pf": metrics["pf"],
            })

    print_comparison_table(all_results)

    print(f"\n{'='*110}")
    print("  DELTA vs NO-COOLDOWN (cooldown=0 baseline)")
    print(f"{'='*110}")
    print(
        f"  {'Config':<16s} {'CD':>3s} {'dTrades':>8s} {'dWinRate':>9s} "
        f"{'dCAGR':>8s} {'dMaxDD':>8s} {'dSharpe':>8s}"
    )
    print("  " + "-" * 72)

    df = pd.DataFrame(all_results)
    for cfg_name in df["config"].unique():
        cfg_df = df[df["config"] == cfg_name]
        base = cfg_df[cfg_df["cooldown"] == 0].iloc[0]
        for _, row in cfg_df.iterrows():
            if row["cooldown"] == 0:
                continue
            print(
                f"  {row['config']:<16s} {row['cooldown']:>3.0f} "
                f"{row['n_trades'] - base['n_trades']:>+8.0f} "
                f"{row['win_rate'] - base['win_rate']:>+8.1%} "
                f"{row['cagr'] - base['cagr']:>+8.1%} "
                f"{row['max_dd'] - base['max_dd']:>+8.1%} "
                f"{row['sharpe'] - base['sharpe']:>+8.2f}"
            )
        print("  " + "-" * 72)

    if crypto_tickers and equity_tickers:
        baseline_cfg = CONFIGS_BY_NAME["baseline"]
        print(f"\n{'='*110}")
        print(f"  ASSET CLASS BREAKDOWN (Baseline config) — "
              f"{len(equity_tickers)} equities, {len(crypto_tickers)} crypto")
        print(f"{'='*110}\n")

        class_results = []
        for asset_class, tickers in [("Equities", equity_tickers), ("Crypto", crypto_tickers)]:
            for cd in cooldowns:
                metrics, trades = _run_sweep(cached, baseline_cfg, cd, tickers)
                avg_bars = trades["bars_held"].mean() if not trades.empty else 0
                class_results.append({
                    "config": asset_class,
                    "cooldown": cd,
                    "n_trades": metrics["n_trades"],
                    "n_years": metrics.get("n_years", 1),
                    "win_rate": metrics["win_rate"],
                    "avg_bars": avg_bars,
                    "cagr": metrics["cagr"],
                    "max_dd": metrics["max_dd"],
                    "sharpe": metrics["sharpe"],
                    "pf": metrics["pf"],
                })

        print_comparison_table(class_results)

    print(f"\n{'='*110}")
    print("  DONE")
    print(f"{'='*110}\n")


if __name__ == "__main__":
    main()
