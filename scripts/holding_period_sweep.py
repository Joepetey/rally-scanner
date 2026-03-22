#!/usr/bin/env python
"""
Holding-period sweep — isolates the effect of time_stop on portfolio performance.

Loads cached predictions (produced by backtest_universe.py) and sweeps only
time_stop, keeping all other parameters fixed at the Conservative config.
Also runs each time_stop with close_only_tp=True to show the realistic
production gap (backtest uses intraday high for TP; production checks close/15m poll).

Usage:
    python scripts/holding_period_sweep.py
    python scripts/holding_period_sweep.py --config baseline
    python scripts/holding_period_sweep.py --time-stops 3 5 8 10 15 20 30
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make sure the package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CONFIGS_BY_NAME, TradingConfig  # noqa: E402
from backtest.common import (  # noqa: E402
    generate_signals_fast,
    simulate_portfolio,
    simulate_trades_fast,
)

CACHE_DIR = PROJECT_ROOT / "backtest_cache"
PREDICTIONS_CACHE = CACHE_DIR / "predictions.pkl"

DEFAULT_TIME_STOPS = [3, 5, 8, 10, 12, 15, 20, 30]


def _load_cache() -> dict[str, pd.DataFrame]:
    if not PREDICTIONS_CACHE.exists():
        print("ERROR: No predictions cache found.")
        print("Run first:  python backtest_universe.py  (or with --tickers AAPL MSFT ...)")
        sys.exit(1)
    with open(PREDICTIONS_CACHE, "rb") as f:
        cached = pickle.load(f)
    print(f"Loaded {len(cached)} cached assets")
    return cached


def _run_sweep(
    cached: dict[str, pd.DataFrame],
    base_cfg: TradingConfig,
    time_stops: list[int],
) -> list[dict]:
    results = []
    for ts in time_stops:
        cfg = TradingConfig(
            name=f"hold_{ts}d",
            p_rally=base_cfg.p_rally,
            comp_score=base_cfg.comp_score,
            max_risk=base_cfg.max_risk,
            vol_k=base_cfg.vol_k,
            profit_atr=base_cfg.profit_atr,
            time_stop=ts,
            leverage=base_cfg.leverage,
            cash_yield=base_cfg.cash_yield,
        )
        for close_only in (False, True):
            all_trades = []
            for ticker, preds in cached.items():
                signal = generate_signals_fast(preds, cfg, require_trend=True)
                trades = simulate_trades_fast(preds, signal, cfg, close_only_tp=close_only)
                if not trades.empty:
                    trades["asset"] = ticker
                    all_trades.append(trades)

            portfolio_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
            metrics = simulate_portfolio(portfolio_trades, cfg)

            n = metrics["n_trades"]
            n_years = metrics.get("n_years", 1)
            tr_yr = n / n_years if n_years > 0 else 0

            if not portfolio_trades.empty:
                exit_counts = portfolio_trades["exit_reason"].value_counts().to_dict()
                tp_rate = exit_counts.get("profit_target", 0) / n if n > 0 else 0
                stop_rate = exit_counts.get("stop", 0) / n if n > 0 else 0
                time_rate = exit_counts.get("time_stop", 0) / n if n > 0 else 0
                trail_rate = exit_counts.get("trail_stop", 0) / n if n > 0 else 0
                avg_bars = portfolio_trades["bars_held"].mean()
            else:
                tp_rate = stop_rate = time_rate = trail_rate = avg_bars = 0.0

            results.append({
                "time_stop": ts,
                "close_only_tp": close_only,
                "cagr": metrics["cagr"],
                "max_dd": metrics["max_dd"],
                "sharpe": metrics["sharpe"],
                "win_rate": metrics["win_rate"],
                "pf": metrics["pf"],
                "n_trades": n,
                "tr_yr": tr_yr,
                "tp_rate": tp_rate,
                "stop_rate": stop_rate,
                "time_stop_rate": time_rate,
                "trail_rate": trail_rate,
                "avg_bars": avg_bars,
            })
    return results


def _print_table(results: list[dict], label: str) -> None:
    rows = [r for r in results if r["close_only_tp"] == (label == "close")]
    tp_label = "close" if label == "close" else "high (backtest)"
    print(f"\n  TP trigger: {tp_label}")
    print(f"  {'Hold':>5} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'WinRate':>8} "
          f"{'PF':>6} {'TP%':>6} {'Stop%':>6} {'Time%':>6} {'Trail%':>6} "
          f"{'AvgBars':>8} {'Tr/Yr':>6}")
    print("  " + "-" * 95)
    for r in rows:
        pf_disp = min(r["pf"], 99.9) if not np.isinf(r["pf"]) else 99.9
        print(
            f"  {r['time_stop']:>5}d "
            f"{r['cagr']:>+6.1%} "
            f"{r['max_dd']:>+6.1%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['win_rate']:>7.0%} "
            f"{pf_disp:>6.2f} "
            f"{r['tp_rate']:>5.0%} "
            f"{r['stop_rate']:>5.0%} "
            f"{r['time_stop_rate']:>5.0%} "
            f"{r['trail_rate']:>5.0%} "
            f"{r['avg_bars']:>8.1f} "
            f"{r['tr_yr']:>6.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Holding-period sweep")
    parser.add_argument(
        "--config", default="conservative",
        help="Base config name (default: conservative). Options: baseline, conservative, aggressive, concentrated",
    )
    parser.add_argument(
        "--time-stops", nargs="+", type=int, default=DEFAULT_TIME_STOPS,
        metavar="N",
        help=f"Holding periods to test in bars (default: {DEFAULT_TIME_STOPS})",
    )
    args = parser.parse_args()

    base_cfg = CONFIGS_BY_NAME.get(args.config.lower().replace(" ", "_"))
    if base_cfg is None:
        print(f"ERROR: Unknown config '{args.config}'. Options: {list(CONFIGS_BY_NAME.keys())}")
        sys.exit(1)

    print("=" * 100)
    print("  HOLDING PERIOD SWEEP")
    print(f"  Base config: {base_cfg.name}  |  p_rally={base_cfg.p_rally}  "
          f"comp_score={base_cfg.comp_score}  profit_atr={base_cfg.profit_atr}x  "
          f"vol_k={base_cfg.vol_k}  max_risk={base_cfg.max_risk:.0%}")
    print(f"  Time stops: {args.time_stops}")
    print("=" * 100)

    cached = _load_cache()
    results = _run_sweep(cached, base_cfg, args.time_stops)

    print("\n  RESULTS — intraday HIGH triggers TP (standard backtest, optimistic):")
    _print_table(results, "high")

    print("\n  RESULTS — CLOSE-only TP (matches production daily-scan behaviour):")
    _print_table(results, "close")

    print("\n  NOTE: The gap between high-TP and close-TP rows shows how much")
    print("  performance the system 'loses' because production checks close, not intraday high.")
    print("  If price alerts (15m) are active, reality sits between the two.")
    print()


if __name__ == "__main__":
    main()
