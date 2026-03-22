#!/usr/bin/env python
"""
2D sweep of profit_atr (TP multiplier) × time_stop (holding period).

Loads cached predictions and runs every combination, then prints:
  1. Sharpe grid   — rows=time_stop, cols=profit_atr  (quick visual)
  2. CAGR grid
  3. Full sorted table (all combos ranked by Sharpe)

Runs in two modes: intraday-high TP (backtest) and close-only TP (production).

Usage:
    python scripts/tp_time_sweep.py
    python scripts/tp_time_sweep.py --config baseline
    python scripts/tp_time_sweep.py --profit-atrs 1.0 1.5 2.0 2.5 3.0 --time-stops 5 8 10 15
    python scripts/tp_time_sweep.py --close-only    # only run production-mode
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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

DEFAULT_PROFIT_ATRS = [1.0, 1.5, 2.0, 2.5, 3.0]
DEFAULT_TIME_STOPS  = [3, 5, 8, 10, 15, 20]


def _load_cache() -> dict[str, pd.DataFrame]:
    if not PREDICTIONS_CACHE.exists():
        print("ERROR: No predictions cache found.")
        print("Run first:  python backtest_universe.py")
        sys.exit(1)
    with open(PREDICTIONS_CACHE, "rb") as f:
        cached = pickle.load(f)
    print(f"Loaded {len(cached)} cached assets")
    return cached


def _run_grid(
    cached: dict[str, pd.DataFrame],
    base_cfg: TradingConfig,
    profit_atrs: list[float],
    time_stops: list[int],
    close_only: bool,
) -> list[dict]:
    total = len(profit_atrs) * len(time_stops)
    done = 0
    results = []

    for pa in profit_atrs:
        for ts in time_stops:
            cfg = TradingConfig(
                name=f"pa{pa}_ts{ts}",
                p_rally=base_cfg.p_rally,
                comp_score=base_cfg.comp_score,
                max_risk=base_cfg.max_risk,
                vol_k=base_cfg.vol_k,
                profit_atr=pa,
                time_stop=ts,
                leverage=base_cfg.leverage,
                cash_yield=base_cfg.cash_yield,
            )

            all_trades = []
            for preds in cached.values():
                signal = generate_signals_fast(preds, cfg, require_trend=True)
                trades = simulate_trades_fast(preds, signal, cfg, close_only_tp=close_only)
                if not trades.empty:
                    all_trades.append(trades)

            portfolio_trades = (
                pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
            )
            m = simulate_portfolio(portfolio_trades, cfg)

            n = m["n_trades"]
            n_years = m.get("n_years", 1)

            if not portfolio_trades.empty:
                ec = portfolio_trades["exit_reason"].value_counts().to_dict()
                tp_rate   = ec.get("profit_target", 0) / n if n > 0 else 0
                stop_rate = ec.get("stop", 0) / n if n > 0 else 0
                time_rate = ec.get("time_stop", 0) / n if n > 0 else 0
                trail_rate = ec.get("trail_stop", 0) / n if n > 0 else 0
                avg_bars  = portfolio_trades["bars_held"].mean()
            else:
                tp_rate = stop_rate = time_rate = trail_rate = avg_bars = 0.0

            results.append({
                "profit_atr": pa,
                "time_stop": ts,
                "cagr":     m["cagr"],
                "max_dd":   m["max_dd"],
                "sharpe":   m["sharpe"],
                "win_rate": m["win_rate"],
                "pf":       m["pf"],
                "n_trades": n,
                "tr_yr":    n / n_years if n_years > 0 else 0,
                "tp_rate":  tp_rate,
                "stop_rate": stop_rate,
                "time_stop_rate": time_rate,
                "trail_rate": trail_rate,
                "avg_bars": avg_bars,
            })

            done += 1
            print(f"  [{done:>3}/{total}] pa={pa:.1f}x  ts={ts:>2}d  "
                  f"CAGR={m['cagr']:>+6.1%}  Sharpe={m['sharpe']:.2f}  "
                  f"MaxDD={m['max_dd']:>+5.1%}", flush=True)

    return results


def _print_grid(results: list[dict], metric: str, label: str,
                profit_atrs: list[float], time_stops: list[int]) -> None:
    print(f"\n  {label} grid  (rows=time_stop, cols=profit_atr)")
    header = f"  {'Hold':>5}  " + "  ".join(f"PA={pa:.1f}" for pa in profit_atrs)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ts in time_stops:
        row_vals = []
        for pa in profit_atrs:
            match = next((r for r in results if r["profit_atr"] == pa and r["time_stop"] == ts), None)
            if match:
                v = match[metric]
                if metric in ("cagr", "max_dd", "win_rate"):
                    row_vals.append(f"{v:>+6.1%}")
                else:
                    row_vals.append(f"{min(v, 99.9):>6.2f}")
            else:
                row_vals.append("   n/a")
        print(f"  {ts:>5}d  " + "  ".join(row_vals))


def _print_ranked(results: list[dict]) -> None:
    sorted_r = sorted(results, key=lambda x: x["sharpe"], reverse=True)
    print(f"\n  {'PA':>5} {'Hold':>5} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'WinRate':>8} {'PF':>6} {'TP%':>5} {'Stop%':>6} {'Time%':>6} "
          f"{'Trail%':>6} {'AvgBars':>8}")
    print("  " + "-" * 100)
    for r in sorted_r:
        pf = min(r["pf"], 99.9) if not np.isinf(r["pf"]) else 99.9
        print(
            f"  {r['profit_atr']:>4.1f}x "
            f"{r['time_stop']:>5}d "
            f"{r['cagr']:>+6.1%} "
            f"{r['max_dd']:>+6.1%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['win_rate']:>7.0%} "
            f"{pf:>6.2f} "
            f"{r['tp_rate']:>4.0%} "
            f"{r['stop_rate']:>5.0%} "
            f"{r['time_stop_rate']:>5.0%} "
            f"{r['trail_rate']:>5.0%} "
            f"{r['avg_bars']:>8.1f}"
        )


def _section(title: str, close_only: bool, cached, base_cfg, profit_atrs, time_stops) -> None:
    mode = "CLOSE-only TP (production)" if close_only else "Intraday-HIGH TP (backtest)"
    print(f"\n{'='*110}")
    print(f"  {title} — {mode}")
    print(f"{'='*110}")
    results = _run_grid(cached, base_cfg, profit_atrs, time_stops, close_only)
    _print_grid(results, "sharpe", "SHARPE", profit_atrs, time_stops)
    _print_grid(results, "cagr",   "CAGR",   profit_atrs, time_stops)
    print("\n  ALL COMBOS RANKED BY SHARPE:")
    _print_ranked(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D TP × holding-period sweep")
    parser.add_argument("--config", default="conservative",
                        help="Base config (default: conservative)")
    parser.add_argument("--profit-atrs", nargs="+", type=float, default=DEFAULT_PROFIT_ATRS,
                        metavar="X", help=f"TP multipliers (default: {DEFAULT_PROFIT_ATRS})")
    parser.add_argument("--time-stops", nargs="+", type=int, default=DEFAULT_TIME_STOPS,
                        metavar="N", help=f"Holding periods in bars (default: {DEFAULT_TIME_STOPS})")
    parser.add_argument("--close-only", action="store_true",
                        help="Only run close-only TP mode (skip intraday-high)")
    parser.add_argument("--high-only", action="store_true",
                        help="Only run intraday-high TP mode (skip close-only)")
    args = parser.parse_args()

    base_cfg = CONFIGS_BY_NAME.get(args.config.lower().replace(" ", "_"))
    if base_cfg is None:
        print(f"ERROR: Unknown config '{args.config}'")
        sys.exit(1)

    n_combos = len(args.profit_atrs) * len(args.time_stops)
    print("=" * 110)
    print("  TP × HOLDING PERIOD 2D SWEEP")
    print(f"  Base config : {base_cfg.name}")
    print(f"  Profit ATRs : {args.profit_atrs}")
    print(f"  Time stops  : {args.time_stops}")
    print(f"  Combos      : {n_combos} × 2 modes = {n_combos * 2} runs")
    print("=" * 110)

    cached = _load_cache()

    if not args.close_only:
        _section("BACKTEST", False, cached, base_cfg, args.profit_atrs, args.time_stops)

    if not args.high_only:
        _section("PRODUCTION", True, cached, base_cfg, args.profit_atrs, args.time_stops)

    print()


if __name__ == "__main__":
    main()
