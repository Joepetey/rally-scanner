#!/usr/bin/env python
"""
Holding-period sweep — isolates the effect of time_stop on portfolio performance.

Loads cached predictions and sweeps only time_stop, keeping all other parameters
fixed. Also runs each time_stop with close_only_tp=True to show the production gap.

Usage:
    python scripts/holding_period_sweep.py
    python scripts/holding_period_sweep.py --config baseline
    python scripts/holding_period_sweep.py --time-stops 3 5 8 10 15 20 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rally_ml.config import CONFIGS_BY_NAME, TradingConfig  # noqa: E402
from sweep_common import compute_exit_rates, filter_tickers, load_cache, run_backtest  # noqa: E402

DEFAULT_TIME_STOPS = [3, 5, 8, 10, 12, 15, 20, 30]


def _run_sweep(cached, base_cfg, time_stops) -> list[dict]:
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
            m, portfolio_trades = run_backtest(
                cached, cfg, close_only_tp=close_only,
            )
            n = m["n_trades"]
            n_years = m.get("n_years", 1)
            rates = compute_exit_rates(portfolio_trades, n)

            results.append({
                "time_stop": ts,
                "close_only_tp": close_only,
                "cagr": m["cagr"],
                "max_dd": m["max_dd"],
                "sharpe": m["sharpe"],
                "win_rate": m["win_rate"],
                "pf": m["pf"],
                "n_trades": n,
                "tr_yr": n / n_years if n_years > 0 else 0,
                **rates,
            })
    return results


def _print_table(results, label) -> None:
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
            f"{r['time_rate']:>5.0%} "
            f"{r['trail_rate']:>5.0%} "
            f"{r['avg_bars']:>8.1f} "
            f"{r['tr_yr']:>6.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Holding-period sweep")
    parser.add_argument("--config", default="conservative")
    parser.add_argument("--time-stops", nargs="+", type=int, default=DEFAULT_TIME_STOPS,
                        metavar="N")
    parser.add_argument("--tickers", nargs="+", metavar="T", default=None)
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

    cached = filter_tickers(load_cache(), args.tickers)
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
