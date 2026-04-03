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
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rally_ml.config import CONFIGS_BY_NAME, TradingConfig  # noqa: E402
from sweep_common import compute_exit_rates, filter_tickers, load_cache, run_backtest  # noqa: E402

DEFAULT_PROFIT_ATRS = [1.0, 1.5, 2.0, 2.5, 3.0]
DEFAULT_TIME_STOPS  = [3, 5, 8, 10, 15, 20]


def _run_grid(
    cached, base_cfg, profit_atrs, time_stops, close_only, limit_sell=False,
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

            m, portfolio_trades = run_backtest(
                cached, cfg,
                close_only_tp=close_only,
                limit_sell_on_tp=limit_sell,
            )

            n = m["n_trades"]
            n_years = m.get("n_years", 1)
            rates = compute_exit_rates(portfolio_trades, n)

            results.append({
                "profit_atr": pa,
                "time_stop": ts,
                "cagr": m["cagr"],
                "max_dd": m["max_dd"],
                "sharpe": m["sharpe"],
                "win_rate": m["win_rate"],
                "pf": m["pf"],
                "n_trades": n,
                "tr_yr": n / n_years if n_years > 0 else 0,
                **rates,
            })

            done += 1
            print(f"  [{done:>3}/{total}] pa={pa:.1f}x  ts={ts:>2}d  "
                  f"CAGR={m['cagr']:>+6.1%}  Sharpe={m['sharpe']:.2f}  "
                  f"MaxDD={m['max_dd']:>+5.1%}", flush=True)

    return results


def _print_grid(results, metric, label, profit_atrs, time_stops) -> None:
    print(f"\n  {label} grid  (rows=time_stop, cols=profit_atr)")
    header = f"  {'Hold':>5}  " + "  ".join(f"PA={pa:.1f}" for pa in profit_atrs)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ts in time_stops:
        row_vals = []
        for pa in profit_atrs:
            match = next(
                (r for r in results if r["profit_atr"] == pa and r["time_stop"] == ts), None,
            )
            if match:
                v = match[metric]
                if metric in ("cagr", "max_dd", "win_rate"):
                    row_vals.append(f"{v:>+6.1%}")
                else:
                    row_vals.append(f"{min(v, 99.9):>6.2f}")
            else:
                row_vals.append("   n/a")
        print(f"  {ts:>5}d  " + "  ".join(row_vals))


def _print_ranked(results) -> None:
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
            f"{r['time_rate']:>5.0%} "
            f"{r['trail_rate']:>5.0%} "
            f"{r['avg_bars']:>8.1f}"
        )


def _section(title, close_only, cached, base_cfg, profit_atrs, time_stops,
             limit_sell=False) -> None:
    if limit_sell:
        mode = "Limit sell on TP touch (deferred exit)"
    elif close_only:
        mode = "CLOSE-only TP (production)"
    else:
        mode = "Intraday-HIGH TP (backtest)"
    print(f"\n{'='*110}")
    print(f"  {title} — {mode}")
    print(f"{'='*110}")
    results = _run_grid(cached, base_cfg, profit_atrs, time_stops, close_only,
                        limit_sell=limit_sell)
    _print_grid(results, "sharpe", "SHARPE", profit_atrs, time_stops)
    _print_grid(results, "cagr",   "CAGR",   profit_atrs, time_stops)
    print("\n  ALL COMBOS RANKED BY SHARPE:")
    _print_ranked(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D TP × holding-period sweep")
    parser.add_argument("--config", default="conservative")
    parser.add_argument("--profit-atrs", nargs="+", type=float, default=DEFAULT_PROFIT_ATRS,
                        metavar="X")
    parser.add_argument("--time-stops", nargs="+", type=int, default=DEFAULT_TIME_STOPS,
                        metavar="N")
    parser.add_argument("--close-only", action="store_true")
    parser.add_argument("--high-only", action="store_true")
    parser.add_argument("--limit-sell", action="store_true")
    parser.add_argument("--tickers", nargs="+", metavar="T", default=None)
    args = parser.parse_args()

    base_cfg = CONFIGS_BY_NAME.get(args.config.lower().replace(" ", "_"))
    if base_cfg is None:
        print(f"ERROR: Unknown config '{args.config}'")
        sys.exit(1)

    n_combos = len(args.profit_atrs) * len(args.time_stops)
    n_modes = 2 + (1 if args.limit_sell else 0)
    print("=" * 110)
    print("  TP × HOLDING PERIOD 2D SWEEP")
    print(f"  Base config : {base_cfg.name}")
    print(f"  Profit ATRs : {args.profit_atrs}")
    print(f"  Time stops  : {args.time_stops}")
    print(f"  Combos      : {n_combos} × {n_modes} modes = {n_combos * n_modes} runs")
    print("=" * 110)

    cached = filter_tickers(load_cache(), args.tickers)

    if not args.close_only:
        _section("BACKTEST", False, cached, base_cfg, args.profit_atrs, args.time_stops)

    if not args.high_only:
        _section("PRODUCTION", True, cached, base_cfg, args.profit_atrs, args.time_stops)

    if args.limit_sell:
        _section("LIMIT SELL ON TP", False, cached, base_cfg, args.profit_atrs, args.time_stops,
                 limit_sell=True)

    print()


if __name__ == "__main__":
    main()
