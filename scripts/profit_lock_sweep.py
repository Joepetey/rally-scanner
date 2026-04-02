#!/usr/bin/env python
"""
Sweep profit_lock_pct values to find the CAGR/Sharpe sweet spot.

Loads cached predictions, runs the baseline config across a range of
profit_lock_pct values (including 0 = disabled), and prints a ranked table.

Usage:
    python scripts/profit_lock_sweep.py
    python scripts/profit_lock_sweep.py --config conservative
    python scripts/profit_lock_sweep.py --profit-atr 1.0 --time-stop 3
    python scripts/profit_lock_sweep.py --lock-pcts 0 0.005 0.01 0.02 0.03 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import rally_ml.config.params as _params_mod  # noqa: E402
from rally_ml.config import CONFIGS_BY_NAME  # noqa: E402
from sweep_common import compute_exit_rates, filter_tickers, load_cache, run_backtest  # noqa: E402

DEFAULT_LOCK_PCTS = [0.0, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]


def _run_one(cached, cfg, lock_pct: float) -> dict:
    # Temporarily mutate PARAMS for profit_lock_pct (simulate_trades_fast reads it)
    original = _params_mod.PARAMS.profit_lock_pct
    try:
        _params_mod.PARAMS.profit_lock_pct = lock_pct
        m, portfolio_trades = run_backtest(cached, cfg)
    finally:
        _params_mod.PARAMS.profit_lock_pct = original

    n = m["n_trades"]
    n_years = m.get("n_years", 1)
    rates = compute_exit_rates(portfolio_trades, n)

    if not portfolio_trades.empty:
        avg_pnl = portfolio_trades["pnl_pct"].mean()
        win_rate = (portfolio_trades["pnl_pct"] > 0).mean()
    else:
        avg_pnl = win_rate = 0.0

    return {
        "lock_pct": lock_pct,
        "cagr": m["cagr"],
        "max_dd": m["max_dd"],
        "sharpe": m["sharpe"],
        "win_rate": win_rate,
        "pf": m["pf"],
        "n_trades": n,
        "tr_yr": n / n_years if n_years > 0 else 0,
        "avg_pnl": avg_pnl,
        **rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profit lock pct sweep")
    parser.add_argument("--config", default="conservative")
    parser.add_argument("--profit-atr", type=float, default=None,
                        help="Override profit_atr (e.g. 1.0 to match PARAMS.profit_atr_mult)")
    parser.add_argument("--time-stop", type=int, default=None,
                        help="Override time_stop bars (e.g. 3 to match PARAMS.time_stop_bars)")
    parser.add_argument("--lock-pcts", nargs="+", type=float, default=DEFAULT_LOCK_PCTS,
                        metavar="X")
    parser.add_argument("--tickers", nargs="+", metavar="T", default=None,
                        help="Limit analysis to specific tickers (e.g. BTC-USD)")
    args = parser.parse_args()

    from rally_ml.config import TradingConfig  # noqa: E402
    cfg = CONFIGS_BY_NAME.get(args.config.lower().replace(" ", "_"))
    if cfg is None:
        print(f"ERROR: Unknown config '{args.config}'")
        sys.exit(1)

    if args.profit_atr is not None or args.time_stop is not None:
        cfg = TradingConfig(
            name=cfg.name,
            p_rally=cfg.p_rally,
            comp_score=cfg.comp_score,
            max_risk=cfg.max_risk,
            vol_k=cfg.vol_k,
            profit_atr=args.profit_atr if args.profit_atr is not None else cfg.profit_atr,
            time_stop=args.time_stop if args.time_stop is not None else cfg.time_stop,
            leverage=cfg.leverage,
            cash_yield=cfg.cash_yield,
        )

    print("=" * 110)
    print("  PROFIT LOCK PCT SWEEP")
    print(f"  Base config : {cfg.name}  (PA={cfg.profit_atr}x, TS={cfg.time_stop}d)")
    print(f"  Lock pcts   : {args.lock_pcts}")
    print("=" * 110)

    cached = filter_tickers(load_cache(), args.tickers)

    results = []
    for pct in args.lock_pcts:
        label = f"{pct:.1%}" if pct > 0 else "disabled"
        r = _run_one(cached, cfg, pct)
        results.append(r)
        print(f"  lock={label:<9}  CAGR={r['cagr']:>+6.1%}  Sharpe={r['sharpe']:.2f}  "
              f"MaxDD={r['max_dd']:>+5.1%}  WinRate={r['win_rate']:.0%}  "
              f"AvgPnL={r['avg_pnl']:>+5.2%}  AvgBars={r['avg_bars']:.1f}",
              flush=True)

    print(f"\n{'='*110}")
    print(f"  {'Lock%':>9} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'WinRate':>8} "
          f"{'PF':>6} {'AvgPnL':>8} {'AvgBars':>8} {'TP%':>5} {'Stop%':>6} "
          f"{'Time%':>6} {'Trail%':>6} {'Tr/Yr':>6}")
    print("  " + "-" * 108)

    for r in results:
        label = f"{r['lock_pct']:.1%}" if r['lock_pct'] > 0 else "disabled"
        pf = min(r["pf"], 99.9) if not np.isinf(r["pf"]) else 99.9
        print(
            f"  {label:>9} "
            f"{r['cagr']:>+6.1%} "
            f"{r['max_dd']:>+6.1%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['win_rate']:>7.0%} "
            f"{pf:>6.2f} "
            f"{r['avg_pnl']:>+7.2%} "
            f"{r['avg_bars']:>8.1f} "
            f"{r['tp_rate']:>4.0%} "
            f"{r['stop_rate']:>5.0%} "
            f"{r['time_rate']:>5.0%} "
            f"{r['trail_rate']:>5.0%} "
            f"{r['tr_yr']:>5.0f}"
        )

    print(f"{'='*110}\n")


if __name__ == "__main__":
    main()
