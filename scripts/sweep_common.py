"""Shared helpers for backtest sweep scripts.

Centralizes exit-reason aggregation, cache loading, and ticker filtering
so individual sweep scripts stay focused on their parameter grid.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "backtest_cache"
PREDICTIONS_CACHE = CACHE_DIR / "predictions.pkl"


def load_cache() -> dict[str, pd.DataFrame]:
    """Load the predictions cache or exit with an error message."""
    if not PREDICTIONS_CACHE.exists():
        print("ERROR: No predictions cache found.")
        print("Run first:  python backtest_universe.py")
        sys.exit(1)
    with open(PREDICTIONS_CACHE, "rb") as f:
        cached = pickle.load(f)
    print(f"Loaded {len(cached)} cached assets")
    return cached


def filter_tickers(
    cached: dict[str, pd.DataFrame], tickers: list[str] | None,
) -> dict[str, pd.DataFrame]:
    """Filter cache to specific tickers, warning about missing ones."""
    if not tickers:
        return cached
    missing = [t for t in tickers if t not in cached]
    if missing:
        print(f"WARNING: tickers not in cache: {missing}")
    filtered = {t: cached[t] for t in tickers if t in cached}
    print(f"Filtered to {len(filtered)} tickers: {list(filtered.keys())}")
    return filtered


def compute_exit_rates(
    portfolio_trades: pd.DataFrame, n_trades: int,
) -> dict[str, float]:
    """Compute exit-reason rates from portfolio trades.

    Returns dict with keys: tp_rate, stop_rate, time_rate, trail_rate, avg_bars.
    All values default to 0.0 if no trades.
    """
    if portfolio_trades.empty or n_trades == 0:
        return {
            "tp_rate": 0.0,
            "stop_rate": 0.0,
            "time_rate": 0.0,
            "trail_rate": 0.0,
            "avg_bars": 0.0,
        }

    ec = portfolio_trades["exit_reason"].value_counts().to_dict()
    return {
        "tp_rate": ec.get("profit_target", 0) / n_trades,
        "stop_rate": ec.get("stop", 0) / n_trades,
        "time_rate": ec.get("time_stop", 0) / n_trades,
        "trail_rate": ec.get("trail_stop", 0) / n_trades,
        "avg_bars": portfolio_trades["bars_held"].mean(),
    }


def run_backtest(
    cached: dict[str, pd.DataFrame],
    cfg,
    *,
    close_only_tp: bool = False,
    limit_sell_on_tp: bool = False,
    cooldown_days: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """Run generate_signals + simulate_trades + simulate_portfolio.

    Returns (metrics_dict, portfolio_trades_df).
    """
    from rally_ml.backtest.common import (
        generate_signals_fast,
        simulate_portfolio,
        simulate_trades_fast,
    )

    all_trades = []
    for preds in cached.values():
        signal = generate_signals_fast(preds, cfg, require_trend=True)
        trades = simulate_trades_fast(
            preds, signal, cfg,
            close_only_tp=close_only_tp,
            limit_sell_on_tp=limit_sell_on_tp,
            cooldown_days=cooldown_days,
        )
        if not trades.empty:
            all_trades.append(trades)

    portfolio_trades = (
        pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    )
    metrics = simulate_portfolio(portfolio_trades, cfg)
    return metrics, portfolio_trades
