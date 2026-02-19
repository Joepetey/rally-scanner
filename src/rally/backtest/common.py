"""Shared backtest primitives — Config presets, signal generation, trade simulation,
and portfolio-level performance evaluation.

Used by optimize.py, backtest_universe.py, and scanner.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PARAMS


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

# Lookup by name for scanner.py apply_config()
CONFIGS_BY_NAME: dict[str, Config] = {c.name.lower().replace(" ", "_"): c for c in CONFIGS}
# Add friendly aliases used by scanner CLI
CONFIGS_BY_NAME.update({
    "baseline": CONFIGS[0],
    "conservative": CONFIGS[9],
    "aggressive": CONFIGS[2],
    "concentrated": CONFIGS[4],
})


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
    p = PARAMS
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
                new_trail = highest_close - p.trailing_stop_atr_mult * atr[entry_idx]
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
                    "exit_reason": exit_reason,
                })
                in_trade = False

        if not in_trade and signal.iloc[i]:
            entry_idx = i
            entry_price = close[i]
            stop_price = preds["RangeLow"].iloc[i]
            if np.isnan(stop_price) or stop_price >= entry_price:
                stop_price = entry_price * (1 - p.fallback_stop_pct)

            p_rally = preds["P_RALLY"].iloc[i]
            atr_pct = preds["ATR_pct"].iloc[i]
            raw_size = cfg.vol_k * (p_rally - cfg.p_rally) / atr_pct if atr_pct > 0 else 0
            size = max(0.0, min(raw_size, cfg.max_risk))
            if size >= 0.01:
                in_trade = True
                bars_held = 0
                highest_close = entry_price
                trailing_stop = entry_price - p.trailing_stop_atr_mult * atr[i]

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def simulate_portfolio(all_trades: pd.DataFrame, cfg: Config,
                       initial_capital: float = 100_000) -> dict:
    """Simulate portfolio with leverage and cash yield on idle capital."""
    if all_trades.empty:
        return {"cagr": 0, "max_dd": 0, "sharpe": 0, "total_return": 0,
                "n_trades": 0, "win_rate": 0, "pf": 0, "n_years": 0}

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
