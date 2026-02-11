"""
Backtesting engine — equity curve, metrics, diagnostics.
"""

import numpy as np
import pandas as pd


def compute_equity_curve(trades: pd.DataFrame, initial_capital: float = 100_000) -> pd.DataFrame:
    """
    Build a daily equity curve from sized trades.
    Returns DataFrame with Date index, columns: equity, drawdown.
    """
    if trades.empty:
        return pd.DataFrame()

    # Build daily PnL from trades
    all_dates = pd.date_range(
        trades["entry_date"].min(),
        trades["exit_date"].max(),
        freq="B",
    )
    daily_pnl = pd.Series(0.0, index=all_dates)

    for _, t in trades.iterrows():
        # Distribute PnL evenly across bars held (simplified)
        mask = (daily_pnl.index >= t["entry_date"]) & (daily_pnl.index <= t["exit_date"])
        n_days = mask.sum()
        if n_days > 0:
            daily_pnl[mask] += (t["pnl_sized"] * initial_capital) / n_days

    equity = initial_capital + daily_pnl.cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    return pd.DataFrame({"equity": equity, "drawdown": drawdown})


def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    """
    Compute the metrics that matter (per spec section 6).
    """
    if trades.empty:
        return {"error": "no trades"}

    pnl = trades["pnl_pct"]
    sized_pnl = trades["pnl_sized"]

    # CAGR
    if not equity.empty and len(equity) > 1:
        total_days = (equity.index[-1] - equity.index[0]).days
        total_return = equity["equity"].iloc[-1] / equity["equity"].iloc[0]
        cagr = total_return ** (365.25 / max(total_days, 1)) - 1 if total_days > 0 else 0
    else:
        cagr = 0

    # Max drawdown
    max_dd = equity["drawdown"].min() if not equity.empty else 0

    # Profit factor
    gross_profit = sized_pnl[sized_pnl > 0].sum()
    gross_loss = -sized_pnl[sized_pnl < 0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Win rate
    win_rate = (pnl > 0).mean()

    # Avg MAE / MFE
    avg_mae = trades["mae"].mean()
    avg_mfe = trades["mfe"].mean()

    # Avg winner / loser
    winners = pnl[pnl > 0]
    losers = pnl[pnl <= 0]
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 0

    # Exit reasons
    exit_counts = trades["exit_reason"].value_counts().to_dict()

    # Avg bars held
    avg_bars = trades["bars_held"].mean()

    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win * 100, 3),
        "avg_loss_pct": round(avg_loss * 100, 3),
        "profit_factor": round(profit_factor, 3),
        "cagr": round(cagr * 100, 3),
        "max_drawdown": round(max_dd * 100, 3),
        "avg_mae_pct": round(avg_mae * 100, 3),
        "avg_mfe_pct": round(avg_mfe * 100, 3),
        "avg_bars_held": round(avg_bars, 1),
        "exit_reasons": exit_counts,
    }


def performance_by_comp_decile(trades: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by COMP_SCORE decile."""
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["comp_decile"] = pd.qcut(trades["comp_score"], 10, labels=False, duplicates="drop")
    return trades.groupby("comp_decile").agg(
        count=("pnl_pct", "count"),
        mean_pnl=("pnl_pct", "mean"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean()),
    ).round(4)


def performance_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by Trend on/off."""
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby("trend").agg(
        count=("pnl_pct", "count"),
        mean_pnl=("pnl_pct", "mean"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean()),
    ).round(4)


def performance_by_year(trades: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by year — critical for regime stress-testing."""
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["year"] = trades["entry_date"].dt.year
    grouped = trades.groupby("year").agg(
        trades=("pnl_pct", "count"),
        mean_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_sized", "sum"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean()),
    ).round(4)
    return grouped


def print_report(trades: pd.DataFrame, equity: pd.DataFrame, asset_name: str) -> dict:
    """Print a full diagnostic report."""
    metrics = compute_metrics(trades, equity)

    print(f"\n{'='*60}")
    print(f"  RALLY DETECTOR BACKTEST — {asset_name}")
    print(f"{'='*60}")

    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return metrics

    print(f"  Total trades:      {metrics['total_trades']}")
    print(f"  Win rate:          {metrics['win_rate']:.1%}")
    print(f"  Avg winner:        {metrics['avg_win_pct']:+.2f}%")
    print(f"  Avg loser:         {metrics['avg_loss_pct']:+.2f}%")
    print(f"  Profit factor:     {metrics['profit_factor']:.2f}")
    print(f"  CAGR:              {metrics['cagr']:+.2f}%")
    print(f"  Max drawdown:      {metrics['max_drawdown']:.2f}%")
    print(f"  Avg MAE:           {metrics['avg_mae_pct']:.2f}%")
    print(f"  Avg MFE:           {metrics['avg_mfe_pct']:+.2f}%")
    print(f"  Avg bars held:     {metrics['avg_bars_held']:.1f}")
    print(f"\n  Exit reasons:")
    for reason, cnt in metrics.get("exit_reasons", {}).items():
        print(f"    {reason:20s}  {cnt}")

    print(f"\n  --- Performance by COMP_SCORE decile ---")
    dec = performance_by_comp_decile(trades)
    if not dec.empty:
        print(dec.to_string())
    else:
        print("  (not enough data)")

    print(f"\n  --- Performance by regime (Trend) ---")
    reg = performance_by_regime(trades)
    if not reg.empty:
        print(reg.to_string())
    else:
        print("  (not enough data)")

    print(f"\n  --- Performance by YEAR ---")
    yearly = performance_by_year(trades)
    if not yearly.empty:
        print(yearly.to_string())
    else:
        print("  (not enough data)")

    print(f"{'='*60}\n")
    return metrics
