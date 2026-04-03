"""Portfolio analytics: PnL, capital, equity history."""

from db.ops.users import set_capital
from db.trading.portfolio import load_equity_history, load_trade_journal
from db.trading.trades import get_pnl_summary


def get_pnl(capital: float, period: str = "all") -> dict:
    """Get P&L summary."""
    days = None
    if period == "30d":
        days = 30
    elif period == "7d":
        days = 7

    summary = get_pnl_summary(days=days)

    result = {
        "period": {"all": "All Time", "30d": "Last 30 Days", "7d": "Last 7 Days"}[period],
        "total_pnl_pct": summary["total_pnl"],
        "avg_pnl_pct": summary["avg_pnl"],
        "win_rate_pct": summary["win_rate"],
        "n_trades": summary["n_trades"],
        "best_trade_pct": summary["best_trade"],
        "worst_trade_pct": summary["worst_trade"],
    }

    dollar_pnl = summary.get("total_pnl_dollar", 0.0)
    if dollar_pnl != 0:
        result["total_pnl_dollar"] = dollar_pnl

    if capital > 0:
        result["capital"] = capital

    return result


def set_capital_amount(discord_id: int, amount: float) -> dict:
    """Set user's portfolio capital. Still needs discord_id for the users table."""
    if amount <= 0:
        return {"error": "Capital amount must be greater than 0"}

    set_capital(discord_id, amount)

    return {
        "success": True,
        "capital": amount,
        "message": f"Portfolio capital set to ${amount:,.2f}",
    }


def get_portfolio(days: int = 30) -> dict:
    """Get system portfolio info."""
    history = load_equity_history(days=days)
    trades = load_trade_journal(limit=10)

    if not history:
        return {"message": "No equity history yet. Run the orchestrator scan to start."}

    latest = history[-1]
    result = {
        "days": days,
        "latest_date": latest.get("date", "?"),
        "n_positions": latest.get("n_positions", 0),
        "total_exposure_pct": float(latest.get("total_exposure", 0)) * 100,
        "n_signals_today": latest.get("n_signals_today", 0),
        "n_scanned": latest.get("n_scanned", 0),
    }

    if trades:
        recent_trades = []
        for t in trades[-5:]:
            pnl = float(t.get("realized_pnl_pct", 0))
            recent_trades.append({
                "ticker": t.get("ticker", "?"),
                "pnl_pct": pnl,
                "exit_reason": t.get("exit_reason", "?"),
            })
        result["recent_closed_trades"] = recent_trades

    return result
