"""Framework-agnostic trading operations for rally-scanner.

All tool-execution logic lives here — any interface (Discord, API, CLI) can call
these functions without importing framework-specific code.
"""

import logging
from datetime import date, datetime

from rally_ml.core.data import fetch_quotes
from rally_ml.core.persistence import load_manifest

from db.portfolio import load_equity_history, load_trade_journal
from db.positions import load_watchlist, save_watchlist
from db.trades import (
    close_trade,
    get_open_trades,
    get_pnl_summary,
    open_trade,
)
from db.trades import (
    get_trade_history as _db_get_trade_history,
)
from db.users import set_capital
from pipeline.scanner import scan_all
from trading.positions import get_merged_positions_sync

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STALE_MODEL_DAYS = 14
MAX_STALE_DISPLAY = 10
DEFAULT_TRADE_LIMIT = 10
WATCHLIST_CAP = 20
MAX_PRICE_TICKERS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dollar_metrics(
    capital: float, size: float, entry: float, stop: float | None = None,
) -> dict:
    """Compute dollar allocation and risk from capital, size fraction, entry price, and stop."""
    metrics: dict = {}
    if capital > 0 and size > 0:
        metrics["dollar_allocation"] = capital * size
        if stop and entry > stop:
            metrics["dollar_risk"] = capital * size * (entry - stop) / entry
    return metrics


# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------

def get_signals(capital: float) -> dict:
    """Get recent entry signals (positions held <= 1 bar)."""
    state = get_merged_positions_sync()
    positions = state.get("positions", [])

    recent_entries = [
        p for p in positions
        if p.get("bars_held", 99) <= 1
    ]

    if not recent_entries:
        return {"message": "No recent signals from the latest scan."}

    signals_data = []
    for p in recent_entries:
        pnl = p.get("unrealized_pnl_pct", 0)
        size = p.get("size", 0)
        signal_info = {
            "ticker": p["ticker"],
            "entry_price": p["entry_price"],
            "stop_price": p.get("stop_price", 0),
            "target_price": p.get("target_price", 0),
            "size_pct": size * 100,
            "pnl_pct": pnl,
            **dollar_metrics(capital, size, p["entry_price"], p.get("stop_price", 0)),
        }

        signals_data.append(signal_info)

    return {
        "count": len(recent_entries),
        "signals": signals_data,
        "capital": capital if capital > 0 else None,
    }


def get_system_positions(capital: float) -> dict:
    """Get system's open positions."""
    state = get_merged_positions_sync()
    positions = state.get("positions", [])

    if not positions:
        return {"message": "The system has no open positions."}

    total_exposure = sum(p.get("size", 0) for p in positions)
    positions_data = []

    for p in positions:
        pnl = p.get("unrealized_pnl_pct", 0)
        size = p.get("size", 0)
        dollars = dollar_metrics(capital, size, p["entry_price"])
        pos_info = {
            "ticker": p["ticker"],
            "entry_price": p["entry_price"],
            "current_price": p.get("current_price", 0),
            "stop_price": p.get("stop_price", 0),
            "target_price": p.get("target_price", 0),
            "size_pct": size * 100,
            "pnl_pct": pnl,
            "bars_held": p.get("bars_held", 0),
            **dollars,
        }

        if "dollar_allocation" in dollars:
            pos_info["dollar_pnl"] = dollars["dollar_allocation"] * pnl / 100

        positions_data.append(pos_info)

    return {
        "count": len(positions),
        "total_exposure_pct": total_exposure * 100,
        "total_exposure_dollars": capital * total_exposure if capital > 0 else None,
        "positions": positions_data,
        "capital": capital if capital > 0 else None,
    }


def get_user_positions(capital: float) -> dict:
    """Get user's tracked open positions."""
    trades = get_open_trades()

    if not trades:
        return {"message": "You have no open positions."}

    positions_data = []
    for trade in trades:
        pos_info = {
            "ticker": trade["ticker"],
            "entry_price": trade["entry_price"],
            "entry_date": trade["entry_date"],
            "size_pct": trade["size"] * 100,
        }
        if trade.get("stop_price"):
            pos_info["stop_price"] = trade["stop_price"]
        if trade.get("target_price"):
            pos_info["target_price"] = trade["target_price"]

        pos_info.update(dollar_metrics(capital, trade["size"], trade["entry_price"]))

        positions_data.append(pos_info)

    return {
        "count": len(trades),
        "positions": positions_data,
        "capital": capital if capital > 0 else None,
    }


def get_watchlist() -> dict:
    """Return tickers near signal threshold from the last scan."""
    result = load_watchlist()
    if isinstance(result.get("tickers"), list) and len(result["tickers"]) > WATCHLIST_CAP:
        result["tickers"] = result["tickers"][:WATCHLIST_CAP]
        result["count"] = WATCHLIST_CAP
    return result


def get_price(tickers: list[str]) -> dict:
    """Fetch current quotes for one or more tickers."""
    if not tickers:
        return {"error": "No tickers provided"}
    if len(tickers) > MAX_PRICE_TICKERS:
        tickers = tickers[:MAX_PRICE_TICKERS]

    try:
        quotes = fetch_quotes(tickers)
    except Exception as e:
        return {"error": f"Failed to fetch prices: {e}"}

    if not quotes:
        return {"error": "No results returned"}

    return {"count": len(quotes), "quotes": quotes}


# ---------------------------------------------------------------------------
# Trade management
# ---------------------------------------------------------------------------

def enter_trade(tool_input: dict, capital: float) -> dict:
    """Record a trade entry."""
    ticker = tool_input["ticker"].upper()
    price = tool_input["price"]
    size = tool_input.get("size")
    stop = tool_input.get("stop")
    target = tool_input.get("target")
    notes = tool_input.get("notes")

    # Auto-fill from system recommendation if not provided
    state = get_merged_positions_sync()
    for p in state.get("positions", []):
        if p.get("ticker", "").upper() == ticker:
            if size is None:
                size = p.get("size", 0)
            if stop is None:
                stop = p.get("stop_price")
            if target is None:
                target = p.get("target_price")
            break

    if size is None:
        size = 0.0

    trade_id = open_trade(
        ticker=ticker,
        entry_price=price,
        size=size,
        stop_price=stop,
        target_price=target,
        notes=notes,
    )

    result = {
        "trade_id": trade_id,
        "ticker": ticker,
        "entry_price": price,
        "size_pct": size * 100,
    }

    if stop:
        risk_pct = (price - stop) / price * 100
        result["stop_price"] = stop
        result["risk_pct"] = risk_pct

    if target:
        reward_pct = (target - price) / price * 100
        result["target_price"] = target
        result["reward_pct"] = reward_pct

    dollars = dollar_metrics(capital, size, price, stop)
    if dollars:
        result.update(dollars)
        result["capital"] = capital

    return result


def exit_trade(tool_input: dict, capital: float) -> dict:
    """Close a trade."""
    ticker = tool_input["ticker"].upper()
    price = tool_input["price"]
    notes = tool_input.get("notes")

    result = close_trade(
        ticker=ticker,
        exit_price=price,
        notes=notes,
        capital=capital,
    )

    if result is None:
        return {"error": f"No open trades found for {ticker}"}

    response = {
        "ticker": result["ticker"],
        "entry_price": result["entry_price"],
        "exit_price": result["exit_price"],
        "pnl_pct": result["pnl_pct"],
        "size_pct": result["size"] * 100,
        "entry_date": result["entry_date"],
        "exit_date": result["exit_date"],
    }

    pnl_dollar = result.get("pnl_dollar")
    if pnl_dollar is not None:
        response["pnl_dollar"] = pnl_dollar

    return response


# ---------------------------------------------------------------------------
# Portfolio & PnL
# ---------------------------------------------------------------------------

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


def get_trade_history(
    ticker: str | None = None, limit: int = DEFAULT_TRADE_LIMIT,
) -> dict:
    """Get trade history."""
    trades = _db_get_trade_history(ticker=ticker, limit=limit)

    if not trades:
        msg = f"No trades found for {ticker.upper()}" if ticker else "No trade history"
        return {"message": msg}

    trades_data = []
    for t in trades:
        trade_info = {
            "ticker": t["ticker"],
            "status": t["status"],
            "entry_price": t["entry_price"],
            "entry_date": t["entry_date"],
            "size_pct": t["size"] * 100,
        }

        if t["status"] == "closed":
            trade_info["exit_price"] = t["exit_price"]
            trade_info["exit_date"] = t["exit_date"]
            trade_info["pnl_pct"] = t.get("pnl_pct", 0)
            if t.get("pnl_dollar") is not None:
                trade_info["pnl_dollar"] = t["pnl_dollar"]
        else:
            if t.get("stop_price"):
                trade_info["stop_price"] = t["stop_price"]
            if t.get("target_price"):
                trade_info["target_price"] = t["target_price"]

        trades_data.append(trade_info)

    return {
        "ticker_filter": ticker.upper() if ticker else None,
        "count": len(trades),
        "trades": trades_data,
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


# ---------------------------------------------------------------------------
# System operations
# ---------------------------------------------------------------------------

def get_health() -> dict:
    """Get model health status."""
    manifest = load_manifest()
    now = datetime.now()

    stale = []
    fresh = []
    for ticker, info in manifest.items():
        try:
            saved_at = datetime.fromisoformat(info["saved_at"])
            age_days = (now - saved_at).days
            if age_days > STALE_MODEL_DAYS:
                stale.append({"ticker": ticker, "age_days": age_days})
            else:
                fresh.append(ticker)
        except (KeyError, ValueError):
            stale.append({"ticker": ticker, "age_days": 999})

    state = get_merged_positions_sync()
    positions = state.get("positions", [])
    total_exposure = sum(p.get("size", 0) for p in positions) if positions else 0

    result = {
        "total_models": len(manifest),
        "fresh_models": len(fresh),
        "stale_models": len(stale),
        "open_positions": len(positions),
        "total_exposure_pct": total_exposure * 100,
    }

    if stale:
        stale_sorted = sorted(stale, key=lambda x: -x["age_days"])[:MAX_STALE_DISPLAY]
        result["stalest_models"] = stale_sorted

    return result


def run_scan(config: str = "conservative") -> dict:
    """Run the market scanner and return results."""
    try:
        logger.info("Running market scan with config: %s", config)
        results = scan_all(tickers=None, config_name=config)

        if not results:
            return {
                "success": False,
                "error": "No models found. Run retrain first.",
            }

        # Persist all scan results for later queries
        state = get_merged_positions_sync()
        positions = state.get("positions", [])
        open_tickers = {p["ticker"] for p in positions}
        save_watchlist(
            [
                {
                    "ticker": r["ticker"],
                    "p_rally": round(r.get("p_rally", 0) * 100, 1),
                    "p_rally_raw": r.get("p_rally_raw", 0),
                    "comp_score": r.get("comp_score", 0),
                    "fail_dn": r.get("fail_dn", 0),
                    "trend": r.get("trend", 0),
                    "golden_cross": r.get("golden_cross", 0),
                    "hmm_compressed": r.get("hmm_compressed", 0),
                    "rv_pctile": r.get("rv_pctile", 0),
                    "atr_pct": r.get("atr_pct", 0),
                    "macd_hist": r.get("macd_hist", 0),
                    "vol_ratio": r.get("vol_ratio", 1),
                    "vix_pctile": r.get("vix_pctile", 0),
                    "rsi": r.get("rsi", 0),
                    "close": r.get("close", 0),
                    "size": r.get("size", 0),
                    "signal": bool(r.get("signal")),
                }
                for r in sorted(results, key=lambda x: x.get("p_rally", 0), reverse=True)
                if r.get("status") == "ok" and r["ticker"] not in open_tickers
            ],
            scan_date=date.today(),
        )

        new_signals = [r for r in results if r.get("signal") and r["ticker"] not in open_tickers]

        current_state = get_merged_positions_sync()
        current_positions = current_state.get("positions", [])
        closed_today = current_state.get("closed_today", [])

        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "success": True,
            "scan_time": scan_time,
            "config": config,
            "tickers_scanned": len(results),
            "new_signals": len(new_signals),
            "existing_open_before_scan": len(positions),
            "total_open_now": len(current_positions),
            "open_positions": [p["ticker"] for p in current_positions],
            "closed_today": len(closed_today),
            "signals": [
                {
                    "ticker": r["ticker"],
                    "p_rally": round(r.get("p_rally", 0) * 100, 1),
                    "comp_score": round(r.get("comp_score", 0), 3),
                    "close": r.get("close", 0),
                    "size_pct": round(r.get("size", 0) * 100, 1),
                    "stop_price": round(r.get("range_low", 0), 2),
                }
                for r in new_signals
            ],
        }
    except Exception as e:
        logger.exception("Scan failed")
        return {
            "success": False,
            "error": str(e),
        }


def run_retrain_marker(tickers: list[str] | None = None) -> dict:
    """Return async-task marker for retrain. No side effects."""
    return {
        "_async_task": "retrain",
        "tickers": tickers,
        "message": "Starting model retraining... This will take 10-30+ minutes. You'll receive progress updates.",  # noqa: E501
    }
