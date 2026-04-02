"""Trade management operations: enter, exit, history."""

from db.trades import (
    close_trade,
    open_trade,
)
from db.trades import (
    get_trade_history as _db_get_trade_history,
)
from services._helpers import dollar_metrics
from trading.positions import get_merged_positions_sync

DEFAULT_TRADE_LIMIT = 10


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
