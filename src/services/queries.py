"""Read-only query operations: signals, positions, watchlist, prices."""

from rally_ml.core.data import fetch_quotes

from db.trading.positions import load_watchlist
from db.trading.trades import get_open_trades
from services._helpers import dollar_metrics
from trading.positions import get_merged_positions_sync

WATCHLIST_CAP = 20
MAX_PRICE_TICKERS = 10


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
