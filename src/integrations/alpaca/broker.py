# stdlib
import asyncio
import os

# third-party
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


def _trading_client() -> "TradingClient":
    """Create an Alpaca TradingClient."""
    return TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=os.environ.get("ALPACA_PAPER_TRADE", "true").lower() == "true",
    )


def _data_client() -> "StockHistoricalDataClient":
    """Create an Alpaca StockHistoricalDataClient."""
    return StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )


def _safe_qty(raw) -> float:
    """Convert Alpaca's qty field (may be str, float, or Decimal) to float."""
    return float(str(raw))


def _normalize_alpaca_symbol(symbol: str) -> str:
    """Convert Alpaca symbol to internal ticker key (e.g. BTC/USD → BTC)."""
    if "/" in symbol:
        return symbol.split("/")[0]
    return symbol


async def get_all_positions() -> list[dict]:
    """Fetch all open positions from the broker."""
    def _sync() -> list[dict]:
        client = _trading_client()
        positions = client.get_all_positions()
        return [
            {
                "ticker": _normalize_alpaca_symbol(str(p.symbol)),
                "qty": _safe_qty(p.qty),
                "avg_entry_price": float(str(p.avg_entry_price)),
                "market_value": float(str(p.market_value)),
                "unrealized_pl": float(str(p.unrealized_pl)),
            }
            for p in positions
        ]

    return await asyncio.to_thread(_sync)
