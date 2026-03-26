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


def _safe_qty(raw) -> int:
    """Convert Alpaca's qty field (may be str, float, or Decimal) to int."""
    return int(float(str(raw)))


async def get_all_positions() -> list[dict]:
    """Fetch all open positions from the broker."""
    def _sync() -> list[dict]:
        client = _trading_client()
        positions = client.get_all_positions()
        return [
            {
                "ticker": str(p.symbol),
                "qty": _safe_qty(p.qty),
                "avg_entry_price": float(str(p.avg_entry_price)),
                "market_value": float(str(p.market_value)),
                "unrealized_pl": float(str(p.unrealized_pl)),
            }
            for p in positions
        ]

    return await asyncio.to_thread(_sync)
