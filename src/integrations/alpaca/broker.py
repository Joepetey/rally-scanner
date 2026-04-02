# stdlib
import asyncio
import os
from contextlib import contextmanager

# third-party
try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# local
import rally_ml.config as config


def is_enabled() -> bool:
    """True if Alpaca auto-execution is turned on."""
    return os.environ.get("ALPACA_AUTO_EXECUTE") == "1"


def has_alpaca_keys() -> bool:
    """True if Alpaca API keys are configured (regardless of auto-execute)."""
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


def _alpaca_symbol(ticker: str) -> str:
    """Convert internal ticker key to Alpaca symbol format (e.g. BTC → BTC/USD)."""
    if ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto":
        return config.ASSETS[ticker].ticker.replace("-", "/")
    return ticker


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


@contextmanager
def simulation_keys():
    """Temporarily swap ALPACA_API_KEY/SECRET_KEY to the simulation account.

    Raises RuntimeError if ALPACA_SIMULATION_API_KEY is not set.
    """
    sim_key = os.environ.get("ALPACA_SIMULATION_API_KEY")
    sim_secret = os.environ.get("ALPACA_SIMULATION_SECRET_KEY")
    if not sim_key or not sim_secret:
        raise RuntimeError(
            "ALPACA_SIMULATION_API_KEY and ALPACA_SIMULATION_SECRET_KEY must be set"
        )

    orig_key = os.environ["ALPACA_API_KEY"]
    orig_secret = os.environ["ALPACA_SECRET_KEY"]
    os.environ["ALPACA_API_KEY"] = sim_key
    os.environ["ALPACA_SECRET_KEY"] = sim_secret
    try:
        yield
    finally:
        os.environ["ALPACA_API_KEY"] = orig_key
        os.environ["ALPACA_SECRET_KEY"] = orig_secret
