# stdlib
import os

# third-party
from pydantic import BaseModel

# local
import config

# ---------------------------------------------------------------------------
# Alpaca error code constants (MIC-120)
# ---------------------------------------------------------------------------

_ERR_NOT_TRADABLE = "42210000"     # symbol not supported on Alpaca
_ERR_SHARES_HELD = "40310000"      # shares held by another open order
_ERR_POSITION_CLOSED = "40410000"  # position already closed (trailing stop triggered)


class OrderResult(BaseModel):
    ticker: str
    side: str
    qty: float
    success: bool
    order_id: str | None = None
    fill_price: float | None = None
    trail_order_id: str | None = None
    error: str | None = None
    already_closed: bool = False
    skipped: bool = False  # intentional non-execution (risk cap, non-tradable, etc.)
    # actual allocated fraction (may differ from signal size after partial sizing)
    actual_size: float | None = None


def is_enabled() -> bool:
    return os.environ.get("ALPACA_AUTO_EXECUTE") == "1"


def has_alpaca_keys() -> bool:
    """True if Alpaca API keys are configured (regardless of auto-execute)."""
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


def _alpaca_symbol(ticker: str) -> str:
    """Convert internal ticker key to Alpaca symbol format (e.g. BTC → BTC/USD)."""
    if ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto":
        return config.ASSETS[ticker].ticker.replace("-", "/")
    return ticker
