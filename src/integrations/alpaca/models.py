# third-party
from pydantic import BaseModel

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


class EntryPlan(BaseModel):
    """Pre-computed entry order — everything needed to submit without re-checking."""
    ticker: str
    qty: float | int
    size: float
    price: float
    is_crypto: bool
