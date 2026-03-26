# stdlib
import asyncio
import logging
import os

# third-party
try:
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
except ImportError:
    pass

# local
from config import PARAMS
from integrations.alpaca.broker import _data_client, _trading_client
from integrations.alpaca.executor import OrderResult

logger = logging.getLogger(__name__)

_TICKER = "SGOV"


def is_enabled() -> bool:
    return os.environ.get("SGOV_ENABLED") == "1"


async def get_sgov_price() -> float:
    """Fetch current SGOV price via Alpaca snapshot. Returns 0.0 on failure."""
    try:
        def _sync() -> float:
            client = _data_client()
            req = StockSnapshotRequest(symbol_or_symbols=[_TICKER])
            snapshots = client.get_stock_snapshot(req)
            snap = snapshots.get(_TICKER)
            if snap is None:
                return 0.0
            latest = snap.latest_trade
            return float(latest.price) if latest else 0.0

        return await asyncio.to_thread(_sync)
    except Exception:
        logger.exception("Failed to fetch SGOV price")
        return 0.0


async def get_sgov_qty() -> int:
    """Return current SGOV share count from Alpaca. Returns 0 if no position."""
    def _sync() -> int:
        client = _trading_client()
        try:
            pos = client.get_open_position(_TICKER)
            return int(pos.qty)
        except Exception as exc:
            # Alpaca raises when position doesn't exist
            if "position does not exist" in str(exc).lower() or "404" in str(exc):
                return 0
            raise

    return await asyncio.to_thread(_sync)


async def buy_sgov(equity: float, idle_fraction: float) -> OrderResult | None:
    """Buy SGOV shares to park idle capital.

    Returns None if the idle fraction is below the minimum threshold or the
    computed qty rounds to zero.
    """
    if idle_fraction < PARAMS.sgov_min_idle_fraction:
        return None

    price = await get_sgov_price()
    if price <= 0.0:
        return None

    qty = int(equity * idle_fraction / price)
    if qty < 1:
        return None

    def _sync() -> OrderResult:
        client = _trading_client()
        req = MarketOrderRequest(
            symbol=_TICKER,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return OrderResult(
            ticker=_TICKER,
            side="buy",
            qty=qty,
            success=True,
            order_id=str(order.id),
        )

    try:
        result = await asyncio.to_thread(_sync)
        logger.info("Parked %d SGOV shares (idle=%.1f%%)", qty, idle_fraction * 100)
        return result
    except Exception as exc:
        logger.exception("Failed to buy SGOV")
        return OrderResult(ticker=_TICKER, side="buy", qty=qty, success=False, error=str(exc))


async def sell_sgov(equity: float, gap_fraction: float) -> OrderResult | None:
    """Sell enough SGOV shares to cover gap_fraction of equity.

    Only sells the minimum needed — capped at the actual qty held.
    Returns None if no SGOV is held or gap is zero.
    """
    if gap_fraction <= 0:
        return None

    qty_held = await get_sgov_qty()
    if qty_held == 0:
        return None

    price = await get_sgov_price()
    if price <= 0:
        return None

    # Add 1 share as rounding buffer so we don't come up fractionally short
    qty_to_sell = min(int(gap_fraction * equity / price) + 1, qty_held)

    def _sync() -> OrderResult:
        client = _trading_client()
        req = MarketOrderRequest(
            symbol=_TICKER,
            qty=qty_to_sell,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return OrderResult(
            ticker=_TICKER,
            side="sell",
            qty=qty_to_sell,
            success=True,
            order_id=str(order.id),
        )

    try:
        result = await asyncio.to_thread(_sync)
        logger.info("Sold %d/%d SGOV shares (gap=%.1f%%)", qty_to_sell, qty_held, gap_fraction * 100)
        return result
    except Exception as exc:
        logger.exception("Failed to sell SGOV")
        return OrderResult(ticker=_TICKER, side="sell", qty=qty_to_sell, success=False, error=str(exc))
