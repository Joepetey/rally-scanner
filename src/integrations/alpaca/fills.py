# stdlib
import asyncio
import logging
from datetime import UTC, datetime, timedelta

# third-party
try:
    from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# local
from integrations.alpaca.broker import _trading_client
from integrations.alpaca.models import _ERR_NOT_TRADABLE

logger = logging.getLogger(__name__)


async def cancel_order(order_id: str) -> bool:
    """Cancel an open order by ID. Returns True if cancelled successfully."""
    try:
        await asyncio.to_thread(
            lambda: _trading_client().cancel_order_by_id(order_id),
        )
        logger.info("Cancelled order %s", order_id)
        return True
    except Exception as e:
        # 42210000: order already filled — benign, the exit we wanted happened
        if _ERR_NOT_TRADABLE in str(e):
            logger.debug("Order %s already filled, cancel skipped", order_id)
        else:
            logger.warning("Failed to cancel order %s: %s", order_id, e)
        return False


async def cancel_exit_orders(
    target_order_id: str | None, stop_order_id: str | None,
) -> None:
    """Cancel OCO exit orders. Cancels each unique ID once; ignores errors."""
    ids = {oid for oid in (target_order_id, stop_order_id) if oid}
    await asyncio.gather(*[cancel_order(oid) for oid in ids])


async def check_exit_fills(
    positions: list[dict],
) -> list[dict]:
    """Check whether any exit orders (target or stop) have filled on the broker.

    Args:
        positions: list of position dicts with 'ticker', 'target_order_id', 'trail_order_id'

    Returns:
        list of {ticker, fill_price, exit_reason} for positions whose exit order filled.
    """
    order_ids: dict[str, tuple[str, str]] = {}  # order_id -> (ticker, "target"|"stop")
    for pos in positions:
        t_oid = pos.get("target_order_id")
        s_oid = pos.get("trail_order_id")
        ticker = pos["ticker"]
        if t_oid:
            order_ids[t_oid] = (ticker, "profit_target")
        if s_oid:
            order_ids[s_oid] = (ticker, "stop")

    if not order_ids:
        return []

    def _sync():
        client = _trading_client()
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
        ))
        filled = []
        for order in orders:
            oid = str(order.id)
            if oid not in order_ids:
                continue
            if order.status != OrderStatus.FILLED:
                continue
            ticker, reason = order_ids[oid]
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0
            filled.append({"ticker": ticker, "fill_price": fill_price, "exit_reason": reason})
        return filled

    return await asyncio.to_thread(_sync)


async def check_pending_fills(order_ids: list[str]) -> dict[str, float]:
    """Check which pending orders have filled. Returns {order_id: fill_price}."""
    if not order_ids:
        return {}

    def _sync():
        client = _trading_client()
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
        ))
        id_set = set(order_ids)
        today = datetime.now(tz=UTC).date()
        fills: dict[str, float] = {}
        for order in orders:
            if order.status != OrderStatus.FILLED:
                continue
            if order.filled_at and order.filled_at.date() != today:
                continue
            oid = str(order.id)
            if oid in id_set and order.filled_avg_price:
                fills[oid] = float(order.filled_avg_price)
        return fills

    return await asyncio.to_thread(_sync)


async def check_trail_stop_fills(
    trail_order_ids: dict[str, str],
) -> dict[str, float]:
    """Check if any trailing stop orders have filled.

    Args:
        trail_order_ids: {ticker: trail_order_id} mapping

    Returns:
        {ticker: fill_price} for filled trailing stops
    """
    if not trail_order_ids:
        return {}

    def _sync():
        client = _trading_client()
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
        ))
        oid_to_ticker = {oid: t for t, oid in trail_order_ids.items()}
        fills: dict[str, float] = {}
        for order in orders:
            if order.status != OrderStatus.FILLED:
                continue
            oid = str(order.id)
            if oid in oid_to_ticker and order.filled_avg_price:
                fills[oid_to_ticker[oid]] = float(order.filled_avg_price)
        return fills

    return await asyncio.to_thread(_sync)


async def get_recent_sell_fills(tickers: list[str]) -> dict[str, float]:
    """Return {ticker: fill_price} for sell orders filled in the last 24h."""
    if not tickers:
        return {}

    def _sync() -> dict[str, float]:
        client = _trading_client()
        after = datetime.now(UTC) - timedelta(hours=24)
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=after,
        ))
        tickers_set = set(tickers)
        fills: dict[str, float] = {}
        for order in orders:
            if order.status != OrderStatus.FILLED:
                continue
            if order.side != OrderSide.SELL:
                continue
            raw_sym = str(order.symbol)
            # Normalize Alpaca crypto symbols (BTC/USD → BTC)
            sym = raw_sym.split("/")[0] if "/" in raw_sym else raw_sym
            if sym in tickers_set and order.filled_avg_price:
                fills[sym] = float(order.filled_avg_price)
        return fills

    return await asyncio.to_thread(_sync)
