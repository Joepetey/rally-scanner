"""Position exit execution with retry logic."""

import asyncio
import logging
import re
import time

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import ClosePositionRequest

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

from integrations.alpaca.broker import _alpaca_symbol, _safe_qty, _trading_client
from integrations.alpaca.models import (
    _ERR_NOT_TRADABLE,
    _ERR_POSITION_CLOSED,
    _ERR_SHARES_HELD,
    OrderResult,
)

_INSUFFICIENT_QTY_RE = re.compile(r"insufficient qty available.*available:\s*(\d+)")

# Re-export order management for backwards compatibility
from integrations.alpaca.orders import (  # noqa: F401
    cancel_exit_orders,
    cancel_order,
    place_exit_orders,
    place_trailing_stop,
)

logger = logging.getLogger(__name__)

# Named constants for retry delays
_CANCEL_SETTLE_SECONDS = 0.5
_RETRY_DELAY_SECONDS = 1.0


def _retry_close_position(
    client: "TradingClient", alpaca_sym: str, ticker: str,
) -> OrderResult:
    """Try up to 3 times to close a position, handling held-shares delays."""
    last_err = None
    for attempt in range(3):
        try:
            order = client.close_position(symbol_or_asset_id=alpaca_sym)
            fill_price = (
                float(order.filled_avg_price)
                if order.filled_avg_price
                else None
            )
            raw_qty = order.qty or order.filled_qty or 0
            qty = _safe_qty(raw_qty)
            return OrderResult(
                ticker=ticker, side="sell", qty=qty, success=True,
                order_id=str(order.id), fill_price=fill_price,
            )
        except Exception as e:
            last_err = e
            err_str = str(e)
            if _ERR_SHARES_HELD in err_str and attempt < 2:
                logger.info(
                    "Shares still held for %s, retrying in %.0fs (attempt %d/3)",
                    ticker, _RETRY_DELAY_SECONDS, attempt + 1,
                )
                time.sleep(_RETRY_DELAY_SECONDS)
            elif _ERR_POSITION_CLOSED in err_str:
                logger.info(
                    "Position %s already closed in Alpaca"
                    " (trailing stop triggered)", ticker,
                )
                return OrderResult(
                    ticker=ticker, side="sell", qty=0,
                    success=True, already_closed=True,
                )
            elif (m := _INSUFFICIENT_QTY_RE.search(err_str)):
                avail = int(m.group(1))
                if avail > 0:
                    logger.warning(
                        "Partial close for %s: only %d shares available",
                        ticker, avail,
                    )
                    order = client.close_position(
                        symbol_or_asset_id=alpaca_sym,
                        close_options=ClosePositionRequest(qty=str(avail)),
                    )
                    fill_price = (
                        float(order.filled_avg_price)
                        if order.filled_avg_price
                        else None
                    )
                    return OrderResult(
                        ticker=ticker, side="sell", qty=_safe_qty(avail),
                        success=True, order_id=str(order.id),
                        fill_price=fill_price,
                    )
                raise
            else:
                raise
    raise last_err  # unreachable, but satisfies type checker


def _execute_exit_sync(
    client: "TradingClient", ticker: str,
    trail_order_id: str | None = None,
    target_order_id: str | None = None,
) -> OrderResult:
    """Execute a single exit using an existing client (synchronous)."""
    order_ids_to_cancel = {
        oid for oid in (trail_order_id, target_order_id) if oid
    }
    if order_ids_to_cancel:
        for oid in order_ids_to_cancel:
            try:
                client.cancel_order_by_id(oid)
                logger.info("Cancelled OCO order %s for %s", oid, ticker)
            except Exception as e:
                if _ERR_NOT_TRADABLE in str(e):
                    logger.debug(
                        "Order %s for %s already filled, cancel skipped",
                        oid, ticker,
                    )
                else:
                    logger.warning(
                        "Could not cancel order %s for %s: %s",
                        oid, ticker, e,
                    )
        time.sleep(_CANCEL_SETTLE_SECONDS)

    alpaca_sym = _alpaca_symbol(ticker).replace("/", "")
    return _retry_close_position(client, alpaca_sym, ticker)


async def execute_exit(
    ticker: str, trail_order_id: str | None = None,
) -> OrderResult:
    """Execute a single exit, creating a new client."""
    client = _trading_client()
    return await asyncio.to_thread(
        _execute_exit_sync, client, ticker, trail_order_id,
    )


async def execute_exits(closed: list[dict]) -> list[OrderResult]:
    """Execute multiple exits sharing a single client."""
    results: list[OrderResult] = []
    if not closed:
        return results

    client = _trading_client()
    for pos in closed:
        ticker = pos["ticker"]
        trail_oid = pos.get("trail_order_id")
        target_oid = pos.get("target_order_id")
        try:
            result = await asyncio.to_thread(
                _execute_exit_sync, client, ticker,
                trail_order_id=trail_oid, target_order_id=target_oid,
            )
            results.append(result)
        except Exception as e:
            results.append(OrderResult(
                ticker=ticker, side="sell", qty=0, success=False,
                error=str(e),
            ))
    return results
