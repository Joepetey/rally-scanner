"""OCO order placement, cancellation, and trailing stop management."""

import asyncio
import json as _json
import logging

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderClass as AlpacaOrderClass
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
    from alpaca.trading.requests import (
        OrderRequest,
        StopLossRequest,
        TakeProfitRequest,
        TrailingStopOrderRequest,
    )

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

import rally_ml.config as config

from integrations.alpaca.broker import _alpaca_symbol, _trading_client
from integrations.alpaca.models import _ERR_NOT_TRADABLE, _ERR_SHARES_HELD

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
        if _ERR_NOT_TRADABLE in str(e):
            logger.debug("Order %s already filled, cancel skipped", order_id)
        else:
            logger.warning("Failed to cancel order %s: %s", order_id, e)
        return False


def _recover_from_shares_held(
    client: "TradingClient", ticker: str, err_str: str,
) -> tuple[str | None, str | None]:
    """Recover existing OCO order IDs from a held-shares error body."""
    try:
        body_start = err_str.find("{")
        body = _json.loads(err_str[body_start:]) if body_start >= 0 else {}
        related = body.get("related_orders", [])
        if related:
            parent_oid = str(related[0])
            order = client.get_order_by_id(parent_oid)
            legs = order.legs or []
            target_oid = str(order.id)
            stop_oid = str(legs[0].id) if legs else target_oid
            logger.info(
                "Recovered existing OCO for %s from held-shares error: "
                "target_order=%s stop_order=%s",
                ticker, target_oid, stop_oid,
            )
            return target_oid, stop_oid
    except Exception as recover_err:
        logger.error(
            "Could not recover OCO IDs for %s after %s: %s",
            ticker, _ERR_SHARES_HELD, recover_err,
        )
    return None, None


async def place_exit_orders(
    ticker: str, qty: float, target_price: float, stop_price: float,
) -> tuple[str | None, str | None]:
    """Place an OCO exit: GTC limit sell at target + stop sell at stop_price.

    Returns (target_order_id, stop_order_id). Either may be None on failure.
    Crypto positions are skipped (Alpaca doesn't support OCO for crypto).
    """
    is_crypto = (
        ticker in config.ASSETS
        and config.ASSETS[ticker].asset_class == "crypto"
    )
    if is_crypto:
        logger.info(
            "Skipping OCO exit for %s: Alpaca does not support OCO for crypto "
            "(monitoring loop handles exits)",
            ticker,
        )
        return None, None

    def _sync():
        client = _trading_client()
        try:
            parent = client.submit_order(OrderRequest(
                symbol=_alpaca_symbol(ticker),
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                order_class=AlpacaOrderClass.OCO,
                take_profit=TakeProfitRequest(
                    limit_price=round(target_price, 2),
                ),
                stop_loss=StopLossRequest(
                    stop_price=round(stop_price, 2),
                ),
            ))
            legs = parent.legs or []
            target_oid = str(parent.id)
            stop_oid = str(legs[0].id) if legs else target_oid
            logger.info(
                "OCO exit placed for %s: target=$%.2f stop=$%.2f "
                "target_order=%s stop_order=%s",
                ticker, target_price, stop_price, target_oid, stop_oid,
            )
            return target_oid, stop_oid
        except Exception as e:
            err_str = str(e)
            if _ERR_SHARES_HELD in err_str:
                return _recover_from_shares_held(client, ticker, err_str)
            logger.error("Failed to place OCO exit for %s: %s", ticker, e)
            return None, None

    return await asyncio.to_thread(_sync)


async def cancel_exit_orders(
    target_order_id: str | None, stop_order_id: str | None,
) -> None:
    """Cancel OCO exit orders. Cancels each unique ID once; ignores errors."""
    ids = {oid for oid in (target_order_id, stop_order_id) if oid}
    await asyncio.gather(*[cancel_order(oid) for oid in ids])


async def place_trailing_stop(
    ticker: str, qty: float, trail_pct: float,
) -> str | None:
    """Place a standalone trailing stop order. Returns trail_order_id or None."""
    def _sync():
        client = _trading_client()
        order = client.submit_order(TrailingStopOrderRequest(
            symbol=_alpaca_symbol(ticker),
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            trail_percent=trail_pct,
        ))
        return str(order.id)

    try:
        trail_id = await asyncio.to_thread(_sync)
        logger.info(
            "Trailing stop placed for %s: %.1f%% trail, order %s",
            ticker, trail_pct, trail_id,
        )
        return trail_id
    except Exception as e:
        logger.warning("Failed to place trailing stop for %s: %s", ticker, e)
        return None
