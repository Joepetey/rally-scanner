# stdlib
import asyncio
import logging
import time

# third-party
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

# local
import rally_ml.config as config

from integrations.alpaca.broker import _safe_qty, _trading_client
from integrations.alpaca.models import (
    _ERR_NOT_TRADABLE,
    _ERR_POSITION_CLOSED,
    _ERR_SHARES_HELD,
    OrderResult,
    _alpaca_symbol,
)

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


def _retry_close_position(
    client: "TradingClient", alpaca_sym: str, ticker: str,
) -> OrderResult:
    """Try up to 3 times to close a position, handling held-shares delays."""
    last_err = None
    for attempt in range(3):
        try:
            order = client.close_position(symbol_or_asset_id=alpaca_sym)
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else None
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
                    "Shares still held for %s, retrying in 1s (attempt %d/3)",
                    ticker, attempt + 1,
                )
                time.sleep(1)
            elif _ERR_POSITION_CLOSED in err_str:
                logger.info(
                    "Position %s already closed in Alpaca (trailing stop triggered)", ticker,
                )
                return OrderResult(
                    ticker=ticker, side="sell", qty=0, success=True, already_closed=True,
                )
            else:
                raise
    raise last_err  # unreachable, but satisfies type checker


def _recover_from_shares_held(
    client: "TradingClient", ticker: str, err_str: str,
) -> tuple[str | None, str | None]:
    """Recover existing OCO order IDs from a held-shares (_ERR_SHARES_HELD) error body."""
    import json as _json
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


def _execute_exit_sync(
    client: "TradingClient", ticker: str, trail_order_id: str | None = None,
    target_order_id: str | None = None,
) -> OrderResult:
    """Execute a single exit using an existing client (synchronous)."""
    # Cancel OCO exit orders before closing. OCO orders hold shares until cancelled —
    # skipping this causes close_position() to fail with _ERR_SHARES_HELD.
    # Cancelling one OCO leg auto-cancels the other on Alpaca, but we cancel both
    # explicitly for robustness. target_order_id is the limit-sell (parent); cancelling
    # it also releases the stop leg even if trail_order_id is stale or missing.
    order_ids_to_cancel = {oid for oid in (trail_order_id, target_order_id) if oid}
    if order_ids_to_cancel:
        for oid in order_ids_to_cancel:
            try:
                client.cancel_order_by_id(oid)
                logger.info("Cancelled OCO order %s for %s", oid, ticker)
            except Exception as e:
                if _ERR_NOT_TRADABLE in str(e):
                    logger.debug("Order %s for %s already filled, cancel skipped", oid, ticker)
                else:
                    logger.warning("Could not cancel order %s for %s: %s", oid, ticker, e)
        # Wait for Alpaca to release the held shares after cancellation
        time.sleep(0.5)

    # close_position puts the symbol in the URL path — slashes cause a 404 because
    # the SDK doesn't URL-encode them. Alpaca stores crypto positions as "BTCUSD"
    # (no slash), so strip "/" for the close call. Equity symbols are unaffected.
    alpaca_sym = _alpaca_symbol(ticker).replace("/", "")
    return _retry_close_position(client, alpaca_sym, ticker)


async def execute_exit(ticker: str, trail_order_id: str | None = None) -> OrderResult:
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


async def place_exit_orders(
    ticker: str, qty: float, target_price: float, stop_price: float,
) -> tuple[str | None, str | None]:
    """Place an OCO exit: GTC limit sell at target + stop sell at stop_price.

    Uses Alpaca's native OCO so both legs share qty — when one fills, Alpaca
    automatically cancels the other.

    Returns (target_order_id, stop_order_id) where both IDs refer to the two
    legs of the OCO order. Either may be None on failure.

    Note: Alpaca does not support OCO orders for crypto assets. Crypto positions
    are managed entirely through the monitoring loop (stream price events →
    stop/target breach detection → market close).
    """
    is_crypto = ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto"
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
                take_profit=TakeProfitRequest(limit_price=round(target_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
            ))
            # Parent order IS the limit (target) leg; child leg is the stop
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
                # Shares held by existing OCO — recover its IDs instead of retrying placement.
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
