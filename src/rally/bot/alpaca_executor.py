# stdlib
import asyncio
import logging
import os

# third-party
from pydantic import BaseModel

try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import (
        OrderSide,
        OrderStatus,
        QueryOrderStatus,
        TimeInForce,
    )
    from alpaca.trading.requests import (
        GetOrdersRequest,
        MarketOrderRequest,
        TrailingStopOrderRequest,
    )

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# local
from .. import config

logger = logging.getLogger(__name__)


class OrderResult(BaseModel):
    ticker: str
    side: str
    qty: int
    success: bool
    order_id: str | None = None
    fill_price: float | None = None
    trail_order_id: str | None = None
    error: str | None = None


def is_enabled() -> bool:
    return os.environ.get("ALPACA_AUTO_EXECUTE") == "1"


def has_alpaca_keys() -> bool:
    """True if Alpaca API keys are configured (regardless of auto-execute)."""
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


def _trading_client():
    """Create an Alpaca TradingClient."""
    return TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=os.environ.get("ALPACA_PAPER_TRADE", "true").lower() == "true",
    )


def _data_client():
    """Create an Alpaca StockHistoricalDataClient."""
    return StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )


async def get_account_equity() -> float:
    def _sync():
        client = _trading_client()
        account = client.get_account()
        return float(account.equity)

    return await asyncio.to_thread(_sync)


async def get_snapshots(tickers: list[str]) -> dict[str, dict]:
    # Filter to equity tickers only
    equity_tickers = [
        t for t in tickers
        if t in config.ASSETS and config.ASSETS[t].asset_class == "equity"
    ]
    if not equity_tickers:
        return {}

    def _sync():
        client = _data_client()
        snapshots = client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=equity_tickers)
        )
        result: dict[str, dict] = {}
        for ticker in equity_tickers:
            snap = snapshots.get(ticker)
            if not snap:
                continue
            trade = snap.latest_trade
            quote = snap.latest_quote
            result[ticker] = {
                "price": float(trade.price) if trade else 0,
                "bid": float(quote.bid_price) if quote else 0,
                "ask": float(quote.ask_price) if quote else 0,
                "bid_size": float(quote.bid_size) if quote else 0,
                "ask_size": float(quote.ask_size) if quote else 0,
            }
        return result

    return await asyncio.to_thread(_sync)


async def execute_entries(signals: list[dict], equity: float) -> list[OrderResult]:
    from ..trading.portfolio import is_circuit_breaker_active
    from ..trading.positions import get_group_exposure, get_total_exposure, load_positions

    results: list[OrderResult] = []

    # Circuit breaker check
    if is_circuit_breaker_active(equity):
        for sig in signals:
            results.append(OrderResult(
                ticker=sig["ticker"], side="buy", qty=0, success=False,
                error="Circuit breaker active — drawdown exceeds threshold",
            ))
        return results

    current_exposure = get_total_exposure()
    max_exposure = config.PARAMS.max_portfolio_exposure
    open_tickers = {p["ticker"] for p in load_positions().get("positions", [])}
    client = _trading_client()

    for sig in signals:
        ticker = sig["ticker"]

        # Skip tickers we already hold
        if ticker in open_tickers:
            logger.info("Skipping %s: already in open positions", ticker)
            continue

        # Skip crypto
        if ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto":
            continue

        price = sig.get("entry_price") or sig["close"]
        size = sig["size"]

        # Portfolio exposure cap
        if current_exposure + size > max_exposure:
            logger.warning(
                "Skipping %s: exposure %.1f%% + %.1f%% > cap %.0f%%",
                ticker, current_exposure * 100, size * 100, max_exposure * 100,
            )
            results.append(OrderResult(
                ticker=ticker, side="buy", qty=0, success=False,
                error=f"Portfolio exposure cap ({max_exposure:.0%}) exceeded",
            ))
            continue

        # Group concentration check
        group = config.TICKER_TO_GROUP.get(ticker)
        if group:
            g_count, g_exp = get_group_exposure(group)
            if g_count >= config.PARAMS.max_group_positions:
                logger.warning(
                    "Skipping %s: group '%s' at %d/%d positions",
                    ticker, group, g_count, config.PARAMS.max_group_positions,
                )
                results.append(OrderResult(
                    ticker=ticker, side="buy", qty=0, success=False,
                    error=f"Group '{group}' at max positions"
                          f" ({config.PARAMS.max_group_positions})",
                ))
                continue
            if g_exp + size > config.PARAMS.max_group_exposure:
                logger.warning(
                    "Skipping %s: group '%s' exposure %.1f%% + %.1f%% > %.0f%%",
                    ticker, group, g_exp * 100, size * 100,
                    config.PARAMS.max_group_exposure * 100,
                )
                results.append(OrderResult(
                    ticker=ticker, side="buy", qty=0, success=False,
                    error=f"Group '{group}' exposure cap"
                          f" ({config.PARAMS.max_group_exposure:.0%})",
                ))
                continue

        qty = int(equity * size / price)
        if qty < 1:
            results.append(OrderResult(
                ticker=ticker, side="buy", qty=0, success=False,
                error="Insufficient equity for 1 share",
            ))
            continue

        try:
            # Place market buy entry
            order = await asyncio.to_thread(
                client.submit_order,
                MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                ),
            )
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else None

            # Trailing stop is deferred until fill is confirmed
            # (handled by _check_price_alerts in discord_bot.py)
            results.append(OrderResult(
                ticker=ticker, side="buy", qty=qty, success=True,
                order_id=str(order.id),
                fill_price=fill_price,
            ))
            current_exposure += size
        except Exception as e:
            results.append(OrderResult(
                ticker=ticker, side="buy", qty=qty, success=False,
                error=str(e),
            ))

    return results


async def cancel_order(order_id: str) -> bool:
    """Cancel an open order by ID. Returns True if cancelled successfully."""
    try:
        await asyncio.to_thread(
            lambda: _trading_client().cancel_order_by_id(order_id),
        )
        logger.info("Cancelled order %s", order_id)
        return True
    except Exception as e:
        logger.warning("Failed to cancel order %s: %s", order_id, e)
        return False


def _execute_exit_sync(
    client, ticker: str, trail_order_id: str | None = None,
) -> OrderResult:
    """Execute a single exit using an existing client (synchronous)."""
    # Cancel the associated trailing stop before closing
    if trail_order_id:
        try:
            client.cancel_order_by_id(trail_order_id)
            logger.info("Cancelled trailing stop %s for %s", trail_order_id, ticker)
        except Exception as e:
            logger.warning(
                "Could not cancel trailing stop %s for %s: %s",
                trail_order_id, ticker, e,
            )

    order = client.close_position(symbol_or_asset_id=ticker)
    fill_price = float(order.filled_avg_price) if order.filled_avg_price else None
    raw_qty = order.qty or order.filled_qty or 0
    qty = int(float(str(raw_qty)))
    return OrderResult(
        ticker=ticker, side="sell", qty=qty, success=True,
        order_id=str(order.id),
        fill_price=fill_price,
    )


async def execute_exit(ticker: str, trail_order_id: str | None = None) -> OrderResult:
    """Execute a single exit, creating a new client."""
    client = _trading_client()
    return await asyncio.to_thread(
        _execute_exit_sync, client, ticker, trail_order_id,
    )


async def execute_exits(closed: list[dict]) -> list[OrderResult]:
    """Execute multiple exits sharing a single client."""
    results: list[OrderResult] = []
    equity_closed = [
        pos for pos in closed
        if not (pos["ticker"] in config.ASSETS
                and config.ASSETS[pos["ticker"]].asset_class == "crypto")
    ]
    if not equity_closed:
        return results

    client = _trading_client()
    for pos in equity_closed:
        ticker = pos["ticker"]
        trail_oid = pos.get("trail_order_id")
        try:
            result = await asyncio.to_thread(
                _execute_exit_sync, client, ticker, trail_order_id=trail_oid,
            )
            results.append(result)
        except Exception as e:
            results.append(OrderResult(
                ticker=ticker, side="sell", qty=0, success=False,
                error=str(e),
            ))
    return results


async def get_all_positions() -> list[dict]:
    """Fetch all open positions from the broker."""
    def _sync():
        client = _trading_client()
        positions = client.get_all_positions()
        return [
            {
                "ticker": str(p.symbol),
                "qty": int(float(str(p.qty))),
                "avg_entry_price": float(str(p.avg_entry_price)),
                "market_value": float(str(p.market_value)),
                "unrealized_pl": float(str(p.unrealized_pl)),
            }
            for p in positions
        ]

    return await asyncio.to_thread(_sync)


async def place_trailing_stop(
    ticker: str, qty: int, trail_pct: float,
) -> str | None:
    """Place a standalone trailing stop order. Returns trail_order_id or None."""
    def _sync():
        client = _trading_client()
        order = client.submit_order(TrailingStopOrderRequest(
            symbol=ticker,
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
        fills: dict[str, float] = {}
        for order in orders:
            if order.status != OrderStatus.FILLED:
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
