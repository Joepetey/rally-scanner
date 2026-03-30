# stdlib
import asyncio
import logging
import os
import time
from datetime import UTC, datetime, timedelta

# third-party
from pydantic import BaseModel

try:
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderClass as AlpacaOrderClass
    from alpaca.trading.enums import (
        OrderSide,
        OrderStatus,
        OrderType,
        QueryOrderStatus,
        TimeInForce,
    )
    from alpaca.trading.requests import (
        GetOrdersRequest,
        MarketOrderRequest,
        OrderRequest,
        StopLossRequest,
        TakeProfitRequest,
        TrailingStopOrderRequest,
    )

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# local
import config
from core.data import fetch_quotes
from db.positions import (
    enqueue_signal,
    load_all_position_meta,
    load_positions,
    log_skipped_signal,
    remove_from_queue,
)
from integrations.alpaca.broker import _data_client, _safe_qty, _trading_client
from trading.positions import async_close_position, get_group_exposure, get_total_exposure
from trading.risk_manager import is_circuit_breaker_active

logger = logging.getLogger(__name__)


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


async def get_account_equity() -> float:
    def _sync():
        client = _trading_client()
        account = client.get_account()
        return float(account.equity)

    return await asyncio.to_thread(_sync)


async def get_all_positions() -> list[dict]:
    def _sync():
        client = _trading_client()
        positions = client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    return await asyncio.to_thread(_sync)


async def get_snapshots(tickers: list[str]) -> dict[str, dict]:
    equity_tickers = [
        t for t in tickers
        if t in config.ASSETS and config.ASSETS[t].asset_class == "equity"
    ]
    crypto_tickers = [
        t for t in tickers
        if t in config.ASSETS and config.ASSETS[t].asset_class == "crypto"
    ]

    result: dict[str, dict] = {}

    if equity_tickers:
        def _sync():
            client = _data_client()
            snapshots = client.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=equity_tickers)
            )
            equity_result: dict[str, dict] = {}
            for ticker in equity_tickers:
                snap = snapshots.get(ticker)
                if not snap:
                    continue
                trade = snap.latest_trade
                quote = snap.latest_quote
                equity_result[ticker] = {
                    "price": float(trade.price) if trade else 0,
                    "bid": float(quote.bid_price) if quote else 0,
                    "ask": float(quote.ask_price) if quote else 0,
                    "bid_size": float(quote.bid_size) if quote else 0,
                    "ask_size": float(quote.ask_size) if quote else 0,
                }
            return equity_result

        result.update(await asyncio.to_thread(_sync))

    if crypto_tickers:
        # Use yfinance for crypto (BTC-USD format)
        yf_tickers = [config.ASSETS[t].ticker for t in crypto_tickers]
        yf_data = await asyncio.to_thread(fetch_quotes, yf_tickers)
        for t in crypto_tickers:
            yf_ticker = config.ASSETS[t].ticker.upper()
            data = yf_data.get(yf_ticker)
            if data and "error" not in data:
                result[t] = {
                    "price": data["price"],
                    "bid": 0,
                    "ask": 0,
                    "bid_size": 0,
                    "ask_size": 0,
                }

    return result



def _check_group_constraints(
    ticker: str, size: float, sig: dict,
) -> OrderResult | None:
    """Check group position count and exposure limits. Returns skip result or None."""
    group = config.TICKER_TO_GROUP.get(ticker)
    if not group:
        return None

    g_count, g_exp = get_group_exposure(group)
    if g_count >= config.PARAMS.max_group_positions:
        logger.warning(
            "Skipping %s: group '%s' at %d/%d positions",
            ticker, group, g_count, config.PARAMS.max_group_positions,
        )
        log_skipped_signal(sig, f"group_cap:{group}")
        return OrderResult(
            ticker=ticker, side="buy", qty=0, success=False, skipped=True,
            error=f"Group '{group}' at max positions"
                  f" ({config.PARAMS.max_group_positions})",
        )
    if g_exp + size > config.PARAMS.max_group_exposure:
        logger.warning(
            "Skipping %s: group '%s' exposure %.1f%% + %.1f%% > %.0f%%",
            ticker, group, g_exp * 100, size * 100,
            config.PARAMS.max_group_exposure * 100,
        )
        log_skipped_signal(sig, f"group_exposure:{group}")
        return OrderResult(
            ticker=ticker, side="buy", qty=0, success=False, skipped=True,
            error=f"Group '{group}' exposure cap"
                  f" ({config.PARAMS.max_group_exposure:.0%})",
        )
    return None


async def _attempt_rotation(
    sig: dict,
    open_positions: list[dict],
    open_tickers: set[str],
    current_exposure: float,
    max_exposure: float,
    min_size: float,
    results: list[OrderResult],
) -> tuple[float, float, list[dict], set[str], bool]:
    """Try to rotate out the weakest position to fund a new entry.

    Returns (size, current_exposure, open_positions, open_tickers, rotated).
    """
    ticker = sig["ticker"]
    size = sig["size"]
    sig_p = sig.get("p_rally", 0)
    weakest = min(open_positions, key=lambda p: p.get("p_rally", 0))

    if sig_p <= weakest.get("p_rally", 0) + config.PARAMS.rotation_p_rally_margin:
        return size, current_exposure, open_positions, open_tickers, False

    wticker = weakest["ticker"]
    logger.info(
        "Rotation: closing %s (P=%.0f%%) to fund %s (P=%.0f%%)",
        wticker, weakest.get("p_rally", 0) * 100, ticker, sig_p * 100,
    )
    try:
        exit_result = await execute_exit(
            wticker, trail_order_id=weakest.get("trail_order_id"),
        )
        if exit_result.success or exit_result.already_closed:
            w_price = exit_result.fill_price or weakest.get("current_price", 0)
            await async_close_position(wticker, w_price, "rotation")
            freed = weakest.get("size", 0)
            current_exposure -= freed
            available = max_exposure - current_exposure
            open_positions = [p for p in open_positions if p["ticker"] != wticker]
            open_tickers.discard(wticker)
            size = min(size, available) if available >= min_size else 0.0
            if size >= min_size:
                results.append(OrderResult(
                    ticker=wticker, side="sell",
                    qty=exit_result.qty, success=True,
                    order_id=exit_result.order_id,
                    fill_price=exit_result.fill_price,
                ))
                return size, current_exposure, open_positions, open_tickers, True
        else:
            logger.warning("Rotation exit failed for %s: %s", wticker, exit_result.error)
    except Exception as e:
        logger.warning("Rotation exit error for %s: %s", wticker, e)

    return size, current_exposure, open_positions, open_tickers, False


async def execute_entries(signals: list[dict], equity: float) -> list[OrderResult]:
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
    min_size = config.PARAMS.min_position_size
    open_positions = load_all_position_meta()
    open_tickers = {p["ticker"] for p in load_positions().get("positions", [])}
    client = _trading_client()

    for sig in signals:
        ticker = sig["ticker"]

        # Skip tickers we already hold
        if ticker in open_tickers:
            logger.info("Skipping %s: already in open positions", ticker)
            continue

        is_crypto = ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto"
        price = sig.get("entry_price") or sig["close"]
        size = sig["size"]
        available = max_exposure - current_exposure

        # --- Partial sizing: scale down to available capital ---
        if current_exposure + size > max_exposure:
            if config.PARAMS.partial_sizing_enabled and available >= min_size:
                logger.info(
                    "%s: partial size %.1f%% → %.1f%% (capital limited)",
                    ticker, size * 100, available * 100,
                )
                size = available
            else:
                # --- Position rotation: exit weakest to fund stronger signal ---
                rotated = False
                if config.PARAMS.rotation_enabled and open_positions:
                    size, current_exposure, open_positions, open_tickers, rotated = (
                        await _attempt_rotation(
                            sig, open_positions, open_tickers,
                            current_exposure, max_exposure, min_size, results,
                        )
                    )

                if not rotated:
                    enqueue_signal(sig, "capital")
                    log_skipped_signal(sig, "capital")
                    logger.info("Queued %s: no capital available", ticker)
                    results.append(OrderResult(
                        ticker=ticker, side="buy", qty=0, success=False, skipped=True,
                        error=f"Portfolio exposure cap ({max_exposure:.0%}) exceeded — queued",
                    ))
                    continue

        # Group concentration check
        group_skip = _check_group_constraints(ticker, size, sig)
        if group_skip:
            results.append(group_skip)
            continue

        # Crypto supports fractional shares; equity requires whole shares
        if is_crypto:
            qty = round(equity * size / price, 8)
            if qty <= 0:
                results.append(OrderResult(
                    ticker=ticker, side="buy", qty=0, success=False,
                    error="Insufficient equity for fractional crypto order",
                ))
                continue
        else:
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
                    symbol=_alpaca_symbol(ticker),
                    qty=qty,
                    side=OrderSide.BUY,
                    # Crypto trades 24/7 — use GTC; equities use DAY
                    time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY,
                ),
            )
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else None

            # Remove from queue if it was waiting
            remove_from_queue(ticker)

            # Trailing stop is deferred until fill is confirmed
            # (handled by _check_price_alerts in discord_bot.py)
            results.append(OrderResult(
                ticker=ticker, side="buy", qty=qty, success=True,
                order_id=str(order.id),
                fill_price=fill_price,
                actual_size=size,
            ))
            current_exposure += size
        except Exception as e:
            err_str = str(e)
            if "42210000" in err_str:
                logger.info("Skipping %s: not tradable on Alpaca", ticker)
                results.append(OrderResult(
                    ticker=ticker, side="buy", qty=0, success=False, skipped=True,
                    error=err_str,
                ))
            else:
                results.append(OrderResult(
                    ticker=ticker, side="buy", qty=qty, success=False,
                    error=err_str,
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
        # 42210000: order already filled — benign, the exit we wanted happened
        if "42210000" in str(e):
            logger.debug("Order %s already filled, cancel skipped", order_id)
        else:
            logger.warning("Failed to cancel order %s: %s", order_id, e)
        return False


def _execute_exit_sync(
    client: "TradingClient", ticker: str, trail_order_id: str | None = None,
    target_order_id: str | None = None,
) -> OrderResult:
    """Execute a single exit using an existing client (synchronous)."""
    # Cancel OCO exit orders before closing. OCO orders hold shares until cancelled —
    # skipping this causes close_position() to fail with error 40310000 (shares held).
    # Cancelling one OCO leg auto-cancels the other on Alpaca, but we cancel both
    # explicitly for robustness. target_order_id is the limit-sell (parent); cancelling
    # it also releases the stop leg even if trail_order_id is stale or missing.
    order_ids_to_cancel = {
        oid for oid in (trail_order_id, target_order_id) if oid
    }
    if order_ids_to_cancel:
        for oid in order_ids_to_cancel:
            try:
                client.cancel_order_by_id(oid)
                logger.info("Cancelled OCO order %s for %s", oid, ticker)
            except Exception as e:
                if "42210000" in str(e):
                    logger.debug("Order %s for %s already filled, cancel skipped", oid, ticker)
                else:
                    logger.warning("Could not cancel order %s for %s: %s", oid, ticker, e)
        # Wait for Alpaca to release the held shares after cancellation
        time.sleep(0.5)

    # close_position puts the symbol in the URL path — slashes cause a 404 because
    # the SDK doesn't URL-encode them. Alpaca stores crypto positions as "BTCUSD"
    # (no slash), so strip "/" for the close call. Equity symbols are unaffected.
    alpaca_sym = _alpaca_symbol(ticker).replace("/", "")

    # Retry close in case cancel hasn't fully released held shares
    last_err = None
    for attempt in range(3):
        try:
            order = client.close_position(symbol_or_asset_id=alpaca_sym)
            fill_price = (
                float(order.filled_avg_price) if order.filled_avg_price else None
            )
            raw_qty = order.qty or order.filled_qty or 0
            qty = _safe_qty(raw_qty)
            return OrderResult(
                ticker=ticker, side="sell", qty=qty, success=True,
                order_id=str(order.id),
                fill_price=fill_price,
            )
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "40310000" in err_str and attempt < 2:
                logger.info(
                    "Shares still held for %s, retrying in 1s (attempt %d/3)",
                    ticker, attempt + 1,
                )
                time.sleep(1)
            elif "40410000" in err_str:
                # Trailing stop already closed the position — treat as success
                logger.info(
                    "Position %s already closed in Alpaca (trailing stop triggered)", ticker,
                )
                return OrderResult(
                    ticker=ticker, side="sell", qty=0, success=True, already_closed=True,
                )
            else:
                raise
    raise last_err  # unreachable, but satisfies type checker


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
            # 40310000: shares already held for an existing OCO order.
            # Parse the related_orders from the error body and recover those IDs
            # so we stop retrying placement and treat the existing order as ours.
            if "40310000" in err_str:
                import json as _json
                try:
                    # Error body is the JSON dict portion of the exception string
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
                        "Could not recover OCO IDs for %s after 40310000: %s",
                        ticker, recover_err,
                    )
            else:
                logger.error("Failed to place OCO exit for %s: %s", ticker, e)
            return None, None

    return await asyncio.to_thread(_sync)


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
