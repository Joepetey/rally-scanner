# stdlib
import asyncio
import logging

# third-party
try:
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# local
import rally_ml.config as config

from db.positions import (
    enqueue_signal,
    load_all_position_meta,
    load_positions,
    log_skipped_signal,
    remove_from_queue,
)
from integrations.alpaca.broker import _data_client, _trading_client
from integrations.alpaca.models import (
    _ERR_NOT_TRADABLE,
    EntryPlan,
    OrderResult,
    _alpaca_symbol,
)
from trading.positions import async_close_position, get_group_exposure, get_total_exposure
from trading.risk_manager import is_circuit_breaker_active

logger = logging.getLogger(__name__)


def check_group_constraints(
    ticker: str,
    size: float,
    group_count: int,
    group_exposure: float,
    max_positions: int,
    max_exposure: float,
) -> str | None:
    """Pure check: group position count and exposure limits.

    Returns a skip reason string, or None if within limits.
    """
    group = config.TICKER_TO_GROUP.get(ticker)
    if not group:
        return None

    if group_count >= max_positions:
        return (
            f"Group '{group}' at max positions ({max_positions})"
        )
    if group_exposure + size > max_exposure:
        return (
            f"Group '{group}' exposure cap ({max_exposure:.0%})"
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
    # Late import to avoid circular dependency with exits module
    from integrations.alpaca.exits import execute_exit

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


def _compute_limit_price(fallback_price: float, ticker: str) -> float:
    """Return a limit price slightly above the bid/ask midpoint.

    Fetches the latest snapshot synchronously. Falls back to
    fallback_price * (1 + buffer) when bid/ask is unavailable (e.g. crypto).
    """
    buffer = config.PARAMS.limit_order_buffer_pct
    is_crypto = ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto"
    if not is_crypto:
        try:
            client = _data_client()
            snaps = client.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=[ticker])
            )
            snap = snaps.get(ticker)
            if snap and snap.latest_quote:
                bid = float(snap.latest_quote.bid_price)
                ask = float(snap.latest_quote.ask_price)
                if bid > 0 and ask > 0:
                    midpoint = (bid + ask) / 2
                    return round(midpoint * (1 + buffer), 2)
        except Exception as e:
            logger.debug("Could not fetch snapshot for %s limit price: %s", ticker, e)
    return round(fallback_price * (1 + buffer), 2)


def _compute_entry_qty(
    equity: float, size: float, price: float, is_crypto: bool,
) -> float | int | None:
    """Compute order quantity from allocation size. Returns None if insufficient."""
    if is_crypto:
        qty = round(equity * size / price, 8)
        return qty if qty > 0 else None
    qty = int(equity * size / price)
    return qty if qty >= 1 else None


def compute_available_size(
    sig_size: float,
    current_exposure: float,
    max_exposure: float,
    min_size: float,
    partial_enabled: bool,
) -> float | None:
    """Pure math: resolve effective entry size given capital constraints.

    Returns the usable size, or None if no capital is available
    (caller decides whether to rotate or queue).
    """
    if current_exposure + sig_size <= max_exposure:
        return sig_size

    available = max_exposure - current_exposure
    if partial_enabled and available >= min_size:
        return available

    return None


async def _compute_entry_size(
    sig: dict,
    current_exposure: float,
    max_exposure: float,
    min_size: float,
    open_positions: list[dict],
    open_tickers: set[str],
    results: list[OrderResult],
) -> tuple[float, float, list[dict], set[str]]:
    """Resolve effective entry size after partial sizing or rotation.

    Returns (size, current_exposure, open_positions, open_tickers).
    Returns size=0.0 if the position must be skipped (caller should continue).
    """
    size = compute_available_size(
        sig["size"], current_exposure, max_exposure, min_size,
        config.PARAMS.partial_sizing_enabled,
    )
    if size is not None:
        if size < sig["size"]:
            logger.info(
                "%s: partial size %.1f%% → %.1f%% (capital limited)",
                sig["ticker"], sig["size"] * 100, size * 100,
            )
        return size, current_exposure, open_positions, open_tickers

    if config.PARAMS.rotation_enabled and open_positions:
        rot_size, current_exposure, open_positions, open_tickers, rotated = (
            await _attempt_rotation(
                sig, open_positions, open_tickers, current_exposure, max_exposure, min_size,
                results,
            )
        )
        if rotated:
            return rot_size, current_exposure, open_positions, open_tickers

    ticker = sig["ticker"]
    enqueue_signal(sig, "capital")
    log_skipped_signal(sig, "capital")
    logger.info("Queued %s: no capital available", ticker)
    results.append(OrderResult(
        ticker=ticker, side="buy", qty=0, success=False, skipped=True,
        error=f"Portfolio exposure cap ({max_exposure:.0%}) exceeded — queued",
    ))
    return 0.0, current_exposure, open_positions, open_tickers


async def prepare_entry_plan(
    signals: list[dict],
    equity: float,
) -> tuple[list[EntryPlan], list[OrderResult]]:
    """Compute what to order without touching the broker.

    Returns (plans, skipped_results). Handles circuit breaker, sizing,
    group constraints, and qty computation. Rotation may trigger broker
    calls via ``_compute_entry_size`` but no new orders are placed.
    """
    skipped: list[OrderResult] = []

    if is_circuit_breaker_active(equity):
        for sig in signals:
            skipped.append(OrderResult(
                ticker=sig["ticker"], side="buy", qty=0, success=False,
                error="Circuit breaker active — drawdown exceeds threshold",
            ))
        return [], skipped

    current_exposure = get_total_exposure()
    max_exposure = config.PARAMS.max_portfolio_exposure
    min_size = config.PARAMS.min_position_size
    open_positions = load_all_position_meta()
    open_tickers = {p["ticker"] for p in load_positions().get("positions", [])}

    plans: list[EntryPlan] = []
    for sig in signals:
        ticker = sig["ticker"]

        if ticker in open_tickers:
            logger.info("Skipping %s: already in open positions", ticker)
            continue

        is_crypto = ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto"
        price = sig.get("entry_price") or sig["close"]

        size, current_exposure, open_positions, open_tickers = await _compute_entry_size(
            sig, current_exposure, max_exposure, min_size, open_positions, open_tickers, skipped,
        )
        if size == 0.0:
            continue

        group = config.TICKER_TO_GROUP.get(ticker)
        if group:
            g_count, g_exp = get_group_exposure(group)
        else:
            g_count, g_exp = 0, 0.0
        group_reason = check_group_constraints(
            ticker, size, g_count, g_exp,
            config.PARAMS.max_group_positions,
            config.PARAMS.max_group_exposure,
        )
        if group_reason:
            log_skipped_signal(sig, f"group:{group}")
            skipped.append(OrderResult(
                ticker=ticker, side="buy", qty=0, success=False, skipped=True,
                error=group_reason,
            ))
            continue

        qty = _compute_entry_qty(equity, size, price, is_crypto)
        if qty is None:
            skipped.append(OrderResult(
                ticker=ticker, side="buy", qty=0, success=False,
                error=(
                    "Insufficient equity for fractional crypto order"
                    if is_crypto else "Insufficient equity for 1 share"
                ),
            ))
            continue

        plans.append(EntryPlan(
            ticker=ticker, qty=qty, size=size, price=price, is_crypto=is_crypto,
        ))
        current_exposure += size

    return plans, skipped


async def execute_entries(signals: list[dict], equity: float) -> list[OrderResult]:
    plans, results = await prepare_entry_plan(signals, equity)
    if not plans:
        return results

    client = _trading_client()
    for plan in plans:
        try:
            limit_price = await asyncio.to_thread(
                _compute_limit_price, plan.price, plan.ticker,
            )
            order = await asyncio.to_thread(
                client.submit_order,
                LimitOrderRequest(
                    symbol=_alpaca_symbol(plan.ticker),
                    qty=plan.qty,
                    side=OrderSide.BUY,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.GTC if plan.is_crypto else TimeInForce.DAY,
                ),
            )
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else None
            remove_from_queue(plan.ticker)
            results.append(OrderResult(
                ticker=plan.ticker, side="buy", qty=plan.qty, success=True,
                order_id=str(order.id),
                fill_price=fill_price,
                actual_size=plan.size,
            ))
        except Exception as e:
            err_str = str(e)
            if _ERR_NOT_TRADABLE in err_str:
                logger.info("Skipping %s: not tradable on Alpaca", plan.ticker)
                results.append(OrderResult(
                    ticker=plan.ticker, side="buy", qty=0, success=False, skipped=True,
                    error=err_str,
                ))
            else:
                results.append(OrderResult(
                    ticker=plan.ticker, side="buy", qty=plan.qty, success=False,
                    error=err_str,
                ))

    return results
