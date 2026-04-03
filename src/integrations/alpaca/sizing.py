"""Entry sizing: capital constraints, partial sizing, rotation, and qty math."""

import logging

try:
    from alpaca.data.requests import StockSnapshotRequest

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

import rally_ml.config as config

from db.trading.positions import enqueue_signal, log_skipped_signal
from integrations.alpaca.broker import _data_client
from integrations.alpaca.exits import execute_exit
from integrations.alpaca.models import OrderResult
from trading.positions import async_close_position

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
        return f"Group '{group}' at max positions ({max_positions})"
    if group_exposure + size > max_exposure:
        return f"Group '{group}' exposure cap ({max_exposure:.0%})"
    return None


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


def _compute_entry_qty(
    equity: float, size: float, price: float, is_crypto: bool,
) -> float | int | None:
    """Compute order quantity from allocation size. Returns None if insufficient."""
    if is_crypto:
        qty = round(equity * size / price, 8)
        return qty if qty > 0 else None
    qty = int(equity * size / price)
    return qty if qty >= 1 else None


def _compute_limit_price(fallback_price: float, ticker: str) -> float:
    """Return a limit price slightly above the bid/ask midpoint.

    Fetches the latest snapshot synchronously. Falls back to
    fallback_price * (1 + buffer) when bid/ask is unavailable.
    """
    buffer = config.PARAMS.limit_order_buffer_pct
    is_crypto = (
        ticker in config.ASSETS
        and config.ASSETS[ticker].asset_class == "crypto"
    )
    if not is_crypto:
        try:
            client = _data_client()
            snaps = client.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=[ticker]),
            )
            snap = snaps.get(ticker)
            if snap and snap.latest_quote:
                bid = float(snap.latest_quote.bid_price)
                ask = float(snap.latest_quote.ask_price)
                if bid > 0 and ask > 0:
                    midpoint = (bid + ask) / 2
                    return round(midpoint * (1 + buffer), 2)
        except Exception as e:
            logger.debug(
                "Could not fetch snapshot for %s limit price: %s", ticker, e,
            )
    return round(fallback_price * (1 + buffer), 2)


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
            w_price = (
                exit_result.fill_price
                or weakest.get("current_price", 0)
            )
            await async_close_position(wticker, w_price, "rotation")
            freed = weakest.get("size", 0)
            current_exposure -= freed
            available = max_exposure - current_exposure
            open_positions = [
                p for p in open_positions if p["ticker"] != wticker
            ]
            open_tickers.discard(wticker)
            size = min(size, available) if available >= min_size else 0.0
            if size >= min_size:
                results.append(OrderResult(
                    ticker=wticker, side="sell",
                    qty=exit_result.qty, success=True,
                    order_id=exit_result.order_id,
                    fill_price=exit_result.fill_price,
                ))
                return (
                    size, current_exposure, open_positions,
                    open_tickers, True,
                )
        else:
            logger.warning(
                "Rotation exit failed for %s: %s",
                wticker, exit_result.error,
            )
    except Exception as e:
        logger.warning("Rotation exit error for %s: %s", wticker, e)

    return size, current_exposure, open_positions, open_tickers, False


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
    Returns size=0.0 if the position must be skipped.
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
                sig, open_positions, open_tickers,
                current_exposure, max_exposure, min_size, results,
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
