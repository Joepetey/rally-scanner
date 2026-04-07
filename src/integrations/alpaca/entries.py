"""Entry execution: orchestrates sizing + risk checks + broker orders."""

import asyncio
import logging

try:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

import rally_ml.config as config

from db.trading.positions import (
    get_recently_closed_tickers,
    load_all_position_meta,
    load_positions,
)
from db.trading.signal_queue import log_skipped_signal, remove_from_queue
from integrations.alpaca.broker import _alpaca_symbol, _trading_client
from integrations.alpaca.models import _ERR_NOT_TRADABLE, EntryPlan, OrderResult
from integrations.alpaca.sizing import (
    _compute_entry_qty,
    _compute_entry_size,
    _compute_limit_price,
    check_group_constraints,
)
from trading.positions import get_group_exposure, get_total_exposure
from trading.risk import is_circuit_breaker_active

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
__all__ = [
    "check_group_constraints",
    "compute_available_size",
    "execute_entries",
    "prepare_entry_plan",
]

from integrations.alpaca.sizing import compute_available_size  # noqa: E402, F401


async def prepare_entry_plan(
    signals: list[dict],
    equity: float,
) -> tuple[list[EntryPlan], list[OrderResult]]:
    """Compute what to order without touching the broker.

    Returns (plans, skipped_results). Handles circuit breaker, sizing,
    group constraints, and qty computation.
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
    open_tickers = {
        p["ticker"] for p in load_positions().get("positions", [])
    }
    cooldown_tickers: set[str] = set()
    if config.PARAMS.cooldown_days > 0:
        cooldown_tickers = get_recently_closed_tickers(config.PARAMS.cooldown_days)

    plans: list[EntryPlan] = []
    for sig in signals:
        ticker = sig["ticker"]

        if ticker in open_tickers:
            logger.info("Skipping %s: already in open positions", ticker)
            skipped.append(OrderResult(
                ticker=ticker, side="buy", qty=0, success=False, skipped=True,
                error="Already in open positions",
            ))
            continue

        if ticker in cooldown_tickers:
            logger.info(
                "Skipping %s: in cooldown period (%dd)",
                ticker, config.PARAMS.cooldown_days,
            )
            skipped.append(OrderResult(
                ticker=ticker, side="buy", qty=0, success=False, skipped=True,
                error=f"In cooldown period ({config.PARAMS.cooldown_days}d)",
            ))
            continue

        is_crypto = (
            ticker in config.ASSETS
            and config.ASSETS[ticker].asset_class == "crypto"
        )
        price = sig.get("entry_price") or sig["close"]

        size, current_exposure, open_positions, open_tickers = (
            await _compute_entry_size(
                sig, current_exposure, max_exposure, min_size,
                open_positions, open_tickers, skipped,
            )
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
                    if is_crypto
                    else "Insufficient equity for 1 share"
                ),
            ))
            continue

        plans.append(EntryPlan(
            ticker=ticker, qty=qty, size=size,
            price=price, is_crypto=is_crypto,
        ))
        current_exposure += size

    return plans, skipped


async def execute_entries(
    signals: list[dict], equity: float,
) -> list[OrderResult]:
    """Execute entry orders via Alpaca broker."""
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
                    time_in_force=(
                        TimeInForce.GTC if plan.is_crypto else TimeInForce.DAY
                    ),
                ),
            )
            fill_price = (
                float(order.filled_avg_price)
                if order.filled_avg_price
                else None
            )
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
                logger.info(
                    "Skipping %s: not tradable on Alpaca", plan.ticker,
                )
                results.append(OrderResult(
                    ticker=plan.ticker, side="buy", qty=0,
                    success=False, skipped=True, error=err_str,
                ))
            else:
                results.append(OrderResult(
                    ticker=plan.ticker, side="buy", qty=plan.qty,
                    success=False, error=err_str,
                ))

    return results
