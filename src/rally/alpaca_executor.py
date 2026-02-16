# stdlib
import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

# third-party
from pydantic import BaseModel

# local
from . import config

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


@asynccontextmanager
async def _mcp_session() -> AsyncGenerator:
    # Lazy imports — mcp package is optional
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {
        **os.environ,
        "ALPACA_API_KEY": os.environ["ALPACA_API_KEY"],
        "ALPACA_SECRET_KEY": os.environ["ALPACA_SECRET_KEY"],
    }
    paper = os.environ.get("ALPACA_PAPER_TRADE", "true").lower() == "true"
    if paper:
        env["ALPACA_PAPER_TRADE"] = "true"

    server_params = StdioServerParameters(
        command="alpaca-mcp-server",
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _call_tool(session, name: str, arguments: dict) -> dict:
    result = await session.call_tool(name, arguments=arguments)
    # MCP tool results come back as content blocks
    for block in result.content:
        if hasattr(block, "text"):
            return json.loads(block.text)
    return {}


async def get_account_equity() -> float:
    async with _mcp_session() as session:
        data = await _call_tool(session, "get_account_info", {})
        return float(data["equity"])


async def get_snapshots(tickers: list[str]) -> dict[str, dict]:
    # Filter to equity tickers only
    equity_tickers = [
        t for t in tickers
        if t in config.ASSETS and config.ASSETS[t].asset_class == "equity"
    ]
    if not equity_tickers:
        return {}

    async with _mcp_session() as session:
        data = await _call_tool(
            session,
            "get_stock_snapshot",
            {"symbol_or_symbols": ",".join(equity_tickers)},
        )

    result: dict[str, dict] = {}
    for ticker in equity_tickers:
        snap = data.get(ticker)
        if not snap:
            continue
        latest_trade = snap.get("latestTrade", snap.get("latest_trade", {}))
        latest_quote = snap.get("latestQuote", snap.get("latest_quote", {}))
        result[ticker] = {
            "price": latest_trade.get("p", latest_trade.get("price", 0)),
            "bid": latest_quote.get("bp", latest_quote.get("bid_price", 0)),
            "ask": latest_quote.get("ap", latest_quote.get("ask_price", 0)),
            "bid_size": latest_quote.get("bs", latest_quote.get("bid_size", 0)),
            "ask_size": latest_quote.get("as", latest_quote.get("ask_size", 0)),
        }
    return result


async def execute_entries(signals: list[dict], equity: float) -> list[OrderResult]:
    from .portfolio import is_circuit_breaker_active
    from .positions import get_group_exposure, get_total_exposure

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

    async with _mcp_session() as session:
        for sig in signals:
            ticker = sig["ticker"]

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
                data = await _call_tool(session, "place_stock_order", {
                    "symbol": ticker,
                    "side": "buy",
                    "quantity": qty,
                    "type": "market",
                    "time_in_force": "day",
                })
                fill_price = None
                if data.get("filled_avg_price"):
                    fill_price = float(data["filled_avg_price"])

                # Trailing stop is deferred until fill is confirmed
                # (handled by _check_price_alerts in discord_bot.py)
                results.append(OrderResult(
                    ticker=ticker, side="buy", qty=qty, success=True,
                    order_id=data.get("id"),
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
        async with _mcp_session() as session:
            await _call_tool(session, "cancel_order_by_id", {"order_id": order_id})
            logger.info("Cancelled order %s", order_id)
            return True
    except Exception as e:
        logger.warning("Failed to cancel order %s: %s", order_id, e)
        return False


async def _execute_exit_with_session(
    session, ticker: str, trail_order_id: str | None = None,
) -> OrderResult:
    """Execute a single exit using an existing MCP session."""
    # Cancel the associated trailing stop before closing
    if trail_order_id:
        try:
            await _call_tool(
                session, "cancel_order_by_id", {"order_id": trail_order_id},
            )
            logger.info("Cancelled trailing stop %s for %s", trail_order_id, ticker)
        except Exception as e:
            logger.warning(
                "Could not cancel trailing stop %s for %s: %s",
                trail_order_id, ticker, e,
            )

    data = await _call_tool(session, "close_position", {"symbol": ticker})

    # Try to get fill price from the close response
    fill_price = None
    if data.get("filled_avg_price"):
        fill_price = float(data["filled_avg_price"])
    elif data.get("id"):
        # Check order status for fill price
        orders = await _call_tool(session, "get_orders", {
            "status": "filled",
            "symbols": ticker,
            "limit": 1,
        })
        if isinstance(orders, list) and orders:
            fill_price = float(orders[0].get("filled_avg_price", 0)) or None

    qty = int(float(data.get("qty", data.get("filled_qty", 0))))
    return OrderResult(
        ticker=ticker, side="sell", qty=qty, success=True,
        order_id=data.get("id"),
        fill_price=fill_price,
    )


async def execute_exit(ticker: str, trail_order_id: str | None = None) -> OrderResult:
    """Execute a single exit, creating a new MCP session."""
    async with _mcp_session() as session:
        return await _execute_exit_with_session(session, ticker, trail_order_id)


async def execute_exits(closed: list[dict]) -> list[OrderResult]:
    """Execute multiple exits sharing a single MCP session."""
    results: list[OrderResult] = []
    equity_closed = [
        pos for pos in closed
        if not (pos["ticker"] in config.ASSETS
                and config.ASSETS[pos["ticker"]].asset_class == "crypto")
    ]
    if not equity_closed:
        return results

    async with _mcp_session() as session:
        for pos in equity_closed:
            ticker = pos["ticker"]
            trail_oid = pos.get("trail_order_id")
            try:
                result = await _execute_exit_with_session(
                    session, ticker, trail_order_id=trail_oid,
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
    async with _mcp_session() as session:
        data = await _call_tool(session, "get_all_positions", {})
    if isinstance(data, list):
        return [
            {
                "ticker": p.get("symbol", ""),
                "qty": int(float(p.get("qty", 0))),
                "avg_entry_price": float(p.get("avg_entry_price", 0)),
                "market_value": float(p.get("market_value", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0)),
            }
            for p in data
        ]
    return []


async def place_trailing_stop(
    ticker: str, qty: int, trail_pct: float,
) -> str | None:
    """Place a standalone trailing stop order. Returns trail_order_id or None."""
    async with _mcp_session() as session:
        try:
            data = await _call_tool(session, "place_stock_order", {
                "symbol": ticker,
                "side": "sell",
                "quantity": qty,
                "type": "trailing_stop",
                "trail_percent": trail_pct,
                "time_in_force": "gtc",
            })
            trail_id = data.get("id")
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

    async with _mcp_session() as session:
        data = await _call_tool(session, "get_orders", {"status": "filled"})

    id_set = set(order_ids)
    fills: dict[str, float] = {}
    if isinstance(data, list):
        for order in data:
            oid = order.get("id")
            if oid in id_set and order.get("filled_avg_price"):
                fills[oid] = float(order["filled_avg_price"])
    return fills


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

    async with _mcp_session() as session:
        data = await _call_tool(session, "get_orders", {"status": "filled"})

    # Invert: order_id -> ticker
    oid_to_ticker = {oid: t for t, oid in trail_order_ids.items()}
    fills: dict[str, float] = {}
    if isinstance(data, list):
        for order in data:
            oid = order.get("id")
            if oid in oid_to_ticker and order.get("filled_avg_price"):
                fills[oid_to_ticker[oid]] = float(order["filled_avg_price"])
    return fills
