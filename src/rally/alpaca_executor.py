# stdlib
import json
import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

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
    results: list[OrderResult] = []

    async with _mcp_session() as session:
        for sig in signals:
            ticker = sig["ticker"]

            # Skip crypto
            if ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto":
                continue

            price = sig["entry_price"]
            size = sig["size"]
            qty = int(equity * size / price)
            if qty < 1:
                results.append(OrderResult(
                    ticker=ticker, side="buy", qty=0, success=False,
                    error="Insufficient equity for 1 share",
                ))
                continue

            try:
                data = await _call_tool(session, "place_stock_order", {
                    "symbol": ticker,
                    "side": "buy",
                    "quantity": qty,
                    "order_type": "market",
                    "time_in_force": "day",
                })
                fill_price = None
                if data.get("filled_avg_price"):
                    fill_price = float(data["filled_avg_price"])
                results.append(OrderResult(
                    ticker=ticker, side="buy", qty=qty, success=True,
                    order_id=data.get("id"),
                    fill_price=fill_price,
                ))
            except Exception as e:
                results.append(OrderResult(
                    ticker=ticker, side="buy", qty=qty, success=False,
                    error=str(e),
                ))

    return results


async def execute_exit(ticker: str) -> OrderResult:
    async with _mcp_session() as session:
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


async def execute_exits(closed: list[dict]) -> list[OrderResult]:
    results: list[OrderResult] = []
    for pos in closed:
        ticker = pos["ticker"]
        if ticker in config.ASSETS and config.ASSETS[ticker].asset_class == "crypto":
            continue
        try:
            result = await execute_exit(ticker)
            results.append(result)
        except Exception as e:
            results.append(OrderResult(
                ticker=ticker, side="sell", qty=0, success=False,
                error=str(e),
            ))
    return results


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
