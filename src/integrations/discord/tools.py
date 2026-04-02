"""Tool definitions and dispatch for the Claude agent.

Each tool has a JSON schema (for Claude) and a handler function.
"""

import logging

from services import portfolio, queries, system, trades

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definitions (Claude API format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_signals",
        "description": (
            "Show recent entry signals from the market scanner. These are BUY SIGNALS only"
            " — NOT confirmed open positions. Whether actual trades were placed depends on"
            " whether Alpaca auto-execution is enabled. Use get_system_positions to see what"
            " is actually open."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_system_positions",
        "description": (
            "Show the system's currently open positions with P&L, stops, and targets."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_user_positions",
        "description": (
            "Show the user's currently open positions that they're tracking."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "enter_trade",
        "description": (
            "Record a new trade entry for the user."
            " Auto-fills size/stop/target from system signal if available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. AAPL, NVDA)",
                },
                "price": {
                    "type": "number",
                    "description": "Entry price in dollars",
                },
                "size": {
                    "type": "number",
                    "description": (
                        "Position size as decimal fraction of capital"
                        " (e.g. 0.15 for 15%). Optional — uses system"
                        " recommendation if available."
                    ),
                },
                "stop": {
                    "type": "number",
                    "description": (
                        "Stop-loss price in dollars. Optional — uses system"
                        " recommendation if available."
                    ),
                },
                "target": {
                    "type": "number",
                    "description": (
                        "Profit target price in dollars. Optional — uses system"
                        " recommendation if available."
                    ),
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about the trade",
                },
            },
            "required": ["ticker", "price"],
        },
    },
    {
        "name": "exit_trade",
        "description": (
            "Close the user's oldest open trade for a ticker (FIFO). Records P&L."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to close",
                },
                "price": {
                    "type": "number",
                    "description": "Exit price in dollars",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about why closing",
                },
            },
            "required": ["ticker", "price"],
        },
    },
    {
        "name": "get_pnl",
        "description": (
            "Show the user's P&L summary including total return, win rate,"
            " best/worst trades."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Time period: 'all', '30d', or '7d'",
                    "enum": ["all", "30d", "7d"],
                },
            },
            "required": [],
        },
    },
    {
        "name": "set_capital",
        "description": (
            "Set the user's portfolio capital amount. Used to calculate"
            " position sizes and dollar-based P&L."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": (
                        "Portfolio capital in dollars (e.g., 10000 for $10,000)"
                    ),
                },
            },
            "required": ["amount"],
        },
    },
    {
        "name": "get_trade_history",
        "description": (
            "Show the user's trade history with entry/exit details and P&L."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Filter by specific ticker (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of trades to show (default 10)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_portfolio",
        "description": (
            "Show system portfolio equity history and recent closed trades."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": (
                        "Number of days of history to show (default 30)"
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_health",
        "description": (
            "Show model health status — how many models are fresh vs stale,"
            " and system status."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_scan",
        "description": (
            "Run the market scanner to find new rally signals. Returns ticker"
            " signals — NOT confirmed trades. Actual positions are only opened"
            " if Alpaca auto-execution is configured; check"
            " get_system_positions to see what is actually open."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "string",
                    "description": (
                        "Scan configuration: 'conservative' (default),"
                        " 'baseline', 'aggressive', or 'concentrated'"
                    ),
                    "enum": [
                        "baseline", "conservative", "aggressive", "concentrated",
                    ],
                },
            },
            "required": [],
        },
    },
    {
        "name": "run_retrain",
        "description": (
            "Retrain models for all tickers in the universe. Long-running"
            " operation (10-30+ minutes). User receives progress updates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of specific tickers to retrain."
                        " If not provided, retrains all tickers in universe."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_price",
        "description": "Look up current stock prices for one or more tickers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of stock ticker symbols (e.g. ['HD', 'AAPL'])"
                    ),
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_watchlist",
        "description": (
            "Show tickers approaching signal threshold (P(rally) > 35%)"
            " from the last scan, sorted by probability. These may"
            " become entry signals soon."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# ---------------------------------------------------------------------------
# Tool Dispatch
# ---------------------------------------------------------------------------

def _dispatch_get_signals(ti: dict, did: int, cap: float) -> dict:
    return queries.get_signals(cap)


def _dispatch_get_system_positions(ti: dict, did: int, cap: float) -> dict:
    return queries.get_system_positions(cap)


def _dispatch_get_user_positions(ti: dict, did: int, cap: float) -> dict:
    return queries.get_user_positions(cap)


def _dispatch_enter_trade(ti: dict, did: int, cap: float) -> dict:
    return trades.enter_trade(ti, cap)


def _dispatch_exit_trade(ti: dict, did: int, cap: float) -> dict:
    return trades.exit_trade(ti, cap)


def _dispatch_get_pnl(ti: dict, did: int, cap: float) -> dict:
    return portfolio.get_pnl(cap, ti.get("period", "all"))


def _dispatch_set_capital(ti: dict, did: int, cap: float) -> dict:
    return portfolio.set_capital_amount(did, ti.get("amount"))


def _dispatch_get_trade_history(ti: dict, did: int, cap: float) -> dict:
    return trades.get_trade_history(
        ti.get("ticker"), ti.get("limit", trades.DEFAULT_TRADE_LIMIT),
    )


def _dispatch_get_portfolio(ti: dict, did: int, cap: float) -> dict:
    return portfolio.get_portfolio(ti.get("days", 30))


def _dispatch_get_health(ti: dict, did: int, cap: float) -> dict:
    return system.get_health()


def _dispatch_run_scan(ti: dict, did: int, cap: float) -> dict:
    return system.run_scan(ti.get("config", "conservative"))


def _dispatch_run_retrain(ti: dict, did: int, cap: float) -> dict:
    return system.run_retrain_marker(ti.get("tickers"))


def _dispatch_get_price(ti: dict, did: int, cap: float) -> dict:
    return queries.get_price(ti.get("tickers", []))


def _dispatch_get_watchlist(ti: dict, did: int, cap: float) -> dict:
    return queries.get_watchlist()


_TOOL_DISPATCH: dict[str, callable] = {
    "get_signals": _dispatch_get_signals,
    "get_system_positions": _dispatch_get_system_positions,
    "get_user_positions": _dispatch_get_user_positions,
    "enter_trade": _dispatch_enter_trade,
    "exit_trade": _dispatch_exit_trade,
    "get_pnl": _dispatch_get_pnl,
    "set_capital": _dispatch_set_capital,
    "get_trade_history": _dispatch_get_trade_history,
    "get_portfolio": _dispatch_get_portfolio,
    "get_health": _dispatch_get_health,
    "run_scan": _dispatch_run_scan,
    "run_retrain": _dispatch_run_retrain,
    "get_price": _dispatch_get_price,
    "get_watchlist": _dispatch_get_watchlist,
}


def execute_tool(
    tool_name: str,
    tool_input: dict,
    discord_id: int,
    capital: float,
) -> dict:
    """Execute a tool and return results."""
    handler = _TOOL_DISPATCH.get(tool_name)
    if handler is None:
        logger.warning("Unknown tool requested: %s (input: %s)", tool_name, tool_input)
        return {"error": f"Unknown tool: {tool_name}"}
    return handler(tool_input, discord_id, capital)
