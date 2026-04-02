"""Agentic Discord bot powered by Claude API.

Thin orchestration layer: translates natural language into tool calls,
delegates all business logic to the services layer.
"""

import json
import logging
import os

import anthropic

from db.users import ensure_user, get_capital
from services import portfolio, queries, system, trades

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic Client
# ---------------------------------------------------------------------------

def _get_client() -> anthropic.Anthropic | None:
    """Get Anthropic client if API key configured."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Tool Definitions
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
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_system_positions",
        "description": "Show the system's currently open positions with P&L, stops, and targets.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_user_positions",
        "description": "Show the user's currently open positions that they're tracking.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
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
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, NVDA)"},  # noqa: E501
                "price": {"type": "number", "description": "Entry price in dollars"},
                "size": {"type": "number", "description": "Position size as decimal fraction of capital (e.g. 0.15 for 15%). Optional - will use system recommendation if available."},  # noqa: E501
                "stop": {"type": "number", "description": "Stop-loss price in dollars. Optional - will use system recommendation if available."},  # noqa: E501
                "target": {"type": "number", "description": "Profit target price in dollars. Optional - will use system recommendation if available."},  # noqa: E501
                "notes": {"type": "string", "description": "Optional notes about the trade"}
            },
            "required": ["ticker", "price"]
        }
    },
    {
        "name": "exit_trade",
        "description": "Close the user's oldest open trade for a ticker (FIFO). Records P&L.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol to close"},
                "price": {"type": "number", "description": "Exit price in dollars"},
                "notes": {"type": "string", "description": "Optional notes about why closing"}
            },
            "required": ["ticker", "price"]
        }
    },
    {
        "name": "get_pnl",
        "description": "Show the user's P&L summary including total return, win rate, best/worst trades.",  # noqa: E501
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "description": "Time period: 'all', '30d', or '7d'", "enum": ["all", "30d", "7d"]}  # noqa: E501
            },
            "required": []
        }
    },
    {
        "name": "set_capital",
        "description": "Set the user's portfolio capital amount. This is used to calculate position sizes and dollar-based P&L.",  # noqa: E501
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Portfolio capital in dollars (e.g., 10000 for $10,000)"}  # noqa: E501
            },
            "required": ["amount"]
        }
    },
    {
        "name": "get_trade_history",
        "description": "Show the user's trade history with entry/exit details and P&L.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Filter by specific ticker (optional)"},
                "limit": {"type": "integer", "description": "Number of trades to show (default 10)"}
            },
            "required": []
        }
    },
    {
        "name": "get_portfolio",
        "description": "Show system portfolio equity history and recent closed trades.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days of history to show (default 30)"}  # noqa: E501
            },
            "required": []
        }
    },
    {
        "name": "get_health",
        "description": "Show model health status - how many models are fresh vs stale, and system status.",  # noqa: E501
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "run_scan",
        "description": "Run the market scanner to find new rally signals. Returns ticker signals — NOT confirmed trades. Actual positions are only opened if Alpaca auto-execution is configured; check get_system_positions to see what is actually open.",  # noqa: E501
        "input_schema": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "string",
                    "description": "Scan configuration: 'conservative' (default), 'baseline', 'aggressive', or 'concentrated'",  # noqa: E501
                    "enum": ["baseline", "conservative", "aggressive", "concentrated"]
                }
            },
            "required": []
        }
    },
    {
        "name": "run_retrain",
        "description": "Retrain models for all tickers in the universe. This is a long-running operation (can take 10-30+ minutes). The user will receive progress updates during training and a completion notification.",  # noqa: E501
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific tickers to retrain. If not provided, retrains all tickers in universe."  # noqa: E501
                }
            },
            "required": []
        }
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
                    "description": "List of stock ticker symbols (e.g. ['HD', 'AAPL'])"
                }
            },
            "required": ["tickers"]
        }
    },
    {
        "name": "get_watchlist",
        "description": (
            "Show tickers approaching signal threshold (P(rally) > 35%)"
            " from the last scan, sorted by probability. These may"
            " become entry signals soon."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
]


# ---------------------------------------------------------------------------
# Tool Dispatch
# ---------------------------------------------------------------------------

_TOOL_DISPATCH: dict = {
    "get_signals": lambda ti, did, cap: queries.get_signals(cap),
    "get_system_positions": lambda ti, did, cap: queries.get_system_positions(cap),
    "get_user_positions": lambda ti, did, cap: queries.get_user_positions(cap),
    "enter_trade": lambda ti, did, cap: trades.enter_trade(ti, cap),
    "exit_trade": lambda ti, did, cap: trades.exit_trade(ti, cap),
    "get_pnl": lambda ti, did, cap: portfolio.get_pnl(cap, ti.get("period", "all")),
    "set_capital": lambda ti, did, cap: portfolio.set_capital_amount(did, ti.get("amount")),
    "get_trade_history": lambda ti, did, cap: trades.get_trade_history(
        ti.get("ticker"), ti.get("limit", trades.DEFAULT_TRADE_LIMIT),
    ),
    "get_portfolio": lambda ti, did, cap: portfolio.get_portfolio(ti.get("days", 30)),
    "get_health": lambda ti, did, cap: system.get_health(),
    "run_scan": lambda ti, did, cap: system.run_scan(ti.get("config", "conservative")),
    "run_retrain": lambda ti, did, cap: system.run_retrain_marker(ti.get("tickers")),
    "get_price": lambda ti, did, cap: queries.get_price(ti.get("tickers", [])),
    "get_watchlist": lambda ti, did, cap: queries.get_watchlist(),
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


# ---------------------------------------------------------------------------
# Main Agent Function
# ---------------------------------------------------------------------------

def _build_system_prompt(discord_username: str, capital: float) -> str:
    capital_context = f"${capital:,.0f}" if capital > 0 else "not set"
    return f"""You are a helpful trading assistant for the Rally Scanner system.

The user {discord_username} is asking about their trades and market signals.
Their portfolio capital is {capital_context}.

Available tools let you:
- Look up current stock prices
- View market signals and system positions
- Track the user's personal trades
- Enter/exit trades on their behalf
- Show performance metrics

Be concise, helpful, and action-oriented. When users ask to enter or exit trades,
use the appropriate tools. Format numbers clearly with $ and %.

Key conventions:
- Position sizes are shown as percentages (e.g., 15% of capital)
- P&L is shown in both percentage and dollar amounts when capital is set
- Stops and targets are price levels, not percentages
- The system uses a volatility-targeted sizing approach
"""


def _call_claude(client, model: str, max_tokens: int, system_prompt: str, history: list[dict]):
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=history,
        tools=TOOLS,
    )


def _run_tool_loop(
    client,
    model: str,
    max_tokens: int,
    system_prompt: str,
    conversation_history: list[dict],
    discord_id: int,
    capital: float,
    discord_username: str,
    async_tasks: list[dict],
    response,
) -> tuple[str | None, list[dict]]:
    """Drive the tool-use loop until stop_reason != 'tool_use'.

    Returns (response_text, history) where response_text is None on rate limit.
    """
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                logger.info("Executing tool: %s with input: %s", block.name, block.input)
                result = execute_tool(block.name, block.input, discord_id, capital)
                if isinstance(result, dict) and "_async_task" in result:
                    async_tasks.append(result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        conversation_history.append({
            "role": "assistant",
            "content": [block.model_dump() for block in response.content],
        })
        conversation_history.append({"role": "user", "content": tool_results})

        try:
            response = _call_claude(client, model, max_tokens, system_prompt, conversation_history)
        except anthropic.RateLimitError:
            logger.warning("Rate limit hit mid-turn for user %s", discord_username)
            return None, conversation_history

    response_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    conversation_history.append({
        "role": "assistant",
        "content": [block.model_dump() for block in response.content],
    })
    return response_text, conversation_history


def process_message(
    user_message: str,
    discord_id: int,
    discord_username: str,
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """Process a user message with Claude, execute tools as needed.

    Args:
        user_message: The user's natural language message
        discord_id: Discord user ID
        discord_username: Discord username for context
        conversation_history: Previous messages (optional)

    Returns:
        Tuple of (response_text, updated_conversation_history, async_tasks)
    """
    async_tasks: list[dict] = []
    client = _get_client()
    if not client:
        return (
            "Claude API not configured. Use slash commands instead.",
            conversation_history or [],
            [],
        )

    ensure_user(discord_id, discord_username)
    capital = get_capital(discord_id)

    if conversation_history is None:
        conversation_history = []

    system_prompt = _build_system_prompt(discord_username, capital)
    conversation_history.append({"role": "user", "content": user_message})

    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
    max_tokens = int(os.environ.get("CLAUDE_MAX_TOKENS", "2048"))

    try:
        response = _call_claude(client, model, max_tokens, system_prompt, conversation_history)
    except anthropic.RateLimitError:
        logger.warning("Rate limit hit for user %s", discord_username)
        conversation_history.pop()
        return (
            "Rate limit reached — too many tokens in flight. Try again in a moment, or type `/clear` to reset your conversation history.",  # noqa: E501
            conversation_history,
            [],
        )

    response_text, conversation_history = _run_tool_loop(
        client, model, max_tokens, system_prompt,
        conversation_history, discord_id, capital, discord_username,
        async_tasks, response,
    )

    if response_text is None:
        return (
            "Rate limit reached mid-response. Try again in a moment, or type `/clear` to reset your conversation history.",  # noqa: E501
            conversation_history,
            async_tasks,
        )

    logger.info("Claude response for %s: %s...", discord_username, response_text[:100])
    return response_text, conversation_history, async_tasks
