"""Agentic Discord bot powered by Claude API.

Provides natural language interface to Discord bot functionality via tool use.
Users can interact conversationally instead of using slash commands.
"""

# Standard library
import json
import logging
import os

# Third-party
import anthropic

# Local
from db.users import ensure_user, get_capital
from integrations.discord.tool_defs import (  # noqa: F401 — re-export
    _TOOL_DISPATCH,
    TOOLS,
    execute_tool,
)

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
