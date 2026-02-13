"""Agentic Discord bot powered by Claude API.

Provides natural language interface to Discord bot functionality via tool use.
Users can interact conversationally instead of using slash commands.
"""

# Standard library
import json
import logging
import os
from typing import Any

# Third-party
import anthropic

# Local
from .discord_db import (
    close_trade,
    ensure_user,
    get_capital,
    get_open_trades,
    get_pnl_summary,
    get_trade_history,
    open_trade,
    set_capital,
)
from .persistence import load_manifest
from .portfolio import load_equity_history, load_trade_journal
from .positions import load_positions

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
        "description": "Show recent entry signals from the market scanner. These are positions the system recently opened or plans to enter.",
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
        "description": "Record a new trade entry for the user. Auto-fills size/stop/target from system signal if available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, NVDA)"},
                "price": {"type": "number", "description": "Entry price in dollars"},
                "size": {"type": "number", "description": "Position size as decimal fraction of capital (e.g. 0.15 for 15%). Optional - will use system recommendation if available."},
                "stop": {"type": "number", "description": "Stop-loss price in dollars. Optional - will use system recommendation if available."},
                "target": {"type": "number", "description": "Profit target price in dollars. Optional - will use system recommendation if available."},
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
        "description": "Show the user's P&L summary including total return, win rate, best/worst trades.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "description": "Time period: 'all', '30d', or '7d'", "enum": ["all", "30d", "7d"]}
            },
            "required": []
        }
    },
    {
        "name": "set_capital",
        "description": "Set the user's portfolio capital amount. This is used to calculate position sizes and dollar-based P&L.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Portfolio capital in dollars (e.g., 10000 for $10,000)"}
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
                "days": {"type": "integer", "description": "Number of days of history to show (default 30)"}
            },
            "required": []
        }
    },
    {
        "name": "get_health",
        "description": "Show model health status - how many models are fresh vs stale, and system status.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "run_scan",
        "description": "Run the market scanner to find new rally signals. This scans all trained tickers and updates positions. Returns scan results and timestamp.",
        "input_schema": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "string",
                    "description": "Scan configuration: 'baseline' (default), 'conservative', 'aggressive', or 'concentrated'",
                    "enum": ["baseline", "conservative", "aggressive", "concentrated"]
                }
            },
            "required": []
        }
    },
    {
        "name": "run_retrain",
        "description": "Retrain models for all tickers in the universe. This is a long-running operation (can take 10-30+ minutes). The user will receive progress updates during training and a completion notification.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific tickers to retrain. If not provided, retrains all tickers in universe."
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
]


# ---------------------------------------------------------------------------
# Tool Execution Functions
# ---------------------------------------------------------------------------

def _get_signals(discord_id: int, capital: float) -> dict[str, Any]:
    """Get recent entry signals (positions held ≤ 1 bar)."""
    state = load_positions()
    positions = state.get("positions", [])

    recent_entries = [
        p for p in positions
        if p.get("bars_held", 99) <= 1
    ]

    if not recent_entries:
        return {"message": "No recent signals from the latest scan."}

    signals_data = []
    for p in recent_entries:
        pnl = p.get("unrealized_pnl_pct", 0)
        size = p.get("size", 0)
        signal_info = {
            "ticker": p["ticker"],
            "entry_price": p["entry_price"],
            "stop_price": p.get("stop_price", 0),
            "target_price": p.get("target_price", 0),
            "size_pct": size * 100,
            "pnl_pct": pnl,
        }

        if capital > 0 and size > 0:
            dollar_size = capital * size
            stop = p.get("stop_price", 0)
            if stop and p["entry_price"] > stop:
                dollar_risk = capital * size * (p["entry_price"] - stop) / p["entry_price"]
                signal_info["dollar_allocation"] = dollar_size
                signal_info["dollar_risk"] = dollar_risk

        signals_data.append(signal_info)

    return {
        "count": len(recent_entries),
        "signals": signals_data,
        "capital": capital if capital > 0 else None
    }


def _get_system_positions(discord_id: int, capital: float) -> dict[str, Any]:
    """Get system's open positions."""
    state = load_positions()
    positions = state.get("positions", [])

    if not positions:
        return {"message": "The system has no open positions."}

    total_exposure = sum(p.get("size", 0) for p in positions)
    positions_data = []

    for p in positions:
        pnl = p.get("unrealized_pnl_pct", 0)
        size = p.get("size", 0)
        pos_info = {
            "ticker": p["ticker"],
            "entry_price": p["entry_price"],
            "current_price": p.get("current_price", 0),
            "stop_price": p.get("stop_price", 0),
            "target_price": p.get("target_price", 0),
            "size_pct": size * 100,
            "pnl_pct": pnl,
            "bars_held": p.get("bars_held", 0),
        }

        if capital > 0 and size > 0:
            dollar_alloc = capital * size
            dollar_pnl = dollar_alloc * pnl / 100
            pos_info["dollar_allocation"] = dollar_alloc
            pos_info["dollar_pnl"] = dollar_pnl

        positions_data.append(pos_info)

    return {
        "count": len(positions),
        "total_exposure_pct": total_exposure * 100,
        "total_exposure_dollars": capital * total_exposure if capital > 0 else None,
        "positions": positions_data,
        "capital": capital if capital > 0 else None
    }


def _get_user_positions(discord_id: int, capital: float) -> dict[str, Any]:
    """Get user's tracked open positions."""
    trades = get_open_trades(discord_id)

    if not trades:
        return {"message": "You have no open positions."}

    positions_data = []
    for trade in trades:
        pos_info = {
            "ticker": trade["ticker"],
            "entry_price": trade["entry_price"],
            "entry_date": trade["entry_date"],
            "size_pct": trade["size"] * 100,
        }
        if trade.get("stop_price"):
            pos_info["stop_price"] = trade["stop_price"]
        if trade.get("target_price"):
            pos_info["target_price"] = trade["target_price"]

        if capital > 0:
            dollar_size = capital * trade["size"]
            pos_info["dollar_allocation"] = dollar_size

        positions_data.append(pos_info)

    return {
        "count": len(trades),
        "positions": positions_data,
        "capital": capital if capital > 0 else None
    }


def _enter_trade(discord_id: int, tool_input: dict[str, Any], capital: float) -> dict[str, Any]:
    """Record a trade entry."""
    ticker = tool_input["ticker"].upper()
    price = tool_input["price"]
    size = tool_input.get("size")
    stop = tool_input.get("stop")
    target = tool_input.get("target")
    notes = tool_input.get("notes")

    # Auto-fill from system recommendation if not provided
    state = load_positions()
    for p in state.get("positions", []):
        if p.get("ticker", "").upper() == ticker:
            if size is None:
                size = p.get("size", 0)
            if stop is None:
                stop = p.get("stop_price")
            if target is None:
                target = p.get("target_price")
            break

    if size is None:
        size = 0.0

    trade_id = open_trade(
        discord_id=discord_id,
        ticker=ticker,
        entry_price=price,
        size=size,
        stop_price=stop,
        target_price=target,
        notes=notes,
    )

    result = {
        "trade_id": trade_id,
        "ticker": ticker,
        "entry_price": price,
        "size_pct": size * 100,
    }

    if stop:
        risk_pct = (price - stop) / price * 100
        result["stop_price"] = stop
        result["risk_pct"] = risk_pct

    if target:
        reward_pct = (target - price) / price * 100
        result["target_price"] = target
        result["reward_pct"] = reward_pct

    if capital > 0 and size > 0:
        dollar_size = capital * size
        result["dollar_allocation"] = dollar_size
        result["capital"] = capital
        if stop:
            dollar_risk = capital * size * (price - stop) / price
            result["dollar_risk"] = dollar_risk

    return result


def _exit_trade(discord_id: int, tool_input: dict[str, Any], capital: float) -> dict[str, Any]:
    """Close a trade."""
    ticker = tool_input["ticker"].upper()
    price = tool_input["price"]
    notes = tool_input.get("notes")

    result = close_trade(
        discord_id=discord_id,
        ticker=ticker,
        exit_price=price,
        notes=notes,
    )

    if result is None:
        return {"error": f"No open trades found for {ticker}"}

    response = {
        "ticker": result["ticker"],
        "entry_price": result["entry_price"],
        "exit_price": result["exit_price"],
        "pnl_pct": result["pnl_pct"],
        "size_pct": result["size"] * 100,
        "entry_date": result["entry_date"],
        "exit_date": result["exit_date"],
    }

    pnl_dollar = result.get("pnl_dollar")
    if pnl_dollar is not None:
        response["pnl_dollar"] = pnl_dollar

    return response


def _get_pnl(discord_id: int, period: str = "all") -> dict[str, Any]:
    """Get P&L summary."""
    days = None
    if period == "30d":
        days = 30
    elif period == "7d":
        days = 7

    summary = get_pnl_summary(discord_id, days=days)
    capital = get_capital(discord_id)

    result = {
        "period": {"all": "All Time", "30d": "Last 30 Days", "7d": "Last 7 Days"}[period],
        "total_pnl_pct": summary["total_pnl"],
        "avg_pnl_pct": summary["avg_pnl"],
        "win_rate_pct": summary["win_rate"],
        "n_trades": summary["n_trades"],
        "best_trade_pct": summary["best_trade"],
        "worst_trade_pct": summary["worst_trade"],
    }

    dollar_pnl = summary.get("total_pnl_dollar", 0.0)
    if dollar_pnl != 0:
        result["total_pnl_dollar"] = dollar_pnl

    if capital > 0:
        result["capital"] = capital

    return result


def _set_capital(discord_id: int, amount: float) -> dict[str, Any]:
    """Set user's portfolio capital."""
    if amount <= 0:
        return {"error": "Capital amount must be greater than 0"}

    set_capital(discord_id, amount)

    return {
        "success": True,
        "capital": amount,
        "message": f"Portfolio capital set to ${amount:,.2f}"
    }


def _get_trade_history(discord_id: int, ticker: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Get trade history."""
    trades = get_trade_history(discord_id, ticker=ticker, limit=limit)

    if not trades:
        msg = f"No trades found for {ticker.upper()}" if ticker else "No trade history"
        return {"message": msg}

    trades_data = []
    for t in trades:
        trade_info = {
            "ticker": t["ticker"],
            "status": t["status"],
            "entry_price": t["entry_price"],
            "entry_date": t["entry_date"],
            "size_pct": t["size"] * 100,
        }

        if t["status"] == "closed":
            trade_info["exit_price"] = t["exit_price"]
            trade_info["exit_date"] = t["exit_date"]
            trade_info["pnl_pct"] = t.get("pnl_pct", 0)
            if t.get("pnl_dollar") is not None:
                trade_info["pnl_dollar"] = t["pnl_dollar"]
        else:
            if t.get("stop_price"):
                trade_info["stop_price"] = t["stop_price"]
            if t.get("target_price"):
                trade_info["target_price"] = t["target_price"]

        trades_data.append(trade_info)

    return {
        "ticker_filter": ticker.upper() if ticker else None,
        "count": len(trades),
        "trades": trades_data
    }


def _get_portfolio(days: int = 30) -> dict[str, Any]:
    """Get system portfolio info."""
    history = load_equity_history(days=days)
    trades = load_trade_journal(limit=10)

    if not history:
        return {"message": "No equity history yet. Run the orchestrator scan to start."}

    latest = history[-1]
    result = {
        "days": days,
        "latest_date": latest.get("date", "?"),
        "n_positions": latest.get("n_positions", 0),
        "total_exposure_pct": float(latest.get("total_exposure", 0)) * 100,
        "n_signals_today": latest.get("n_signals_today", 0),
        "n_scanned": latest.get("n_scanned", 0),
    }

    if trades:
        recent_trades = []
        for t in trades[-5:]:
            pnl = float(t.get("realized_pnl_pct", 0))
            recent_trades.append({
                "ticker": t.get("ticker", "?"),
                "pnl_pct": pnl,
                "exit_reason": t.get("exit_reason", "?"),
            })
        result["recent_closed_trades"] = recent_trades

    return result


def _get_health() -> dict[str, Any]:
    """Get model health status."""
    from datetime import datetime

    manifest = load_manifest()
    now = datetime.now()

    stale = []
    fresh = []
    for ticker, info in manifest.items():
        try:
            saved_at = datetime.fromisoformat(info["saved_at"])
            age_days = (now - saved_at).days
            if age_days > 14:
                stale.append({"ticker": ticker, "age_days": age_days})
            else:
                fresh.append(ticker)
        except (KeyError, ValueError):
            stale.append({"ticker": ticker, "age_days": 999})

    state = load_positions()
    positions = state.get("positions", [])
    total_exposure = sum(p.get("size", 0) for p in positions) if positions else 0

    result = {
        "total_models": len(manifest),
        "fresh_models": len(fresh),
        "stale_models": len(stale),
        "open_positions": len(positions),
        "total_exposure_pct": total_exposure * 100,
    }

    if stale:
        stale_sorted = sorted(stale, key=lambda x: -x["age_days"])[:10]
        result["stalest_models"] = stale_sorted

    return result


def _run_scan(config: str = "baseline") -> dict[str, Any]:
    """Run the market scanner and return results."""
    from datetime import datetime
    from .scanner import scan_all
    from .positions import load_positions

    try:
        # Run the scan
        logger.info(f"Running market scan with config: {config}")
        results = scan_all(tickers=None, show_positions=False, config_name=config)

        if not results:
            return {
                "success": False,
                "error": "No models found. Run retrain first."
            }

        # Get updated positions
        state = load_positions()
        positions = state.get("positions", [])

        # Find new signals (bars_held <= 1)
        new_signals = [p for p in positions if p.get("bars_held", 99) <= 1]

        # Count closed positions
        closed_today = state.get("closed_today", [])

        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "success": True,
            "scan_time": scan_time,
            "config": config,
            "tickers_scanned": len(results),
            "new_signals": len(new_signals),
            "total_open": len(positions),
            "closed_today": len(closed_today),
            "signals": [
                {
                    "ticker": p["ticker"],
                    "entry_price": p["entry_price"],
                    "size_pct": p.get("size", 0) * 100,
                    "stop_price": p.get("stop_price"),
                    "target_price": p.get("target_price"),
                }
                for p in new_signals
            ]
        }
    except Exception as e:
        logger.exception("Scan failed")
        return {
            "success": False,
            "error": str(e)
        }


def _get_price(tickers: list[str]) -> dict[str, Any]:
    """Fetch current quotes for one or more tickers."""
    from .data import fetch_quotes

    if not tickers:
        return {"error": "No tickers provided"}
    if len(tickers) > 10:
        tickers = tickers[:10]

    try:
        quotes = fetch_quotes(tickers)
    except Exception as e:
        return {"error": f"Failed to fetch prices: {e}"}

    if not quotes:
        return {"error": "No results returned"}

    return {"count": len(quotes), "quotes": quotes}


def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    discord_id: int,
    capital: float,
) -> dict[str, Any]:
    """Execute a tool and return results."""
    if tool_name == "get_signals":
        return _get_signals(discord_id, capital)
    elif tool_name == "get_system_positions":
        return _get_system_positions(discord_id, capital)
    elif tool_name == "get_user_positions":
        return _get_user_positions(discord_id, capital)
    elif tool_name == "enter_trade":
        return _enter_trade(discord_id, tool_input, capital)
    elif tool_name == "exit_trade":
        return _exit_trade(discord_id, tool_input, capital)
    elif tool_name == "get_pnl":
        period = tool_input.get("period", "all")
        return _get_pnl(discord_id, period)
    elif tool_name == "set_capital":
        amount = tool_input.get("amount")
        return _set_capital(discord_id, amount)
    elif tool_name == "get_trade_history":
        ticker = tool_input.get("ticker")
        limit = tool_input.get("limit", 10)
        return _get_trade_history(discord_id, ticker, limit)
    elif tool_name == "get_portfolio":
        days = tool_input.get("days", 30)
        return _get_portfolio(days)
    elif tool_name == "get_health":
        return _get_health()
    elif tool_name == "run_scan":
        config = tool_input.get("config", "baseline")
        return _run_scan(config)
    elif tool_name == "run_retrain":
        # Mark as async task - will be handled specially by discord_bot
        tickers = tool_input.get("tickers")
        return {
            "_async_task": "retrain",
            "tickers": tickers,
            "message": "Starting model retraining... This will take 10-30+ minutes. You'll receive progress updates."
        }
    elif tool_name == "get_price":
        return _get_price(tool_input.get("tickers", []))
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Main Agent Function
# ---------------------------------------------------------------------------

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
    async_tasks = []  # Track async tasks to run
    client = _get_client()
    if not client:
        return (
            "Claude API not configured. Use slash commands instead.",
            conversation_history or [],
            []
        )

    # Ensure user exists and get their capital
    ensure_user(discord_id, discord_username)
    capital = get_capital(discord_id)

    # Initialize history
    if conversation_history is None:
        conversation_history = []

    # Build system prompt with context
    capital_context = f"${capital:,.0f}" if capital > 0 else "not set"
    system_prompt = f"""You are a helpful trading assistant for the Rally Scanner system.

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

    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Call Claude with tools
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
    max_tokens = int(os.environ.get("CLAUDE_MAX_TOKENS", "2048"))

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=conversation_history,
        tools=TOOLS,
    )

    # Handle tool use loop
    while response.stop_reason == "tool_use":
        # Extract and execute tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                logger.info(f"Executing tool: {block.name} with input: {block.input}")
                result = execute_tool(
                    block.name,
                    block.input,
                    discord_id,
                    capital
                )

                # Check if this is an async task
                if isinstance(result, dict) and "_async_task" in result:
                    async_tasks.append(result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                })

        # Add assistant response to history (convert TextBlocks to dicts)
        conversation_history.append({
            "role": "assistant",
            "content": [block.model_dump() for block in response.content]
        })

        # Add tool results
        conversation_history.append({
            "role": "user",
            "content": tool_results
        })

        # Continue conversation
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=conversation_history,
            tools=TOOLS,
        )

    # Extract final text response
    response_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            response_text += block.text

    # Add final response to history (convert TextBlocks to dicts)
    conversation_history.append({
        "role": "assistant",
        "content": [block.model_dump() for block in response.content]
    })

    logger.info(f"Claude response for {discord_username}: {response_text[:100]}...")

    return response_text, conversation_history, async_tasks
