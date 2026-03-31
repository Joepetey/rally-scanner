"""Tests for Discord agent tool validation — integrations/discord/agent.py (MIC-99).

Tests tool execution functions, dollar metrics, capital validation, and tool schema.
"""

import sys
from unittest.mock import MagicMock, patch

# Stub anthropic before importing agent (it imports at module level)
sys.modules.setdefault("anthropic", MagicMock())

from integrations.discord.agent import (  # noqa: E402
    TOOLS,
    _dollar_metrics,
    _set_capital,
    execute_tool,
)

DISCORD_ID = 12345


# ---------------------------------------------------------------------------
# _dollar_metrics
# ---------------------------------------------------------------------------


class TestDollarMetrics:

    def test_basic_calculation(self):
        metrics = _dollar_metrics(capital=10000, size=0.05, entry=100.0)
        assert metrics["dollar_allocation"] == 500.0

    def test_with_stop(self):
        metrics = _dollar_metrics(capital=10000, size=0.05, entry=100.0, stop=95.0)
        assert metrics["dollar_allocation"] == 500.0
        # dollar_risk = 500 * (100-95)/100 = 25
        assert metrics["dollar_risk"] == 25.0

    def test_zero_capital(self):
        metrics = _dollar_metrics(capital=0, size=0.05, entry=100.0)
        assert metrics == {}

    def test_zero_size(self):
        metrics = _dollar_metrics(capital=10000, size=0, entry=100.0)
        assert metrics == {}

    def test_no_stop(self):
        metrics = _dollar_metrics(capital=10000, size=0.05, entry=100.0, stop=None)
        assert "dollar_allocation" in metrics
        assert "dollar_risk" not in metrics

    def test_stop_above_entry_no_risk(self):
        """When stop >= entry, no dollar_risk calculated (condition: entry > stop)."""
        metrics = _dollar_metrics(capital=10000, size=0.05, entry=100.0, stop=105.0)
        assert "dollar_risk" not in metrics


# ---------------------------------------------------------------------------
# _set_capital
# ---------------------------------------------------------------------------


class TestSetCapital:

    @patch("integrations.discord.agent.set_capital")
    def test_positive_capital(self, mock_set):
        result = _set_capital(DISCORD_ID, 10000.0)
        assert result["success"] is True
        assert result["capital"] == 10000.0
        mock_set.assert_called_once_with(DISCORD_ID, 10000.0)

    def test_zero_capital_rejected(self):
        result = _set_capital(DISCORD_ID, 0)
        assert "error" in result

    def test_negative_capital_rejected(self):
        result = _set_capital(DISCORD_ID, -5000)
        assert "error" in result


# ---------------------------------------------------------------------------
# _enter_trade
# ---------------------------------------------------------------------------


class TestEnterTrade:

    @patch("integrations.discord.agent.get_merged_positions_sync")
    @patch("integrations.discord.agent.open_trade", return_value=1)
    def test_basic_entry(self, mock_open, mock_positions):
        from integrations.discord.agent import _enter_trade

        mock_positions.return_value = {"positions": []}
        result = _enter_trade(DISCORD_ID, {"ticker": "aapl", "price": 150.0}, 10000.0)
        assert result["ticker"] == "AAPL"
        assert result["entry_price"] == 150.0
        assert result["trade_id"] == 1

    @patch("integrations.discord.agent.get_merged_positions_sync")
    @patch("integrations.discord.agent.open_trade", return_value=2)
    def test_auto_fills_from_system_signal(self, mock_open, mock_positions):
        from integrations.discord.agent import _enter_trade

        mock_positions.return_value = {
            "positions": [{
                "ticker": "NVDA", "size": 0.10,
                "stop_price": 140.0, "target_price": 170.0,
            }],
        }
        result = _enter_trade(
            DISCORD_ID,
            {"ticker": "nvda", "price": 155.0},
            10000.0,
        )
        assert result["ticker"] == "NVDA"
        # Size auto-filled from system signal
        mock_open.assert_called_once()
        call_kwargs = mock_open.call_args
        assert call_kwargs[1]["size"] == 0.10
        assert call_kwargs[1]["stop_price"] == 140.0
        assert call_kwargs[1]["target_price"] == 170.0


# ---------------------------------------------------------------------------
# _exit_trade
# ---------------------------------------------------------------------------


class TestExitTrade:

    @patch("integrations.discord.agent.close_trade")
    def test_successful_exit(self, mock_close):
        from integrations.discord.agent import _exit_trade

        mock_close.return_value = {
            "ticker": "AAPL", "entry_price": 100.0, "exit_price": 110.0,
            "pnl_pct": 10.0, "size": 0.10, "entry_date": "2024-01-10",
            "exit_date": "2024-01-20",
        }
        result = _exit_trade(DISCORD_ID, {"ticker": "aapl", "price": 110.0}, 10000.0)
        assert result["pnl_pct"] == 10.0
        assert result["ticker"] == "AAPL"

    @patch("integrations.discord.agent.close_trade", return_value=None)
    def test_no_open_trade(self, mock_close):
        from integrations.discord.agent import _exit_trade

        result = _exit_trade(DISCORD_ID, {"ticker": "aapl", "price": 110.0}, 10000.0)
        assert "error" in result


# ---------------------------------------------------------------------------
# _get_system_positions
# ---------------------------------------------------------------------------


class TestGetSystemPositions:

    @patch("integrations.discord.agent.get_merged_positions_sync")
    def test_with_positions(self, mock_positions):
        from integrations.discord.agent import _get_system_positions

        mock_positions.return_value = {
            "positions": [
                {"ticker": "AAPL", "entry_price": 100.0, "current_price": 105.0,
                 "stop_price": 95.0, "target_price": 110.0, "size": 0.10,
                 "unrealized_pnl_pct": 5.0, "bars_held": 3},
                {"ticker": "MSFT", "entry_price": 200.0, "current_price": 195.0,
                 "stop_price": 190.0, "target_price": 220.0, "size": 0.08,
                 "unrealized_pnl_pct": -2.5, "bars_held": 1},
            ],
        }
        result = _get_system_positions(DISCORD_ID, 10000.0)
        assert result["count"] == 2
        assert len(result["positions"]) == 2
        # Dollar metrics calculated
        aapl = result["positions"][0]
        assert aapl["dollar_allocation"] == 1000.0  # 10000 * 0.10

    @patch("integrations.discord.agent.get_merged_positions_sync")
    def test_no_positions(self, mock_positions):
        from integrations.discord.agent import _get_system_positions

        mock_positions.return_value = {"positions": []}
        result = _get_system_positions(DISCORD_ID, 10000.0)
        assert "message" in result


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------


class TestExecuteTool:

    @patch("integrations.discord.agent._get_price")
    def test_dispatches_get_price(self, mock_gp):
        mock_gp.return_value = {"count": 1, "quotes": {}}
        execute_tool("get_price", {"tickers": ["HD"]}, DISCORD_ID, 10000)
        mock_gp.assert_called_once_with(["HD"])

    @patch("integrations.discord.agent._set_capital")
    def test_dispatches_set_capital(self, mock_sc):
        mock_sc.return_value = {"success": True}
        execute_tool("set_capital", {"amount": 5000}, DISCORD_ID, 10000)
        mock_sc.assert_called_once_with(DISCORD_ID, 5000)

    def test_unknown_tool(self):
        result = execute_tool("nonexistent_tool", {}, DISCORD_ID, 10000)
        assert "error" in result

    @patch("integrations.discord.agent._get_watchlist")
    def test_dispatches_get_watchlist(self, mock_wl):
        mock_wl.return_value = {"tickers": []}
        execute_tool("get_watchlist", {}, DISCORD_ID, 10000)
        mock_wl.assert_called_once()

    def test_retrain_returns_async_task(self):
        result = execute_tool("run_retrain", {}, DISCORD_ID, 10000)
        assert result["_async_task"] == "retrain"


# ---------------------------------------------------------------------------
# Tool schema validation
# ---------------------------------------------------------------------------


class TestToolSchema:

    def test_all_tools_have_required_fields(self):
        for tool in TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing 'description'"
            assert "input_schema" in tool, f"Tool {tool['name']} missing 'input_schema'"
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_tool_names_are_unique(self):
        names = [t["name"] for t in TOOLS]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_enter_trade_requires_ticker_and_price(self):
        enter = next(t for t in TOOLS if t["name"] == "enter_trade")
        assert "ticker" in enter["input_schema"]["required"]
        assert "price" in enter["input_schema"]["required"]

    def test_exit_trade_requires_ticker_and_price(self):
        exit_t = next(t for t in TOOLS if t["name"] == "exit_trade")
        assert "ticker" in exit_t["input_schema"]["required"]
        assert "price" in exit_t["input_schema"]["required"]

    def test_get_price_requires_tickers(self):
        gp = next(t for t in TOOLS if t["name"] == "get_price")
        assert "tickers" in gp["input_schema"]["required"]

    def test_set_capital_requires_amount(self):
        sc = next(t for t in TOOLS if t["name"] == "set_capital")
        assert "amount" in sc["input_schema"]["required"]
