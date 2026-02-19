"""Tests for notification formatters — no actual sending."""

from unittest.mock import patch

from rally.bot.notify import (
    notify_error,
    notify_exits,
    notify_retrain_complete,
    notify_signals,
    send_discord,
)


def test_send_discord_noop_without_env():
    """Discord should no-op and return False when env vars are missing."""
    assert send_discord([{"test": True}]) is False


@patch("rally.bot.notify.send_discord")
def test_notify_signals_calls_discord(mock_discord):
    signals = [
        {"ticker": "AAPL", "p_rally": 0.72, "close": 180.0,
         "size": 0.10, "range_low": 175.0, "atr_pct": 0.02},
        {"ticker": "MSFT", "p_rally": 0.65, "close": 400.0,
         "size": 0.08, "range_low": 390.0, "atr_pct": 0.015},
    ]
    notify_signals(signals)
    assert mock_discord.called


@patch("rally.bot.notify.send_discord")
def test_notify_exits_calls_discord(mock_discord):
    closed = [
        {"ticker": "AAPL", "exit_reason": "profit_target",
         "realized_pnl_pct": 4.5, "bars_held": 5},
    ]
    notify_exits(closed)
    assert mock_discord.called


@patch("rally.bot.notify.send_discord")
def test_notify_retrain_complete_calls_discord(mock_discord):
    health = {"fresh_count": 14, "total_count": 14, "stale_count": 0}
    notify_retrain_complete(health, elapsed=120.0)
    assert mock_discord.called


@patch("rally.bot.notify.send_discord")
def test_notify_error_calls_discord(mock_discord):
    notify_error("Test error", "Something went wrong")
    assert mock_discord.called


@patch("rally.bot.notify.notify")
def test_signal_formatter_content(mock_notify):
    signals = [
        {"ticker": "AAPL", "p_rally": 0.72, "close": 180.0,
         "size": 0.10, "range_low": 175.0, "atr_pct": 0.02},
    ]
    notify_signals(signals)
    call_args = mock_notify.call_args
    body = call_args[0][1]  # second positional arg
    assert "AAPL" in body
    assert "SIGNALS" in body


@patch("rally.bot.notify.notify")
def test_exit_formatter_content(mock_notify):
    closed = [
        {"ticker": "MSFT", "exit_reason": "stop",
         "realized_pnl_pct": -1.5, "bars_held": 3},
    ]
    notify_exits(closed)
    call_args = mock_notify.call_args
    body = call_args[0][1]
    assert "MSFT" in body
    assert "stop" in body
