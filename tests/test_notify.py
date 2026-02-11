"""Tests for notification formatters — no actual sending."""

from unittest.mock import patch

from rally.notify import (
    notify_signals,
    notify_exits,
    notify_retrain_complete,
    notify_error,
    notify,
    send_telegram,
    send_email,
    send_webhook,
)


def test_send_telegram_noop_without_env():
    """Telegram should no-op and return False when env vars are missing."""
    assert send_telegram("test") is False


def test_send_email_noop_without_env():
    """Email should no-op and return False when env vars are missing."""
    assert send_email("subj", "body") is False


def test_send_webhook_noop_without_env():
    """Webhook should no-op and return False when env vars are missing."""
    assert send_webhook({"test": True}) is False


@patch("rally.notify.send_telegram")
@patch("rally.notify.send_email")
def test_notify_signals_calls_backends(mock_email, mock_tg):
    signals = [
        {"ticker": "AAPL", "p_rally": 0.72, "close": 180.0,
         "size": 0.10, "range_low": 175.0, "atr_pct": 0.02},
        {"ticker": "MSFT", "p_rally": 0.65, "close": 400.0,
         "size": 0.08, "range_low": 390.0, "atr_pct": 0.015},
    ]
    notify_signals(signals)
    # Verify notify was called (which calls backends)
    assert mock_tg.called or mock_email.called


@patch("rally.notify.send_telegram")
@patch("rally.notify.send_email")
def test_notify_exits_calls_backends(mock_email, mock_tg):
    closed = [
        {"ticker": "AAPL", "exit_reason": "profit_target",
         "realized_pnl_pct": 4.5, "bars_held": 5},
    ]
    notify_exits(closed)
    assert mock_tg.called or mock_email.called


@patch("rally.notify.send_telegram")
@patch("rally.notify.send_email")
def test_notify_retrain_complete_calls_backends(mock_email, mock_tg):
    health = {"fresh_count": 14, "total_count": 14, "stale_count": 0}
    notify_retrain_complete(health, elapsed=120.0)
    assert mock_tg.called or mock_email.called


@patch("rally.notify.send_telegram")
@patch("rally.notify.send_email")
def test_notify_error_calls_backends(mock_email, mock_tg):
    notify_error("Test error", "Something went wrong")
    assert mock_tg.called or mock_email.called


@patch("rally.notify.notify")
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


@patch("rally.notify.notify")
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
