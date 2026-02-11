"""Tests for Discord embed builders in notify.py."""

from rally.notify import _error_embed, _exit_embed, _retrain_embed, _signal_embed


def test_signal_embed_structure():
    signals = [
        {"ticker": "AAPL", "p_rally": 0.72, "close": 185.0, "atr_pct": 0.02,
         "size": 0.10, "range_low": 180.0},
        {"ticker": "MSFT", "p_rally": 0.65, "close": 420.0, "atr_pct": 0.015,
         "size": 0.08, "range_low": 410.0},
    ]
    embed = _signal_embed(signals)

    assert embed["title"] == "New Signals (2)"
    assert embed["color"] == 0x00FF00
    assert len(embed["fields"]) == 2
    # Sorted by p_rally desc — AAPL first
    assert embed["fields"][0]["name"] == "AAPL"
    assert embed["fields"][1]["name"] == "MSFT"


def test_signal_embed_field_content():
    signals = [{"ticker": "SPY", "p_rally": 0.55, "close": 500.0, "atr_pct": 0.01,
                "size": 0.05, "range_low": 495.0}]
    embed = _signal_embed(signals)

    field = embed["fields"][0]
    assert "P(rally):" in field["value"]
    assert "$500.00" in field["value"]
    assert "$495.00" in field["value"]
    assert field["inline"] is True


def test_signal_embed_empty():
    embed = _signal_embed([])
    assert embed["title"] == "New Signals (0)"
    assert embed["fields"] == []


def test_exit_embed_win():
    closed = [{"ticker": "AAPL", "realized_pnl_pct": 5.2, "exit_reason": "profit_target",
               "bars_held": 7}]
    embed = _exit_embed(closed)

    assert embed["color"] == 0x00FF00  # green for win
    assert len(embed["fields"]) == 1
    assert "+5.20%" in embed["fields"][0]["value"]


def test_exit_embed_loss():
    closed = [{"ticker": "TSLA", "realized_pnl_pct": -3.5, "exit_reason": "stop",
               "bars_held": 2}]
    embed = _exit_embed(closed)

    assert embed["color"] == 0xFF0000  # red for loss
    assert "-3.50%" in embed["fields"][0]["value"]


def test_exit_embed_mixed():
    closed = [
        {"ticker": "AAPL", "realized_pnl_pct": 5.0, "exit_reason": "target", "bars_held": 5},
        {"ticker": "TSLA", "realized_pnl_pct": -2.0, "exit_reason": "stop", "bars_held": 3},
    ]
    embed = _exit_embed(closed)
    assert embed["color"] == 0xFF0000  # red because any loss


def test_retrain_embed():
    health = {"fresh_count": 800, "total_count": 843, "stale_count": 43}
    embed = _retrain_embed(health, elapsed=120.5)

    assert embed["title"] == "Retrain Complete"
    assert embed["color"] == 0x0099FF
    assert len(embed["fields"]) == 3
    assert "800/843" in embed["fields"][0]["value"]
    assert "43" in embed["fields"][1]["value"]
    assert "121s" in embed["fields"][2]["value"] or "120s" in embed["fields"][2]["value"]


def test_error_embed():
    embed = _error_embed("Model staleness", "50% of models are stale")

    assert embed["title"] == "Error: Model staleness"
    assert embed["color"] == 0xFF0000
    assert "50% of models are stale" in embed["description"]


def test_error_embed_truncates_long_description():
    long_text = "x" * 5000
    embed = _error_embed("Big error", long_text)
    assert len(embed["description"]) <= 4096
