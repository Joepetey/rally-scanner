"""Tests for price alert embeds and alert-checking logic."""

from rally.bot.notify import _approaching_alert_embed, _price_alert_embed

# ---------------------------------------------------------------------------
# _price_alert_embed
# ---------------------------------------------------------------------------


def test_price_alert_embed_stop_breach():
    alerts = [{
        "ticker": "AAPL",
        "alert_type": "stop_breached",
        "current_price": 144.50,
        "level_price": 145.00,
        "level_name": "Stop",
        "entry_price": 150.00,
        "pnl_pct": -3.67,
    }]
    embed = _price_alert_embed(alerts)

    assert embed["title"] == "Price Alert (1)"
    assert embed["color"] == 0xFF0000  # red for stop breach
    assert "BREACHED" in embed["fields"][0]["value"]
    assert "$144.50" in embed["fields"][0]["value"]
    assert "-3.67%" in embed["fields"][0]["value"]
    assert embed["footer"]["text"]


def test_price_alert_embed_target_breach():
    alerts = [{
        "ticker": "NVDA",
        "alert_type": "target_breached",
        "current_price": 535.00,
        "level_price": 530.00,
        "level_name": "Target",
        "entry_price": 500.00,
        "pnl_pct": 7.00,
    }]
    embed = _price_alert_embed(alerts)

    assert embed["color"] == 0x00FF00  # green for target breach
    assert "Target BREACHED" in embed["fields"][0]["value"]
    assert "+7.00%" in embed["fields"][0]["value"]


def test_price_alert_embed_mixed():
    alerts = [
        {"ticker": "AAPL", "alert_type": "stop_breached", "current_price": 144,
         "level_price": 145, "level_name": "Stop", "entry_price": 150, "pnl_pct": -4.0},
        {"ticker": "NVDA", "alert_type": "target_breached", "current_price": 535,
         "level_price": 530, "level_name": "Target", "entry_price": 500, "pnl_pct": 7.0},
    ]
    embed = _price_alert_embed(alerts)

    assert embed["color"] == 0xFF0000  # red because any stop breach
    assert len(embed["fields"]) == 2


def test_price_alert_embed_empty():
    embed = _price_alert_embed([])
    assert embed["title"] == "Price Alert (0)"
    assert embed["fields"] == []


# ---------------------------------------------------------------------------
# _approaching_alert_embed
# ---------------------------------------------------------------------------


def test_approaching_alert_embed():
    alerts = [{
        "ticker": "MSFT",
        "alert_type": "near_stop",
        "current_price": 392.00,
        "level_price": 390.00,
        "level_name": "Stop",
        "distance_pct": 0.5,
        "entry_price": 400.00,
        "pnl_pct": -2.00,
    }]
    embed = _approaching_alert_embed(alerts)

    assert embed["title"] == "Price Warning (1)"
    assert embed["color"] == 0xFF8C00  # orange
    assert "Approaching" in embed["fields"][0]["value"]
    assert "0.5% away" in embed["fields"][0]["value"]
    assert "$392.00" in embed["fields"][0]["value"]


def test_approaching_alert_near_target():
    alerts = [{
        "ticker": "GOOG",
        "alert_type": "near_target",
        "current_price": 178.00,
        "level_price": 180.00,
        "level_name": "Target",
        "distance_pct": 1.1,
        "entry_price": 170.00,
        "pnl_pct": 4.71,
    }]
    embed = _approaching_alert_embed(alerts)

    assert "Approaching **Target**" in embed["fields"][0]["value"]
    assert "+4.71%" in embed["fields"][0]["value"]


def test_approaching_alert_embed_empty():
    embed = _approaching_alert_embed([])
    assert embed["title"] == "Price Warning (0)"
    assert embed["fields"] == []
