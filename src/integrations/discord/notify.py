"""Discord notification system for market-rally alerts.

Sends alerts via Discord using either bot token + channel ID or webhook URL.
Silently no-ops if not configured via environment variables.
"""

import json
import logging
import os
from urllib.request import Request, urlopen

from integrations.discord.embeds import (  # noqa: F401 — re-export
    _approaching_alert_embed,
    _cash_parking_embed,
    _error_embed,
    _exit_embed,
    _fill_confirmation_embed,
    _order_embed,
    _order_failure_embed,
    _positions_embed,
    _price_alert_embed,
    _regime_shift_embed,
    _retrain_embed,
    _risk_action_embed,
    _signal_embed,
    _stream_degraded_embed,
    _stream_recovered_embed,
)

logger = logging.getLogger(__name__)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Discord Backend
# ---------------------------------------------------------------------------

def send_discord(embeds: list[dict]) -> bool:
    """Send Discord embed(s) via bot token + channel, or webhook URL.

    Tries bot token + channel ID first (richer: multiple embeds).
    Falls back to DISCORD_WEBHOOK_URL if set.
    """
    token = _env("DISCORD_BOT_TOKEN")
    channel_id = _env("DISCORD_CHANNEL_ID")
    webhook_url = _env("DISCORD_WEBHOOK_URL")

    if token and channel_id:
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        payload = json.dumps({"embeds": embeds}).encode()
        try:
            req = Request(url, data=payload, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bot {token}",
            })
            with urlopen(req, timeout=10) as resp:
                resp.read()
            logger.info("Discord message sent via bot")
            return True
        except Exception as e:
            logger.error("Discord bot send failed: %s", e)
            return False

    if webhook_url:
        payload = json.dumps({"embeds": embeds}).encode()
        try:
            req = Request(webhook_url, data=payload, headers={
                "Content-Type": "application/json",
            })
            with urlopen(req, timeout=10) as resp:
                resp.read()
            logger.info("Discord message sent via webhook")
            return True
        except Exception as e:
            logger.error("Discord webhook failed: %s", e)
            return False

    return False
