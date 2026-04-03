"""Discord HTTP transport — sends embeds via bot token or webhook."""

import json
import logging
import os
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def send_discord(embeds: list[dict]) -> bool:
    """Send Discord embed(s) via bot token + channel, or webhook URL.

    Tries bot token + channel ID first (richer: multiple embeds).
    Falls back to DISCORD_WEBHOOK_URL if set.
    """
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    channel_id = os.environ.get("DISCORD_CHANNEL_ID", "")
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

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
