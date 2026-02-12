"""Discord notification system for market-rally alerts.

Sends alerts via Discord using either bot token + channel ID or webhook URL.
Silently no-ops if not configured via environment variables.
"""

import json
import logging
import os
from urllib.request import Request, urlopen

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
            logger.error(f"Discord bot send failed: {e}")
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
            logger.error(f"Discord webhook failed: {e}")
            return False

    return False


# ---------------------------------------------------------------------------
# Discord embed builders
# ---------------------------------------------------------------------------

def _signal_embed(signals: list[dict]) -> dict:
    """Build a Discord embed for new signals."""
    fields = []
    for s in sorted(signals, key=lambda x: x.get("p_rally", 0), reverse=True):
        close = s.get("close", 0)
        atr_pct = s.get("atr_pct", 0.02)
        target = close * (1 + 2.0 * atr_pct)
        fields.append({
            "name": s.get("ticker", "?"),
            "value": (
                f"P(rally): **{s.get('p_rally', 0):.0%}**\n"
                f"Price: ${close:.2f}\n"
                f"Stop: ${s.get('range_low', 0):.2f}\n"
                f"Target: ${target:.2f}\n"
                f"Size: {s.get('size', 0):.0%}"
            ),
            "inline": True,
        })
    return {
        "title": f"New Signals ({len(signals)})",
        "color": 0x00FF00,
        "fields": fields[:25],  # Discord limit
    }


def _exit_embed(closed: list[dict]) -> dict:
    """Build a Discord embed for position exits."""
    fields = []
    for c in closed:
        pnl = c.get("realized_pnl_pct", 0)
        sign = "+" if pnl >= 0 else ""
        fields.append({
            "name": c.get("ticker", "?"),
            "value": (
                f"Reason: {c.get('exit_reason', '?')}\n"
                f"PnL: **{sign}{pnl:.2f}%**\n"
                f"Bars held: {c.get('bars_held', 0)}"
            ),
            "inline": True,
        })
    # Use red if any loss, green if all wins
    any_loss = any(c.get("realized_pnl_pct", 0) < 0 for c in closed)
    return {
        "title": f"Position Exits ({len(closed)})",
        "color": 0xFF0000 if any_loss else 0x00FF00,
        "fields": fields[:25],
    }


def _retrain_embed(health: dict, elapsed: float) -> dict:
    """Build a Discord embed for retrain completion."""
    return {
        "title": "Retrain Complete",
        "color": 0x0099FF,
        "fields": [
            {"name": "Models", "value": (
                f"{health.get('fresh_count', 0)}/{health.get('total_count', 0)} fresh"
            ), "inline": True},
            {"name": "Stale", "value": str(
                health.get("stale_count", 0)
            ), "inline": True},
            {"name": "Elapsed", "value": f"{elapsed:.0f}s", "inline": True},
        ],
    }


def _error_embed(title: str, details: str) -> dict:
    """Build a Discord embed for error alerts."""
    return {
        "title": f"Error: {title}",
        "color": 0xFF0000,
        "description": details[:4096],  # Discord limit
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def notify(
    subject: str, body: str,
    payload: dict | None = None, discord_embeds: list[dict] | None = None,
) -> None:
    """Send notification via Discord (if configured)."""
    if discord_embeds:
        send_discord(discord_embeds)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def notify_signals(signals: list[dict]) -> None:
    """Format and send new entry signal alerts."""
    lines = [f"*NEW SIGNALS* ({len(signals)})\n"]
    for s in sorted(signals, key=lambda x: x.get("p_rally", 0), reverse=True):
        close = s.get("close", 0)
        atr_pct = s.get("atr_pct", 0.02)
        target = close * (1 + 2.0 * atr_pct)
        lines.append(
            f"  {s.get('ticker', '?'):6s}  P={s.get('p_rally', 0):.0%}  "
            f"${close:.2f}  Size={s.get('size', 0):.0%}  "
            f"Stop=${s.get('range_low', 0):.2f}  Target=${target:.2f}"
        )
    body = "\n".join(lines)
    notify(
        "New Signals", body,
        discord_embeds=[_signal_embed(signals)],
    )


def notify_exits(closed: list[dict]) -> None:
    """Format and send position exit alerts."""
    lines = [f"*EXITS* ({len(closed)})\n"]
    for c in closed:
        pnl = c.get("realized_pnl_pct", 0)
        sign = "+" if pnl >= 0 else ""
        lines.append(
            f"  {c.get('ticker', '?'):6s}  {c.get('exit_reason', '?'):15s}  "
            f"PnL: {sign}{pnl:.2f}%  ({c.get('bars_held', 0)} bars)"
        )
    body = "\n".join(lines)
    notify(
        "Position Exits", body,
        discord_embeds=[_exit_embed(closed)],
    )


def notify_retrain_complete(health: dict, elapsed: float) -> None:
    """Notify on retrain completion with health summary."""
    body = (
        f"*RETRAIN COMPLETE*\n"
        f"  Models: {health.get('fresh_count', 0)}/{health.get('total_count', 0)} fresh\n"
        f"  Stale: {health.get('stale_count', 0)}\n"
        f"  Elapsed: {elapsed:.0f}s"
    )
    notify(
        "Retrain Complete", body,
        discord_embeds=[_retrain_embed(health, elapsed)],
    )


def notify_error(title: str, details: str) -> None:
    """Send error/warning notification."""
    body = f"*ERROR: {title}*\n{details}"
    notify(
        f"Error: {title}", body,
        discord_embeds=[_error_embed(title, details)],
    )
