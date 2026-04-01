"""Discord notification system for market-rally alerts.

Sends alerts via Discord using either bot token + channel ID or webhook URL.
Silently no-ops if not configured via environment variables.
"""

import json
import logging
import os
from urllib.request import Request, urlopen

from config import PARAMS as _P

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_GREEN = 0x00FF00
_RED = 0xFF0000
_ORANGE = 0xFF8C00
_RED_ORANGE = 0xFF4500
_SKY = 0x87CEEB
_GOLD = 0xFFD700
_GRAY = 0x95A5A6
_BLUE = 0x0099FF

_EMOJI_STOP = "\u26a0\ufe0f"   # ⚠️
_EMOJI_TARGET = "\u2705"        # ✅
_EMOJI_CLOSE = "\u274c"         # ❌
_EMOJI_WARN = "\u26a0\ufe0f"   # ⚠️

_FOOTER_INTRADAY = "Intraday alert \u2014 daily scan handles exits"
_FOOTER_INTRADAY_WARN = "Intraday warning \u2014 daily scan handles exits"


def _pnl_sign(pnl: float) -> str:
    return "+" if pnl >= 0 else ""


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


# ---------------------------------------------------------------------------
# Discord embed builders
# ---------------------------------------------------------------------------

def _signal_embed(signals: list[dict]) -> dict:
    """Build a Discord embed for new signals."""
    fields = []
    for s in sorted(signals, key=lambda x: x.get("p_rally", 0), reverse=True):
        close = s.get("close", 0)
        atr_pct = s.get("atr_pct", _P.default_atr_pct)
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
        "color": _GREEN,
        "fields": fields[:25],  # Discord limit
    }


def _exit_embed(closed: list[dict]) -> dict:
    """Build a Discord embed for position exits."""
    fields = []
    for c in closed:
        pnl = c.get("realized_pnl_pct") or 0
        fields.append({
            "name": c.get("ticker", "?"),
            "value": (
                f"Reason: {c.get('exit_reason', '?')}\n"
                f"PnL: **{_pnl_sign(pnl)}{pnl:.2f}%**\n"
                f"Bars held: {c.get('bars_held') or 0}"
            ),
            "inline": True,
        })
    # Use red if any loss, green if all wins
    any_loss = any((c.get("realized_pnl_pct") or 0) < 0 for c in closed)
    return {
        "title": f"Position Exits ({len(closed)})",
        "color": _RED if any_loss else _GREEN,
        "fields": fields[:25],
    }


def _retrain_embed(health: dict, elapsed: float) -> dict:
    """Build a Discord embed for retrain completion."""
    return {
        "title": "Retrain Complete",
        "color": _BLUE,
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


def _price_alert_embed(alerts: list[dict]) -> dict:
    """Build a Discord embed for price breach alerts.

    Each alert dict: {ticker, alert_type, current_price, level_price,
                      level_name, entry_price, pnl_pct}
    """
    fields = []
    for a in alerts:
        pnl = a.get("pnl_pct", 0)
        emoji = _EMOJI_STOP if "stop" in a["alert_type"] else _EMOJI_TARGET
        fields.append({
            "name": f"{emoji} {a['ticker']}",
            "value": (
                f"**{a['level_name']} BREACHED**\n"
                f"Level: ${a['level_price']:.2f}\n"
                f"Current: ${a['current_price']:.2f}\n"
                f"Entry: ${a['entry_price']:.2f}\n"
                f"PnL: {_pnl_sign(pnl)}{pnl:.2f}%"
            ),
            "inline": True,
        })
    any_stop = any("stop" in a["alert_type"] for a in alerts)
    return {
        "title": f"Price Alert ({len(alerts)})",
        "color": _RED if any_stop else _GREEN,
        "fields": fields[:25],
        "footer": {"text": _FOOTER_INTRADAY},
    }


def _approaching_alert_embed(alerts: list[dict]) -> dict:
    """Build a Discord embed for 'approaching level' warnings.

    Each alert dict: {ticker, alert_type, current_price, level_price,
                      level_name, distance_pct, entry_price, pnl_pct}
    """
    fields = []
    for a in alerts:
        pnl = a.get("pnl_pct", 0)
        fields.append({
            "name": a["ticker"],
            "value": (
                f"Approaching **{a['level_name']}**\n"
                f"Level: ${a['level_price']:.2f} ({a['distance_pct']:.1f}% away)\n"
                f"Current: ${a['current_price']:.2f}\n"
                f"PnL: {_pnl_sign(pnl)}{pnl:.2f}%"
            ),
            "inline": True,
        })
    return {
        "title": f"Price Warning ({len(alerts)})",
        "color": _ORANGE,
        "fields": fields[:25],
        "footer": {"text": _FOOTER_INTRADAY_WARN},
    }


def _order_embed(results: list, equity: float) -> dict:
    """Build a Discord embed for successful Alpaca order executions."""
    fields = []
    for r in results:
        parts = [f"Side: **{r['side'].upper()}**", f"Qty: {r['qty']}"]
        if r.get("fill_price"):
            parts.append(f"Fill: ${r['fill_price']:.2f}")
        if r.get("order_id"):
            parts.append(f"Order: `{r['order_id'][:8]}`")
        fields.append({
            "name": r["ticker"],
            "value": "\n".join(parts),
            "inline": True,
        })
    return {
        "title": f"Alpaca Orders ({len(results)})",
        "color": _SKY,
        "fields": fields[:25],
        "footer": {"text": f"Account equity: ${equity:,.0f}"},
    }


def _fill_confirmation_embed(fills: list[dict]) -> dict:
    """Build a Discord embed for confirmed entry fill prices.

    Each dict: {ticker, fill_price, qty, stop_price, target_price}
    """
    fields = []
    for f in fills:
        fields.append({
            "name": f["ticker"],
            "value": (
                f"Fill confirmed: **${f['fill_price']:.2f}**\n"
                f"Qty: {f.get('qty', '?')}\n"
                f"Stop: ${f.get('stop_price', 0):.2f}  Target: ${f.get('target_price', 0):.2f}"
            ),
            "inline": True,
        })
    return {
        "title": f"Entry Fills Confirmed ({len(fills)})",
        "color": _SKY,
        "fields": fields[:25],
        "footer": {"text": "Intraday fill confirmation"},
    }


def _order_failure_embed(results: list) -> dict:
    """Build a Discord embed for failed Alpaca order executions."""
    fields = []
    for r in results:
        fields.append({
            "name": r["ticker"],
            "value": f"Side: {r['side'].upper()}\nError: {r.get('error', '?')}",
            "inline": True,
        })
    return {
        "title": f"Alpaca Order Failures ({len(results)})",
        "color": _RED_ORANGE,
        "fields": fields[:25],
        "footer": {"text": "System positions unaffected"},
    }


def _regime_shift_embed(transitions: list[dict]) -> dict:
    """Build a Discord embed for regime shift alerts.

    Each transition dict: {ticker, prev_regime, new_regime,
                           p_compressed, p_normal, p_expanding}
    """
    fields = []
    for t in transitions:
        fields.append({
            "name": t["ticker"],
            "value": (
                f"**{t['prev_regime']}** \u2192 **{t['new_regime']}**\n"
                f"P(exp): {t['p_expanding']:.0%}\n"
                f"P(comp): {t['p_compressed']:.0%}"
            ),
            "inline": True,
        })
    any_expanding = any(t["new_regime"] == "expanding" for t in transitions)
    return {
        "title": f"Regime Shift ({len(transitions)})",
        "color": _RED if any_expanding else _ORANGE,
        "fields": fields[:25],
        "footer": {"text": "HMM volatility regime transition detected"},
    }


def _risk_action_embed(results: list[dict]) -> dict:
    """Build a Discord embed for proactive risk actions.

    Each result dict: {ticker, action, success, reason, ...}
    """
    fields = []
    for r in results:
        if r.get("skipped"):
            continue
        if r["action"] == "close":
            pnl = r.get("pnl_pct", 0)
            fields.append({
                "name": f"{_EMOJI_CLOSE} {r['ticker']}",
                "value": (
                    f"**CLOSED** (risk reduction)\n"
                    f"PnL: {_pnl_sign(pnl)}{pnl:.2f}%\n"
                    f"Reason: {r['reason']}"
                ),
                "inline": True,
            })
        elif r["action"] == "tighten":
            old = r.get("old_stop", 0)
            new = r.get("new_stop", 0)
            fields.append({
                "name": f"{_EMOJI_WARN} {r['ticker']}",
                "value": (
                    f"**Stop tightened**\n"
                    f"${old:.2f} \u2192 ${new:.2f}\n"
                    f"Reason: {r['reason']}"
                ),
                "inline": True,
            })
    if not fields:
        return {"title": "Risk Actions", "color": _GRAY, "description": "No actions needed"}
    any_close = any(r["action"] == "close" for r in results if not r.get("skipped"))
    return {
        "title": f"Proactive Risk Actions ({len(fields)})",
        "color": _RED if any_close else _ORANGE,
        "fields": fields[:25],
        "footer": {"text": "Automated risk management"},
    }


def _positions_embed(positions: list[dict]) -> dict:
    """Build a Discord embed showing open positions."""
    if not positions:
        return {
            "title": "Open Positions (0)",
            "color": _GRAY,
            "description": "No open positions",
        }

    sorted_pos = sorted(
        positions,
        key=lambda x: x.get("unrealized_pnl_pct", 0),
        reverse=True,
    )
    lines = []
    for p in sorted_pos:
        pnl = p.get("unrealized_pnl_pct", 0)
        sign = "+" if pnl >= 0 else ""
        size = p.get("size", 0)
        current = p.get("current_price", p.get("entry_price", 0))
        lines.append(
            f"`{p['ticker']:<6s}` "
            f"${current:>7.2f}  "
            f"**{sign}{pnl:.2f}%**  "
            f"Stop ${p['stop_price']:.2f}  "
            f"Target ${p['target_price']:.2f}  "
            f"{p.get('bars_held', 0)}d  "
            f"{size:.0%}"
        )
    any_loss = any(p.get("unrealized_pnl_pct", 0) < 0 for p in positions)
    return {
        "title": f"Open Positions ({len(positions)})",
        "color": _ORANGE if any_loss else _GREEN,
        "description": "\n".join(lines),
    }


def _cash_parking_embed(action: str, result, equity: float, idle_fraction: float) -> dict:
    """Build a Discord embed for SGOV cash parking actions.

    action: 'park' | 'unpark'
    result: OrderResult
    """
    label = "Gap" if action == "unpark" else "Idle"
    parts = [f"Qty: {result.qty}"]
    if idle_fraction > 0:
        parts.append(f"{label}: {idle_fraction:.1%}")
    if result.fill_price:
        parts.append(f"Fill: ${result.fill_price:.2f}")
    if result.order_id:
        parts.append(f"Order: `{result.order_id[:8]}`")
    title = "SGOV Parked — Cash Management" if action == "park" else "SGOV Sold — Capital Freed"
    return {
        "title": title,
        "color": _GOLD,
        "description": "\n".join(parts),
        "footer": {"text": f"Account equity: ${equity:,.0f}"},
    }


def _error_embed(title: str, details: str) -> dict:
    """Build a Discord embed for error alerts."""
    return {
        "title": f"Error: {title}",
        "color": _RED,
        "description": details[:4096],  # Discord limit
    }


def _stream_degraded_embed(disconnected_minutes: int) -> dict:
    """Build a Discord embed for stream degradation (polling-only fallback)."""
    return {
        "title": "Stream Degraded — Polling-Only Mode",
        "color": _ORANGE,
        "description": (
            f"The Alpaca WebSocket stream has been disconnected for **{disconnected_minutes} min**.\n"  # noqa: E501
            "Price alert latency has degraded from seconds to ~15 minutes.\n"
            "Reconnect attempts are ongoing — check Railway logs for details."
        ),
    }


def _stream_recovered_embed(downtime_minutes: int) -> dict:
    """Build a Discord embed for stream recovery after a degradation period."""
    return {
        "title": "Stream Recovered",
        "color": _GREEN,
        "description": (
            f"The Alpaca WebSocket stream has reconnected after **{downtime_minutes} min** of downtime.\n"  # noqa: E501
            "Real-time price alerts are active again."
        ),
    }

