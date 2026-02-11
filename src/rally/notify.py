"""Multi-backend notification system for market-rally alerts.

Supports Telegram (primary), email (SMTP), and generic webhooks.
Each backend silently no-ops if not configured via environment variables.
"""

import json
import logging
import os
import smtplib
from email.mime.text import MIMEText
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def send_telegram(text: str) -> bool:
    """Send Telegram message via Bot API. Returns True on success."""
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("TELEGRAM_CHAT_ID")

    if not all([token, chat_id]):
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }).encode()

    try:
        req = Request(url, data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            resp.read()
        logger.info("Telegram message sent")
        return True
    except Exception as e:
        logger.error(f"Telegram failed: {e}")
        return False


def send_email(subject: str, body: str) -> bool:
    """Send email via SMTP/TLS. Returns True on success."""
    host = _env("SMTP_HOST")
    port = int(_env("SMTP_PORT", "587"))
    user = _env("SMTP_USER")
    password = _env("SMTP_PASSWORD")
    to_addr = _env("NOTIFY_EMAIL")

    if not all([host, user, password, to_addr]):
        return False

    msg = MIMEText(body, "plain")
    msg["Subject"] = f"[Rally] {subject}"
    msg["From"] = user
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        logger.info(f"Email sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False


def send_webhook(payload: dict) -> bool:
    """POST JSON to a webhook URL. Returns True on success."""
    url = _env("WEBHOOK_URL")
    if not url:
        return False

    data = json.dumps(payload).encode()
    try:
        req = Request(url, data=data,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            resp.read()
        logger.info("Webhook delivered")
        return True
    except Exception as e:
        logger.error(f"Webhook failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def notify(subject: str, body: str, payload: dict | None = None) -> None:
    """Send notification via all configured backends."""
    send_telegram(body)
    send_email(subject, body)
    if payload:
        send_webhook(payload)


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
    notify("New Signals", body, {"type": "signals", "count": len(signals)})


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
    notify("Position Exits", body, {"type": "exits", "count": len(closed)})


def notify_retrain_complete(health: dict, elapsed: float) -> None:
    """Notify on retrain completion with health summary."""
    body = (
        f"*RETRAIN COMPLETE*\n"
        f"  Models: {health.get('fresh_count', 0)}/{health.get('total_count', 0)} fresh\n"
        f"  Stale: {health.get('stale_count', 0)}\n"
        f"  Elapsed: {elapsed:.0f}s"
    )
    notify("Retrain Complete", body, {"type": "retrain", "health": health})


def notify_error(title: str, details: str) -> None:
    """Send error/warning notification."""
    body = f"*ERROR: {title}*\n{details}"
    notify(f"Error: {title}", body, {"type": "error", "title": title})
