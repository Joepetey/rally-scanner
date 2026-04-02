"""Trading event → Discord embed dispatch.

Each handler converts a typed TradingScheduler event into one or more
Discord embeds and sends them via the provided send_alert callback.
"""

import logging
import os
import traceback
from collections.abc import Awaitable, Callable

import discord
from discord.ext import commands

from db.ops.events import log_discord_message
from trading.engine import (
    AlertEvent,
    ExitResult,
    HousekeepingResult,
    RegimeEvent,
    RetrainResult,
    RiskActionEvent,
    ScanResult,
    StreamDegradedEvent,
    StreamRecoveredEvent,
    WatchlistEvent,
)

from .notify import (
    _approaching_alert_embed,
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


async def _handle_alert_event(
    event: AlertEvent, send_alert: Callable,
) -> None:
    if event.alert_type in ("stop_breached", "target_breached"):
        embed = discord.Embed.from_dict(_price_alert_embed([event.model_dump()]))
        await send_alert(embed, "price_alert")
    elif event.alert_type in ("near_stop", "near_target"):
        embed = discord.Embed.from_dict(
            _approaching_alert_embed([event.model_dump()]),
        )
        await send_alert(embed, "price_alert")


async def _handle_exit_result(
    event: ExitResult, send_alert: Callable,
) -> None:
    embed = discord.Embed.from_dict(_exit_embed([event.model_dump()]))
    await send_alert(embed, "exit")


async def _handle_housekeeping_result(
    event: HousekeepingResult, send_alert: Callable,
) -> None:
    if event.fills_confirmed:
        fills = [f.model_dump() for f in event.fills_confirmed]
        embed = discord.Embed.from_dict(_fill_confirmation_embed(fills))
        await send_alert(embed, "order")


async def _handle_scan_result(
    event: ScanResult, send_alert: Callable,
) -> None:
    if event.error:
        embed = discord.Embed(
            title="\u26a0\ufe0f Daily Scan Skipped \u2014 No Trained Models",
            description=(
                "No trained models were found on the volume.\n"
                "Run retrain before the next scan: ask me to"
                " **retrain** or trigger it manually."
            ),
            color=discord.Color.orange(),
        )
        await send_alert(embed, "scan_error")
        return

    if event.signals:
        sig_embed = discord.Embed.from_dict(_signal_embed(event.signals))
        if event.scan_type == "premarket":
            sig_embed.title = (
                f"Pre-Market Signals ({len(event.signals)})"
                " \u2014 executing at open"
            )
        elif event.scan_type not in (
            "daily", "morning", "cascade", "post_retrain",
        ):
            sig_embed.title = f"Mid-day Signal ({len(event.signals)})"
            sig_embed.set_footer(text="From watchlist mid-day scan")
        await send_alert(sig_embed, "signal")

    pos = event.positions_summary.get("positions", [])
    await send_alert(
        discord.Embed.from_dict(_positions_embed(pos)), "positions",
    )

    if event.exits:
        await send_alert(
            discord.Embed.from_dict(_exit_embed(event.exits)), "exit",
        )

    entry_ok = [
        o for o in event.orders
        if o.get("side") == "buy" and o.get("success")
    ]
    entry_fail = [
        o for o in event.orders
        if o.get("side") == "buy"
        and not o.get("success")
        and not o.get("skipped")
    ]
    exit_ok = [
        o for o in event.orders
        if o.get("side") == "sell" and o.get("success")
    ]

    if entry_ok and event.equity:
        await send_alert(
            discord.Embed.from_dict(_order_embed(entry_ok, event.equity)),
            "order",
        )
    if entry_fail:
        await send_alert(
            discord.Embed.from_dict(_order_failure_embed(entry_fail)),
            "order_failure",
        )
    if exit_ok and event.equity:
        await send_alert(
            discord.Embed.from_dict(_order_embed(exit_ok, event.equity)),
            "order",
        )


async def _handle_watchlist_event(
    event: WatchlistEvent, send_alert: Callable,
) -> None:
    if event.signals:
        sig_embed = discord.Embed.from_dict(_signal_embed(event.signals))
        sig_embed.title = f"Mid-day Signal ({len(event.signals)})"
        sig_embed.set_footer(text="From watchlist mid-day scan")
        await send_alert(sig_embed, "signal")


async def _handle_regime_event(
    event: RegimeEvent, send_alert: Callable,
) -> None:
    await send_alert(
        discord.Embed.from_dict(_regime_shift_embed(event.transitions)),
        "regime_shift",
    )
    if event.cascade_triggered:
        await send_alert(discord.Embed(
            title="Regime Cascade \u2014 Early Scan Triggered",
            description=(
                f"{len(event.transitions)} regime shifts detected"
                " simultaneously.\n"
                f"Tickers: {', '.join(t['ticker'] for t in event.transitions)}\n"
                "Running full scan now..."
            ),
            color=0xFF4500,
        ), "regime_shift")


async def _handle_retrain_result(
    event: RetrainResult, send_alert: Callable,
) -> None:
    health = {
        "total_count": event.manifest_size,
        "fresh_count": event.manifest_size,
        "stale_count": 0,
    }
    await send_alert(
        discord.Embed.from_dict(_retrain_embed(health, event.duration_seconds)),
        "retrain",
    )


async def _handle_risk_action_event(
    event: RiskActionEvent, send_alert: Callable,
) -> None:
    await send_alert(
        discord.Embed.from_dict(_risk_action_embed(event.actions)),
        "risk_action",
    )


async def _handle_stream_degraded(
    event: StreamDegradedEvent, send_alert: Callable,
) -> None:
    await send_alert(
        discord.Embed.from_dict(
            _stream_degraded_embed(event.disconnected_minutes),
        ),
        "stream_degraded",
    )


async def _handle_stream_recovered(
    event: StreamRecoveredEvent, send_alert: Callable,
) -> None:
    await send_alert(
        discord.Embed.from_dict(
            _stream_recovered_embed(event.downtime_minutes),
        ),
        "stream_recovered",
    )


_EVENT_HANDLERS: dict[type, Callable] = {
    AlertEvent: _handle_alert_event,
    ExitResult: _handle_exit_result,
    HousekeepingResult: _handle_housekeeping_result,
    ScanResult: _handle_scan_result,
    WatchlistEvent: _handle_watchlist_event,
    RegimeEvent: _handle_regime_event,
    RetrainResult: _handle_retrain_result,
    RiskActionEvent: _handle_risk_action_event,
    StreamDegradedEvent: _handle_stream_degraded,
    StreamRecoveredEvent: _handle_stream_recovered,
}


def make_discord_event_handler(
    bot: commands.Bot,
) -> Callable[..., Awaitable[None]]:
    """Return an async handler that sends TradingScheduler events to Discord."""
    alert_channel_id = os.environ.get("DISCORD_CHANNEL_ID", "")

    async def _send_alert(embed: discord.Embed, msg_type: str = "other") -> None:
        if not alert_channel_id:
            return
        channel = bot.get_channel(int(alert_channel_id))
        if channel:
            await channel.send(embed=embed)
        log_discord_message(
            msg_type, embed.title, (embed.description or "")[:500] or None,
        )

    async def _send_error_alert(task_name: str, error: Exception) -> None:
        tb = "".join(
            traceback.format_exception(type(error), error, error.__traceback__),
        )
        details = f"```\n{tb[-3500:]}\n```"
        embed = discord.Embed.from_dict(
            _error_embed(f"{task_name} Failed", details),
        )
        await _send_alert(embed, "error")

    async def _handle_trading_event(event) -> None:
        """Dispatch a typed TradingScheduler event to its handler."""
        handler = _EVENT_HANDLERS.get(type(event))
        if handler is None:
            return
        try:
            await handler(event, _send_alert)
        except Exception as e:
            logger.exception(
                "Event handler error for %s", type(event).__name__,
            )
            await _send_error_alert(
                f"Event Handler ({type(event).__name__})", e,
            )

    return _handle_trading_event
