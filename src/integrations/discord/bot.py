"""Interactive Discord bot for market-rally — slash commands + rich embeds.

Thin notification sink: all trading logic lives in TradingScheduler.
Receives typed events, converts them to Discord embeds, and sends them.

Also exposes an HTTP API on $PORT (default 8080) for local tools to query
positions from the Railway deployment.
"""

import asyncio
import json as _json
import logging
import os
import queue
import time as _time
import traceback

import discord
from aiohttp import web
from discord.ext import commands

from core.persistence import load_manifest
from db.conversations import get_conversation_history, save_conversation_history
from db.events import log_discord_message
from db.users import ensure_user
from pipeline.retrain import retrain_all
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
from trading.positions import get_merged_positions
from trading.scheduler import TradingScheduler

from .agent import process_message
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

# Colors
GREEN = 0x00FF00
RED = 0xFF0000
BLUE = 0x0099FF
GOLD = 0xFFD700
GRAY = 0x95A5A6


# ---------------------------------------------------------------------------
# HTTP API server (runs alongside the Discord bot)
# ---------------------------------------------------------------------------

async def _handle_positions(request: web.Request) -> web.Response:
    """Return merged Alpaca + DB positions as JSON."""
    api_key = os.environ.get("RALLY_API_KEY")
    if api_key:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {api_key}":
            return web.json_response({"error": "unauthorized"}, status=401)
    state = await get_merged_positions()
    return web.Response(
        text=_json.dumps(state), content_type="application/json",
    )


async def _handle_health(request: web.Request) -> web.Response:
    """Simple health check."""
    return web.json_response({"status": "ok"})


async def start_api_server() -> None:
    """Start the aiohttp API server alongside the Discord bot."""
    app = web.Application()
    app.router.add_get("/api/positions", _handle_positions)
    app.router.add_get("/health", _handle_health)
    port = int(os.environ.get("PORT", 8080))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("API server listening on port %d", port)


# ---------------------------------------------------------------------------

def _ensure_caller(interaction: discord.Interaction) -> None:
    """Auto-register the calling Discord user."""
    ensure_user(interaction.user.id, str(interaction.user))


def _ensure_caller_from_message(message: discord.Message) -> None:
    """Auto-register user from message (not interaction)."""
    ensure_user(message.author.id, str(message.author))


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class RallyBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info("Synced %d slash commands", len(self.tree.get_commands()))

    async def on_ready(self) -> None:
        logger.info("Bot connected as %s (ID: %s)", self.user, self.user.id)
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="rally signals",
            )
        )
        asyncio.create_task(start_api_server())


def make_bot(token: str) -> RallyBot:
    """Create and configure the agentic Discord bot."""
    bot = RallyBot()

    # ------------------------------------------------------------------
    # Manual retrain task — used by Claude agent (progress updates)
    # ------------------------------------------------------------------
    async def _run_retrain_task(
        channel: discord.abc.Messageable,
        tickers: list[str] | None = None
    ) -> None:
        """Run model retraining with live progress updates."""
        try:
            await channel.send(
                "🔄 **Starting model retraining...**\n"
                "This will take 10-30+ minutes depending on the number of tickers."
            )

            start_time = _time.time()
            progress_queue: queue.Queue[tuple[int, int, int, int]] = queue.Queue()

            def on_progress(done: int, total: int, success: int, failed: int) -> None:
                try:
                    progress_queue.get_nowait()
                except queue.Empty:
                    pass
                progress_queue.put((done, total, success, failed))

            async def send_updates() -> None:
                while True:
                    await asyncio.sleep(300)
                    elapsed_min = int((_time.time() - start_time) / 60)
                    try:
                        done, total, success, failed = progress_queue.get_nowait()
                        pct = int(done / total * 100) if total else 0
                        await channel.send(
                            f"⏳ **Retraining... {elapsed_min}m elapsed**\n"
                            f"• Progress: {done}/{total} ({pct}%)\n"
                            f"• Success: {success} | Failed: {failed}"
                        )
                    except queue.Empty:
                        await channel.send(
                            f"⏳ **Retraining... {elapsed_min}m elapsed** (fetching data)"
                        )

            update_task = asyncio.create_task(send_updates())
            try:
                await asyncio.to_thread(retrain_all, tickers, False, on_progress)
            finally:
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass

            elapsed = _time.time() - start_time
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)
            try:
                done, total, success, failed = progress_queue.get_nowait()
            except queue.Empty:
                manifest = load_manifest()
                success = len(manifest) if manifest else 0
                failed = 0
                total = success

            await channel.send(
                f"✅ **Retraining complete!**\n"
                f"• Trained: {success}/{total} models\n"
                f"• Failed: {failed}\n"
                f"• Time: {minutes}m {seconds}s\n"
                f"• You can now run scans to find signals!"
            )
        except Exception as e:
            logger.exception("Retrain task failed")
            await channel.send(f"❌ **Retraining failed:** {str(e)}")

    # ------------------------------------------------------------------
    # Scheduler (only when ENABLE_SCHEDULER=1)
    # ------------------------------------------------------------------
    if os.environ.get("ENABLE_SCHEDULER", "").strip() in ("1", "true", "yes"):
        alert_channel_id = os.environ.get("DISCORD_CHANNEL_ID", "")

        async def _send_alert(embed: discord.Embed, msg_type: str = "other") -> None:
            """Send an embed to the alerts channel and log to DB."""
            if not alert_channel_id:
                return
            channel = bot.get_channel(int(alert_channel_id))
            if channel:
                await channel.send(embed=embed)
            log_discord_message(msg_type, embed.title, (embed.description or "")[:500] or None)

        async def _send_error_alert(task_name: str, error: Exception) -> None:
            """Send an error embed to the alerts channel."""
            tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            details = f"```\n{tb[-3500:]}\n```"
            embed = discord.Embed.from_dict(_error_embed(f"{task_name} Failed", details))
            await _send_alert(embed, "error")

        async def _handle_trading_event(event) -> None:
            """Convert a typed TradingScheduler event to Discord embeds."""
            try:
                match event:
                    case AlertEvent(alert_type="stop_breached" | "target_breached"):
                        embed = discord.Embed.from_dict(_price_alert_embed([event.model_dump()]))
                        await _send_alert(embed, "price_alert")

                    case AlertEvent(alert_type="near_stop" | "near_target"):
                        embed = discord.Embed.from_dict(_approaching_alert_embed([event.model_dump()]))
                        await _send_alert(embed, "price_alert")

                    case ExitResult():
                        embed = discord.Embed.from_dict(_exit_embed([event.model_dump()]))
                        await _send_alert(embed, "exit")

                    case HousekeepingResult() if event.fills_confirmed:
                        fills = [f.model_dump() for f in event.fills_confirmed]
                        embed = discord.Embed.from_dict(_fill_confirmation_embed(fills))
                        await _send_alert(embed, "order")

                    case ScanResult() if event.error:
                        embed = discord.Embed(
                            title="⚠️ Daily Scan Skipped — No Trained Models",
                            description=(
                                "No trained models were found on the volume.\n"
                                "Run retrain before the next scan: ask me to **retrain** or trigger it manually."
                            ),
                            color=discord.Color.orange(),
                        )
                        await _send_alert(embed, "scan_error")

                    case ScanResult():
                        if event.signals:
                            sig_embed = discord.Embed.from_dict(_signal_embed(event.signals))
                            if event.scan_type not in ("daily", "morning", "cascade", "post_retrain"):
                                sig_embed.title = f"Mid-day Signal ({len(event.signals)})"
                                sig_embed.set_footer(text="From watchlist mid-day scan")
                            await _send_alert(sig_embed, "signal")

                        pos = event.positions_summary.get("positions", [])
                        await _send_alert(discord.Embed.from_dict(_positions_embed(pos)), "positions")

                        if event.exits:
                            await _send_alert(discord.Embed.from_dict(_exit_embed(event.exits)), "exit")

                        entry_ok = [o for o in event.orders
                                    if o.get("side") == "buy" and o.get("success")]
                        entry_fail = [o for o in event.orders
                                      if o.get("side") == "buy" and not o.get("success") and not o.get("skipped")]
                        exit_ok = [o for o in event.orders
                                   if o.get("side") == "sell" and o.get("success")]

                        if entry_ok and event.equity:
                            await _send_alert(
                                discord.Embed.from_dict(_order_embed(entry_ok, event.equity)), "order"
                            )
                        if entry_fail:
                            await _send_alert(
                                discord.Embed.from_dict(_order_failure_embed(entry_fail)), "order_failure"
                            )
                        if exit_ok and event.equity:
                            await _send_alert(
                                discord.Embed.from_dict(_order_embed(exit_ok, event.equity)), "order"
                            )

                    case WatchlistEvent() if event.signals:
                        sig_embed = discord.Embed.from_dict(_signal_embed(event.signals))
                        sig_embed.title = f"Mid-day Signal ({len(event.signals)})"
                        sig_embed.set_footer(text="From watchlist mid-day scan")
                        await _send_alert(sig_embed, "signal")

                    case RegimeEvent():
                        await _send_alert(
                            discord.Embed.from_dict(_regime_shift_embed(event.transitions)),
                            "regime_shift",
                        )
                        if event.cascade_triggered:
                            await _send_alert(discord.Embed(
                                title="Regime Cascade — Early Scan Triggered",
                                description=(
                                    f"{len(event.transitions)} regime shifts detected simultaneously.\n"
                                    f"Tickers: {', '.join(t['ticker'] for t in event.transitions)}\n"
                                    "Running full scan now..."
                                ),
                                color=0xFF4500,
                            ), "regime_shift")

                    case RetrainResult():
                        health = {
                            "total_count": event.manifest_size,
                            "fresh_count": event.manifest_size,
                            "stale_count": 0,
                        }
                        await _send_alert(
                            discord.Embed.from_dict(_retrain_embed(health, event.duration_seconds)),
                            "retrain",
                        )

                    case RiskActionEvent():
                        await _send_alert(
                            discord.Embed.from_dict(_risk_action_embed(event.actions)),
                            "risk_action",
                        )

                    case StreamDegradedEvent():
                        await _send_alert(
                            discord.Embed.from_dict(
                                _stream_degraded_embed(event.disconnected_minutes)
                            ),
                            "stream_degraded",
                        )

                    case StreamRecoveredEvent():
                        await _send_alert(
                            discord.Embed.from_dict(
                                _stream_recovered_embed(event.downtime_minutes)
                            ),
                            "stream_recovered",
                        )

            except Exception as e:
                logger.exception("Event handler error for %s", type(event).__name__)
                await _send_error_alert(f"Event Handler ({type(event).__name__})", e)

        _scheduler = TradingScheduler(on_event=_handle_trading_event)

        @bot.event
        async def on_ready() -> None:
            logger.info("Bot connected as %s (ID: %s)", bot.user, bot.user.id)
            await bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="rally signals",
                )
            )
            asyncio.create_task(start_api_server())
            try:
                await _scheduler.start()
                logger.info("TradingScheduler started")
            except Exception as e:
                logger.exception("Failed to start TradingScheduler")
                await _send_error_alert("TradingScheduler Startup", e)

        async def _bot_close() -> None:
            await _scheduler.stop()

        bot._scheduler_stop = _bot_close

    # ------------------------------------------------------------------
    # Message handler for natural language interaction via Claude
    # ------------------------------------------------------------------
    @bot.event
    async def on_message(message: discord.Message) -> None:
        """Handle natural language messages via Claude."""
        if message.author == bot.user:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = bot.user in message.mentions

        if not (is_dm or is_mentioned):
            return

        if not os.environ.get("ANTHROPIC_API_KEY"):
            await message.channel.send(
                "💡 **Tip**: Set `ANTHROPIC_API_KEY` in `.env` to enable the agentic bot!"
            )
            return

        _ensure_caller_from_message(message)

        history = await asyncio.to_thread(
            get_conversation_history,
            message.author.id
        )

        async with message.channel.typing():
            try:
                content = message.content
                if bot.user.mentioned_in(message):
                    content = content.replace(f"<@{bot.user.id}>", "").strip()

                response_text, updated_history, async_tasks = await asyncio.to_thread(
                    process_message,
                    content,
                    message.author.id,
                    str(message.author),
                    history
                )

                await asyncio.to_thread(
                    save_conversation_history,
                    message.author.id,
                    updated_history
                )

                if len(response_text) <= 2000:
                    await message.channel.send(response_text)
                else:
                    chunks = [response_text[i:i+2000] for i in range(0, len(response_text), 2000)]
                    for chunk in chunks:
                        await message.channel.send(chunk)

                for task in async_tasks:
                    if task.get("_async_task") == "retrain":
                        asyncio.create_task(_run_retrain_task(
                            message.channel,
                            task.get("tickers")
                        ))

            except Exception as e:
                logger.exception("Claude message processing failed")
                await message.channel.send(
                    f"⚠️ Error processing your request: {str(e)}"
                )

    return bot
