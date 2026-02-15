"""Interactive Discord bot for market-rally — slash commands + rich embeds.

Reads shared disk state (positions.json, manifest.json, equity_history.csv)
for system queries. Writes to SQLite for per-user trade tracking.

Includes a built-in scheduler for scan/retrain so no external cron is needed
(useful for Railway/cloud deployment).
"""

import asyncio
import logging
import os
import zoneinfo
from datetime import datetime, time

import discord
from discord import app_commands
from discord.ext import commands, tasks

from .claude_agent import process_message
from .discord_db import (
    clear_conversation_history,
    close_trade,
    ensure_user,
    get_capital,
    get_conversation_history,
    get_open_trades,
    get_pnl_summary,
    get_trade_history,
    open_trade,
    save_conversation_history,
    set_capital,
)
from .persistence import load_manifest
from .portfolio import load_equity_history, load_trade_journal
from .positions import load_positions

logger = logging.getLogger(__name__)

# Colors
GREEN = 0x00FF00
RED = 0xFF0000
BLUE = 0x0099FF
GOLD = 0xFFD700
GRAY = 0x95A5A6


def _ensure_caller(interaction: discord.Interaction) -> None:
    """Auto-register the calling Discord user."""
    ensure_user(interaction.user.id, str(interaction.user))


def _ensure_caller_from_message(message: discord.Message) -> None:
    """Auto-register user from message (not interaction)."""
    ensure_user(message.author.id, str(message.author))


def _get_system_recommendation(ticker: str) -> dict | None:
    """Look up the system's current recommendation for a ticker.

    Checks open positions first (already entered by the system), then
    falls back to the latest scan results cached in positions.json.
    Returns dict with size, stop_price, target_price, entry_price or None.
    """
    from .positions import load_positions

    state = load_positions()

    # Check open positions (system already entered this trade)
    for p in state.get("positions", []):
        if p.get("ticker", "").upper() == ticker.upper():
            return {
                "entry_price": p.get("entry_price"),
                "size": p.get("size", 0),
                "stop_price": p.get("stop_price"),
                "target_price": p.get("target_price"),
            }

    return None


class RallyBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info(f"Synced {len(self.tree.get_commands())} slash commands")

    async def on_ready(self) -> None:
        logger.info(f"Bot connected as {self.user} (ID: {self.user.id})")
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="rally signals",
            )
        )


def make_bot(token: str) -> RallyBot:
    """Create and configure the agentic Discord bot."""
    bot = RallyBot()

    # ------------------------------------------------------------------
    # Async task handlers for long-running operations
    # ------------------------------------------------------------------
    async def _run_retrain_task(
        channel: discord.abc.Messageable,
        tickers: list[str] | None = None
    ) -> None:
        """Run model retraining with progress updates."""
        import time
        from .retrain import retrain_all
        from .persistence import load_manifest

        try:
            # Send initial message
            await channel.send("🔄 **Starting model retraining...**\nThis will take 10-30+ minutes depending on the number of tickers.")

            # Run retrain in thread and time it
            start_time = time.time()

            # Send progress update every 5 minutes
            async def send_updates():
                elapsed = 0
                while True:
                    await asyncio.sleep(300)  # 5 minutes
                    elapsed += 5
                    await channel.send(f"⏳ Still training... {elapsed} minutes elapsed")

            # Start update task
            update_task = asyncio.create_task(send_updates())

            try:
                # Run the actual retraining
                await asyncio.to_thread(retrain_all, tickers)
            finally:
                # Cancel update task
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass

            # Calculate elapsed time
            elapsed = time.time() - start_time
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)

            # Get model count
            manifest = load_manifest()
            model_count = len(manifest) if manifest else 0

            # Send completion message
            await channel.send(
                f"✅ **Retraining complete!**\n"
                f"• Trained: {model_count} models\n"
                f"• Time: {minutes}m {seconds}s\n"
                f"• You can now run scans to find signals!"
            )

        except Exception as e:
            logger.exception("Retrain task failed")
            await channel.send(f"❌ **Retraining failed:** {str(e)}")

    # ------------------------------------------------------------------
    # Message handler for natural language interaction via Claude
    # ------------------------------------------------------------------
    @bot.event
    async def on_message(message: discord.Message) -> None:
        """Handle natural language messages via Claude."""
        # Ignore bot's own messages
        if message.author == bot.user:
            return

        # Only respond to DMs or mentions
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = bot.user in message.mentions

        if not (is_dm or is_mentioned):
            return

        # Check if Claude is enabled
        if not os.environ.get("ANTHROPIC_API_KEY"):
            await message.channel.send(
                "💡 **Tip**: Set `ANTHROPIC_API_KEY` in `.env` to enable the agentic bot!"
            )
            return

        # Ensure user is registered
        _ensure_caller_from_message(message)

        # Get conversation history
        history = await asyncio.to_thread(
            get_conversation_history,
            message.author.id
        )

        # Show typing indicator
        async with message.channel.typing():
            try:
                # Remove mention from message if present
                content = message.content
                if bot.user.mentioned_in(message):
                    content = content.replace(f"<@{bot.user.id}>", "").strip()

                # Process message with Claude (runs in thread to avoid blocking)
                response_text, updated_history, async_tasks = await asyncio.to_thread(
                    process_message,
                    content,
                    message.author.id,
                    str(message.author),
                    history
                )

                # Save updated history
                await asyncio.to_thread(
                    save_conversation_history,
                    message.author.id,
                    updated_history
                )

                # Send response (split if too long for Discord's 2000 char limit)
                if len(response_text) <= 2000:
                    await message.channel.send(response_text)
                else:
                    # Split into chunks
                    chunks = [response_text[i:i+2000] for i in range(0, len(response_text), 2000)]
                    for chunk in chunks:
                        await message.channel.send(chunk)

                # Handle async tasks (long-running operations like retrain)
                for task in async_tasks:
                    if task.get("_async_task") == "retrain":
                        # Spawn retrain task in background
                        asyncio.create_task(_run_retrain_task(
                            message.channel,
                            task.get("tickers")
                        ))

            except Exception as e:
                logger.exception("Claude message processing failed")
                await message.channel.send(
                    f"⚠️ Error processing your request: {str(e)}"
                )

    # ------------------------------------------------------------------
    # Built-in scheduler (replaces cron for cloud deployment)
    # Set ENABLE_SCHEDULER=1 in .env to activate
    # ------------------------------------------------------------------
    if os.environ.get("ENABLE_SCHEDULER", "").strip() in ("1", "true", "yes"):
        alert_channel_id = os.environ.get("DISCORD_CHANNEL_ID", "")

        async def _send_alert(embed: discord.Embed) -> None:
            """Send an embed to the alerts channel."""
            if not alert_channel_id:
                return
            channel = bot.get_channel(int(alert_channel_id))
            if channel:
                await channel.send(embed=embed)

        async def _run_scan() -> None:
            """Run the scan pipeline in a thread, then post results."""
            from .notify import _exit_embed, _signal_embed
            from .portfolio import record_closed_trades, update_daily_snapshot
            from .positions import load_positions
            from .scanner import scan_all

            logger.info("Scheduler: starting daily scan")
            results = await asyncio.to_thread(
                scan_all, None, True, "baseline"
            )
            if not results:
                return

            signals = [r for r in results if r.get("signal")]
            if signals:
                await _send_alert(discord.Embed.from_dict(_signal_embed(signals)))

            positions = load_positions()
            closed = positions.get("closed_today", [])
            if closed:
                await _send_alert(discord.Embed.from_dict(_exit_embed(closed)))
                record_closed_trades(closed)

            update_daily_snapshot(positions, results)
            logger.info(
                f"Scheduler: scan complete — {len(signals)} signals, "
                f"{len(closed)} exits"
            )

        async def _run_retrain() -> None:
            """Run retrain in a thread, then post results."""
            import time as _time

            from .notify import _retrain_embed
            from .retrain import retrain_all

            logger.info("Scheduler: starting weekly retrain")
            t0 = _time.time()
            await asyncio.to_thread(retrain_all)
            elapsed = _time.time() - t0

            from .persistence import load_manifest

            manifest = load_manifest()
            health = {
                "total_count": len(manifest),
                "fresh_count": len(manifest),
                "stale_count": 0,
            }
            await _send_alert(
                discord.Embed.from_dict(_retrain_embed(health, elapsed))
            )
            logger.info(f"Scheduler: retrain complete ({elapsed:.0f}s)")

        # -- Price alert configuration --
        _alert_interval = int(os.environ.get("PRICE_ALERT_INTERVAL_MINUTES", "15"))
        _alert_proximity_pct = float(os.environ.get("ALERT_PROXIMITY_PCT", "1.5"))
        _sent_alerts: set[tuple[str, str, str]] = set()  # (ticker, alert_type, date)

        _ET = zoneinfo.ZoneInfo("America/New_York")

        def _is_market_open() -> bool:
            """True if Mon-Fri 9:30 AM - 4:00 PM ET."""
            now = datetime.now(_ET)
            if now.weekday() >= 5:
                return False
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            return market_open <= now <= market_close

        async def _check_price_alerts() -> None:
            """Fetch live prices for open positions and alert on breaches."""
            from .data import fetch_quotes
            from .notify import _approaching_alert_embed, _price_alert_embed
            from .positions import load_positions

            if not _is_market_open():
                return

            state = load_positions()
            positions = state.get("positions", [])
            if not positions:
                return

            today = datetime.now(_ET).strftime("%Y-%m-%d")

            # Clear stale alerts from previous days
            stale = {k for k in _sent_alerts if k[2] != today}
            _sent_alerts.difference_update(stale)

            tickers = [p["ticker"] for p in positions]
            quotes = await asyncio.to_thread(fetch_quotes, tickers)

            breach_alerts: list[dict] = []
            approach_alerts: list[dict] = []

            for pos in positions:
                ticker = pos["ticker"]
                quote = quotes.get(ticker)
                if not quote or "error" in quote:
                    continue

                price = quote["price"]
                entry = pos["entry_price"]
                stop = pos.get("stop_price", 0)
                target = pos.get("target_price", 0)
                trailing = pos.get("trailing_stop", 0)
                effective_stop = max(stop, trailing)
                pnl_pct = round((price / entry - 1) * 100, 2) if entry else 0

                # Check stop breach
                if effective_stop > 0 and price <= effective_stop:
                    key = (ticker, "stop_breached", today)
                    if key not in _sent_alerts:
                        _sent_alerts.add(key)
                        level_name = "Trailing Stop" if trailing > stop else "Stop"
                        breach_alerts.append({
                            "ticker": ticker,
                            "alert_type": "stop_breached",
                            "current_price": price,
                            "level_price": effective_stop,
                            "level_name": level_name,
                            "entry_price": entry,
                            "pnl_pct": pnl_pct,
                        })

                # Check target breach
                elif target > 0 and price >= target:
                    key = (ticker, "target_breached", today)
                    if key not in _sent_alerts:
                        _sent_alerts.add(key)
                        breach_alerts.append({
                            "ticker": ticker,
                            "alert_type": "target_breached",
                            "current_price": price,
                            "level_price": target,
                            "level_name": "Target",
                            "entry_price": entry,
                            "pnl_pct": pnl_pct,
                        })

                # Check approaching stop
                elif _alert_proximity_pct > 0 and effective_stop > 0:
                    distance = (price / effective_stop - 1) * 100
                    if 0 < distance <= _alert_proximity_pct:
                        key = (ticker, "near_stop", today)
                        if key not in _sent_alerts:
                            _sent_alerts.add(key)
                            level_name = "Trailing Stop" if trailing > stop else "Stop"
                            approach_alerts.append({
                                "ticker": ticker,
                                "alert_type": "near_stop",
                                "current_price": price,
                                "level_price": effective_stop,
                                "level_name": level_name,
                                "distance_pct": round(distance, 1),
                                "entry_price": entry,
                                "pnl_pct": pnl_pct,
                            })

                # Check approaching target
                elif _alert_proximity_pct > 0 and target > 0:
                    distance = (target / price - 1) * 100
                    if 0 < distance <= _alert_proximity_pct:
                        key = (ticker, "near_target", today)
                        if key not in _sent_alerts:
                            _sent_alerts.add(key)
                            approach_alerts.append({
                                "ticker": ticker,
                                "alert_type": "near_target",
                                "current_price": price,
                                "level_price": target,
                                "level_name": "Target",
                                "distance_pct": round(distance, 1),
                                "entry_price": entry,
                                "pnl_pct": pnl_pct,
                            })

            if breach_alerts:
                await _send_alert(
                    discord.Embed.from_dict(_price_alert_embed(breach_alerts))
                )
            if approach_alerts:
                await _send_alert(
                    discord.Embed.from_dict(_approaching_alert_embed(approach_alerts))
                )

            n = len(breach_alerts) + len(approach_alerts)
            if n:
                logger.info(f"Price alerts: {len(breach_alerts)} breaches, "
                            f"{len(approach_alerts)} warnings")

        # Price alerts: every N minutes during market hours
        @tasks.loop(minutes=_alert_interval)
        async def scheduled_price_alerts() -> None:
            try:
                await _check_price_alerts()
            except Exception:
                logger.exception("Price alert check failed")

        # Daily scan: 4:30 PM ET (21:30 UTC)
        @tasks.loop(time=time(hour=21, minute=30))
        async def scheduled_scan() -> None:
            weekday = datetime.utcnow().weekday()
            if weekday >= 5:  # skip weekends
                return
            try:
                await _run_scan()
            except Exception:
                logger.exception("Scheduled scan failed")

        # Weekly retrain: Sunday 6 PM ET (23:00 UTC)
        @tasks.loop(time=time(hour=23, minute=0))
        async def scheduled_retrain() -> None:
            if datetime.utcnow().weekday() != 6:  # Sunday only
                return
            try:
                await _run_retrain()
            except Exception:
                logger.exception("Scheduled retrain failed")

        @bot.event
        async def on_ready() -> None:
            logger.info(f"Bot connected as {bot.user} (ID: {bot.user.id})")
            await bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="rally signals",
                )
            )
            if not scheduled_scan.is_running():
                scheduled_scan.start()
                logger.info("Scheduler: daily scan armed (21:30 UTC / 4:30 PM ET)")
            if not scheduled_retrain.is_running():
                scheduled_retrain.start()
                logger.info("Scheduler: weekly retrain armed (Sun 23:00 UTC)")
            if not scheduled_price_alerts.is_running():
                scheduled_price_alerts.start()
                logger.info(
                    f"Scheduler: price alerts armed "
                    f"(every {_alert_interval}m during market hours)"
                )

    return bot
