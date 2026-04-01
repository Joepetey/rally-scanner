"""Interactive Discord bot for market-rally — slash commands + rich embeds.

Thin notification sink: all trading logic lives in TradingScheduler.
Receives typed events, converts them to Discord embeds, and sends them.
"""

import asyncio
import logging
import os
import queue
import time as _time

import discord
from discord.ext import commands

from core.persistence import load_manifest
from db.conversations import get_conversation_history, save_conversation_history
from db.users import ensure_user
from integrations.discord.event_handlers import (  # noqa: F401 — re-export
    make_discord_event_handler,
)
from pipeline.retrain import retrain_all

from .agent import process_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

def _ensure_caller(interaction: discord.Interaction) -> None:
    """Auto-register the calling Discord user."""
    ensure_user(interaction.user.id, str(interaction.user))


def _ensure_caller_from_message(message: discord.Message) -> None:
    """Auto-register user from message (not interaction)."""
    ensure_user(message.author.id, str(message.author))


# ---------------------------------------------------------------------------
# Module-level task helpers (MIC-129, MIC-130)
# ---------------------------------------------------------------------------

async def _run_retrain_task(
    channel: discord.abc.Messageable,
    tickers: list[str] | None = None,
) -> None:
    """Run model retraining with live progress updates."""
    try:
        await channel.send(
            "\U0001f504 **Starting model retraining...**\n"
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
                        f"\u23f3 **Retraining... {elapsed_min}m elapsed**\n"
                        f"\u2022 Progress: {done}/{total} ({pct}%)\n"
                        f"\u2022 Success: {success} | Failed: {failed}"
                    )
                except queue.Empty:
                    await channel.send(
                        f"\u23f3 **Retraining... {elapsed_min}m elapsed** (fetching data)"
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
            f"\u2705 **Retraining complete!**\n"
            f"\u2022 Trained: {success}/{total} models\n"
            f"\u2022 Failed: {failed}\n"
            f"\u2022 Time: {minutes}m {seconds}s\n"
            f"\u2022 You can now run scans to find signals!"
        )
    except Exception as e:
        logger.exception("Retrain task failed")
        await channel.send(f"\u274c **Retraining failed:** {str(e)}")


async def _simulate_command(
    ctx: commands.Context, scenario: str = "", *, rest: str = "",
) -> None:
    """Run a BTC-USD paper trading simulation: !simulate <target|stop|trail|time> [equity]."""
    from integrations.alpaca.broker import simulation_keys
    from simulation.runner import SimulationRunner

    equity_override: float = 0.0
    if rest.strip():
        try:
            equity_override = float(rest.strip())
        except ValueError:
            await ctx.send(
                f"\u26a0\ufe0f Invalid equity value: `{rest.strip()}` \u2014 must be a number."
            )
            return

    if not scenario:
        await ctx.send(
            "Usage: `!simulate <scenario> [equity]`\n"
            "Scenarios: `target` `stop` `trail` `time`"
        )
        return

    scheduler = ctx.bot.scheduler
    if scheduler is None:
        await ctx.send("\u26a0\ufe0f Scheduler not initialised \u2014 cannot run simulation.")
        return

    try:
        sim_ctx = simulation_keys()
        sim_ctx.__enter__()
    except RuntimeError as e:
        await ctx.send(f"\u26a0\ufe0f {e}")
        return

    try:
        if equity_override > 0:
            equity = equity_override
        else:
            try:
                from integrations.alpaca.executor import get_account_equity
                equity = await get_account_equity()
            except Exception as e:
                await ctx.send(f"\u26a0\ufe0f Could not fetch account equity: {e}")
                return

        inject_fn = scheduler._stream.inject_trade if scheduler._stream else None

        async def send_embed(embed_dict: dict) -> None:
            await ctx.send(embed=discord.Embed.from_dict(embed_dict))

        runner = SimulationRunner(inject_fn=inject_fn)

        await ctx.send(
            f"\u25b6\ufe0f **Simulation starting** \u2014 scenario: `{scenario}`"
            f" | equity: `${equity:,.0f}`\n"
            f"BTC paper order will be placed on simulation account. Results follow..."
        )
        result = await runner.run(scenario, equity, send_embed)

        if result.success:
            pnl = result.realized_pnl_pct or 0.0
            sign = "+" if pnl >= 0 else ""
            await ctx.send(
                f"\u2705 **Simulation complete** \u2014 `{scenario}`\n"
                f"Entry: `${result.entry_price:,.2f}` \u2192 Exit: `${result.exit_price:,.2f}`\n"
                f"Reason: `{result.exit_reason}` | PnL: `{sign}{pnl:.2f}%`"
            )
        else:
            await ctx.send(f"\u274c **Simulation failed** \u2014 `{scenario}`\n{result.error}")
    finally:
        sim_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class RallyBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        # Set by app.py after TradingScheduler.start() so !simulate can reach the stream
        self.scheduler = None

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


def make_bot(token: str) -> RallyBot:
    """Create and configure the agentic Discord bot."""
    bot = RallyBot()

    # Register simulate command
    bot.command(name="simulate")(_simulate_command)

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
                "\U0001f4a1 **Tip**: Set `ANTHROPIC_API_KEY` in `.env` to enable the agentic bot!"
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
                    f"\u26a0\ufe0f Error processing your request: {str(e)}"
                )

    return bot
