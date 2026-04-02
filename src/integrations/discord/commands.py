"""Discord bot commands — slash commands and message handler."""

import asyncio
import logging
import os

import discord
from discord.ext import commands

from db.ops.conversations import get_conversation_history, save_conversation_history
from db.ops.users import ensure_user
from services.async_tasks import run_retrain, run_simulation

from .agent import process_message

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 2000


def register_commands(bot: commands.Bot) -> None:
    """Register all commands and event handlers on the bot."""

    @bot.command(name="simulate")
    async def _simulate_cmd(
        ctx: commands.Context, scenario: str = "", *, rest: str = "",
    ) -> None:
        equity_override: float = 0.0
        if rest.strip():
            try:
                equity_override = float(rest.strip())
            except ValueError:
                await ctx.send(
                    f"\u26a0\ufe0f Invalid equity value: `{rest.strip()}`"
                    " \u2014 must be a number."
                )
                return

        scheduler = ctx.bot.scheduler
        if scheduler is None:
            await ctx.send(
                "\u26a0\ufe0f Scheduler not initialised"
                " \u2014 cannot run simulation."
            )
            return

        inject_fn = (
            scheduler._stream.inject_trade if scheduler._stream else None
        )

        async def send(text: str) -> None:
            await ctx.send(text)

        async def send_embed(embed_dict: dict) -> None:
            await ctx.send(embed=discord.Embed.from_dict(embed_dict))

        await run_simulation(
            scenario, equity_override, inject_fn, send, send_embed,
        )

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
                "\U0001f4a1 **Tip**: Set `ANTHROPIC_API_KEY` in `.env`"
                " to enable the agentic bot!"
            )
            return

        ensure_user(message.author.id, str(message.author))

        history = await asyncio.to_thread(
            get_conversation_history, message.author.id,
        )

        async with message.channel.typing():
            try:
                content = message.content
                if bot.user.mentioned_in(message):
                    content = content.replace(
                        f"<@{bot.user.id}>", "",
                    ).strip()

                response_text, updated_history, async_tasks = (
                    await asyncio.to_thread(
                        process_message,
                        content,
                        message.author.id,
                        str(message.author),
                        history,
                    )
                )

                await asyncio.to_thread(
                    save_conversation_history,
                    message.author.id,
                    updated_history,
                )

                if len(response_text) <= _MAX_MESSAGE_LENGTH:
                    await message.channel.send(response_text)
                else:
                    for i in range(0, len(response_text), _MAX_MESSAGE_LENGTH):
                        chunk = response_text[i:i + _MAX_MESSAGE_LENGTH]
                        await message.channel.send(chunk)

                for task in async_tasks:
                    if task.get("_async_task") == "retrain":
                        async def _send(
                            text: str, ch=message.channel,
                        ) -> None:
                            await ch.send(text)
                        asyncio.create_task(run_retrain(
                            _send, task.get("tickers"),
                        ))

            except Exception as e:
                logger.exception("Claude message processing failed")
                await message.channel.send(
                    f"\u26a0\ufe0f Error processing your request: {e!s}"
                )
