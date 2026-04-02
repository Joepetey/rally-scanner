"""Interactive Discord bot for market-rally — slash commands + rich embeds.

Thin notification sink: all trading logic lives in TradingScheduler.
Receives typed events, converts them to Discord embeds, and sends them.
"""

import logging

import discord
from discord.ext import commands

from .commands import register_commands
from .event_handlers import make_discord_event_handler

logger = logging.getLogger(__name__)

__all__ = ["RallyBot", "make_bot", "make_discord_event_handler"]


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
    register_commands(bot)
    return bot
