"""TradingScheduler-first entrypoint.

TradingScheduler and the HTTP API server always start.
Discord connects as an optional side-task if DISCORD_BOT_TOKEN is set.
"""

import asyncio
import logging
import os
import signal

from db.pool import init_pool
from db.schema import init_schema
from api import start_api_server
from integrations.discord.bot import make_bot, make_discord_event_handler
from log import setup_logging
from monitoring import init_sentry
from trading.scheduler import TradingScheduler

logger = logging.getLogger(__name__)


async def _noop_event_handler(event) -> None:
    pass


async def main() -> None:
    token = os.environ.get("DISCORD_BOT_TOKEN")

    if token:
        bot = make_bot(token)
        on_event = make_discord_event_handler(bot)
    else:
        logger.warning("DISCORD_BOT_TOKEN not set — running without Discord")
        bot = None
        on_event = _noop_event_handler

    # Scheduler always starts first
    scheduler = TradingScheduler(on_event=on_event)
    await scheduler.start()

    # API server always starts
    await start_api_server()

    # Signal handling
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    # Discord runs as a task alongside the scheduler (optional)
    bot_task = asyncio.create_task(bot.start(token)) if bot else None

    if bot_task:
        await asyncio.wait(
            [bot_task, asyncio.create_task(stop_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
    else:
        await stop_event.wait()

    # Graceful teardown: scheduler first, then Discord
    await scheduler.stop()
    if bot and not bot.is_closed():
        await bot.close()
    if bot_task and not bot_task.done():
        bot_task.cancel()
        try:
            await bot_task
        except (asyncio.CancelledError, Exception):
            pass


def main_sync() -> int:
    init_sentry()
    setup_logging(name="rally_app")
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set.")
        return 1
    init_pool()
    init_schema()
    asyncio.run(main())
    return 0
