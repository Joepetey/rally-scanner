"""TradingScheduler-first entrypoint.

TradingScheduler and the HTTP API server always start.
Discord connects as an optional side-task if DISCORD_BOT_TOKEN is set.
"""

import asyncio
import logging
import os
import signal

import sentry_sdk

from monitoring import init_sentry

# Initialize Sentry before any other application imports so startup errors are captured
init_sentry()

try:
    from rally_ml.core.persistence import configure as configure_manifest
    from rally_ml.core.universe import configure as configure_universe

    from api import start_api_server
    from db.ml_stores import PostgresManifestStore, PostgresUniverseCacheStore
    from db.pool import init_pool
    from db.schema import init_schema
    from integrations.discord.bot import make_bot, make_discord_event_handler
    from log import setup_logging
    from trading.scheduler import TradingScheduler
except Exception:
    sentry_sdk.capture_exception()
    raise

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

    # Give the bot access to the scheduler so !simulate can reach the stream
    if bot:
        bot.scheduler = scheduler

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
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("Bot task shutdown error", exc_info=True)


def main_sync() -> int:
    setup_logging(name="rally_app")
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set.")
        return 1
    init_pool()
    init_schema()
    configure_manifest(PostgresManifestStore())
    configure_universe(PostgresUniverseCacheStore())
    asyncio.run(main())
    return 0
