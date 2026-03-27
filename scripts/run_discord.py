#!/usr/bin/env python
"""
Run the Market Rally Discord bot.

Usage:
    python scripts/run_discord.py

Requires DISCORD_BOT_TOKEN and DATABASE_URL in .env (or environment).

Setup:
    1. Create a bot at https://discord.com/developers/applications
    2. Enable MESSAGE CONTENT intent in Bot settings
    3. Generate invite URL with bot + applications.commands scopes
    4. Set DISCORD_BOT_TOKEN in .env
    5. Optionally set DISCORD_CHANNEL_ID for orchestrator alerts
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from integrations.discord.bot import make_bot
from db import init_pool, init_schema
from log import setup_logging
from monitoring import init_sentry


def main() -> int:
    init_sentry()
    setup_logging(name="discord_bot")

    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("ERROR: DISCORD_BOT_TOKEN not set. Add it to .env or environment.")
        return 1

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set. Add it to .env or environment.")
        return 1

    init_pool()
    init_schema()

    bot = make_bot(token)
    bot.run(token, log_handler=None)  # logging already configured
    return 0


if __name__ == "__main__":
    sys.exit(main())
