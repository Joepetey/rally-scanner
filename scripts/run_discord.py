#!/usr/bin/env python
"""
Run the Market Rally Discord bot.

Usage:
    python scripts/run_discord.py

Requires DISCORD_BOT_TOKEN in .env (or environment).

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

from rally.bot.discord_bot import make_bot
from rally.bot.discord_db import init_db
from rally.log import setup_logging


def main() -> int:
    setup_logging(name="discord_bot")

    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("ERROR: DISCORD_BOT_TOKEN not set. Add it to .env or environment.")
        return 1

    init_db()

    bot = make_bot(token)
    bot.run(token, log_handler=None)  # logging already configured
    return 0


if __name__ == "__main__":
    sys.exit(main())
