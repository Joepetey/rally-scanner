#!/usr/bin/env python
"""Interactive .env file generator for Rally Scanner.

Safely creates a .env file from user input without risking commits to git.
Validates inputs and provides helpful prompts with links to get credentials.
"""

import os
import sys
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a bold header."""
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{BLUE}ℹ {text}{RESET}")


def get_input(prompt: str, default: str = "", required: bool = True, secret: bool = False) -> str:
    """Get user input with optional default and validation."""
    if default:
        display_prompt = f"{prompt} [{default}]: "
    else:
        display_prompt = f"{prompt}: "

    if secret:
        display_prompt = f"{YELLOW}{display_prompt}{RESET}"

    while True:
        value = input(display_prompt).strip()

        if not value and default:
            return default

        if not value and required:
            print_error("This field is required. Please enter a value.")
            continue

        return value


def validate_discord_token(token: str) -> bool:
    """Validate Discord bot token format."""
    if not token:
        return True  # Allow empty for optional
    return token.startswith("MT") or token.startswith("OT")  # Discord tokens start with MT or OT


def validate_channel_id(channel_id: str) -> bool:
    """Validate Discord channel ID format."""
    if not channel_id:
        return True  # Allow empty for optional
    return channel_id.isdigit() and len(channel_id) >= 17


def validate_anthropic_key(key: str) -> bool:
    """Validate Anthropic API key format."""
    if not key:
        return True  # Allow empty for optional
    return key.startswith("sk-ant-")


def main() -> int:
    """Run interactive .env setup."""
    print_header("Rally Scanner - Environment Setup")

    print("This script will help you create a .env file with your credentials.")
    print("Your secrets will be stored locally and NEVER committed to git.\n")

    # Check if .env already exists
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        print_warning(f".env file already exists at: {env_path}")
        overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if overwrite != "y":
            print("Aborted. Your existing .env file was not modified.")
            return 0

    # Ensure .gitignore includes .env
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            gitignore_content = f.read()
        if ".env" not in gitignore_content:
            print_warning(".env not found in .gitignore, adding it now...")
            with open(gitignore_path, "a") as f:
                f.write("\n# Environment variables (added by setup_env.py)\n.env\n")
            print_success("Added .env to .gitignore")

    env_vars = {}

    # Discord Bot Setup
    print_header("1. Discord Bot Configuration")
    print_info("Get these from: https://discord.com/developers/applications")
    print("  1. Create/select your application")
    print("  2. Go to 'Bot' section")
    print("  3. Copy the bot token")
    print("  4. Enable 'MESSAGE CONTENT INTENT'\n")

    while True:
        token = get_input("Discord Bot Token", required=True, secret=True)
        if validate_discord_token(token):
            env_vars["DISCORD_BOT_TOKEN"] = token
            break
        print_error("Invalid token format. Discord tokens start with 'MT' or 'OT'")

    print_info("\nGet Channel ID:")
    print("  1. Enable Developer Mode in Discord (Settings → Advanced)")
    print("  2. Right-click your alerts channel → Copy Channel ID\n")

    while True:
        channel_id = get_input("Discord Channel ID (for alerts)", required=True)
        if validate_channel_id(channel_id):
            env_vars["DISCORD_CHANNEL_ID"] = channel_id
            break
        print_error("Invalid channel ID. Should be 17-19 digits.")

    # Optional Discord webhook
    webhook = get_input("Discord Webhook URL (optional, press Enter to skip)", required=False)
    if webhook:
        env_vars["DISCORD_WEBHOOK_URL"] = webhook

    # Claude API Setup
    print_header("2. Claude API Configuration")
    print_info("Get your API key from: https://console.anthropic.com")
    print("  1. Sign up or log in")
    print("  2. Add a payment method (you get $5 free credit)")
    print("  3. Go to Settings → API Keys")
    print("  4. Create a new key\n")

    while True:
        api_key = get_input("Anthropic API Key", required=True, secret=True)
        if validate_anthropic_key(api_key):
            env_vars["ANTHROPIC_API_KEY"] = api_key
            break
        print_error("Invalid API key format. Should start with 'sk-ant-'")

    model = get_input(
        "Claude Model",
        default="claude-sonnet-4-5-20250929",
        required=False
    )
    env_vars["CLAUDE_MODEL"] = model or "claude-sonnet-4-5-20250929"

    max_tokens = get_input(
        "Max Tokens per Response",
        default="2048",
        required=False
    )
    env_vars["CLAUDE_MAX_TOKENS"] = max_tokens or "2048"

    # Scheduler
    print_header("3. Scheduler Configuration")
    print_info("Enable scheduler for automated scans and retraining")
    print("  - Daily scans: Mon-Fri at 4:30 PM ET (21:30 UTC)")
    print("  - Weekly retrain: Sunday at 6:00 PM ET (23:00 UTC)")
    print("  - Set to 1 for Railway/cloud deployment")
    print("  - Set to 0 for local testing\n")

    scheduler = get_input("Enable Scheduler? (0 or 1)", default="0", required=False)
    env_vars["ENABLE_SCHEDULER"] = scheduler or "0"

    # Optional configs
    print_header("4. Optional Configurations (press Enter to skip)")

    # Telegram (even though we removed it from code, keep for backwards compat)
    telegram_token = get_input("Telegram Bot Token (optional)", required=False)
    if telegram_token:
        env_vars["TELEGRAM_BOT_TOKEN"] = telegram_token
        telegram_chat = get_input("Telegram Chat ID (optional)", required=False)
        if telegram_chat:
            env_vars["TELEGRAM_CHAT_ID"] = telegram_chat

    # Email (even though we removed it from code, keep for backwards compat)
    smtp_host = get_input("SMTP Host (optional, e.g., smtp.gmail.com)", required=False)
    if smtp_host:
        env_vars["SMTP_HOST"] = smtp_host
        env_vars["SMTP_PORT"] = get_input("SMTP Port", default="587", required=False) or "587"
        env_vars["SMTP_USER"] = get_input("SMTP Username", required=False)
        env_vars["SMTP_PASSWORD"] = get_input("SMTP Password", required=False, secret=True)
        env_vars["NOTIFY_EMAIL"] = get_input("Notification Email Address", required=False)

    # Generic webhook
    generic_webhook = get_input("Generic Webhook URL (optional)", required=False)
    if generic_webhook:
        env_vars["WEBHOOK_URL"] = generic_webhook

    # Write .env file
    print_header("5. Writing Configuration")

    try:
        with open(env_path, "w") as f:
            f.write("# Rally Scanner Environment Configuration\n")
            f.write("# Generated by scripts/setup_env.py\n")
            f.write(f"# DO NOT COMMIT THIS FILE TO GIT\n\n")

            # Write in organized sections
            f.write("# --- Discord Bot ---\n")
            for key in ["DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_WEBHOOK_URL"]:
                if key in env_vars:
                    f.write(f"{key}={env_vars[key]}\n")

            f.write("\n# --- Claude API ---\n")
            for key in ["ANTHROPIC_API_KEY", "CLAUDE_MODEL", "CLAUDE_MAX_TOKENS"]:
                if key in env_vars:
                    f.write(f"{key}={env_vars[key]}\n")

            f.write("\n# --- Scheduler ---\n")
            if "ENABLE_SCHEDULER" in env_vars:
                f.write(f"ENABLE_SCHEDULER={env_vars['ENABLE_SCHEDULER']}\n")

            # Optional sections
            if any(k.startswith("TELEGRAM_") for k in env_vars):
                f.write("\n# --- Telegram (optional) ---\n")
                for key in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
                    if key in env_vars:
                        f.write(f"{key}={env_vars[key]}\n")

            if any(k.startswith("SMTP_") or k == "NOTIFY_EMAIL" for k in env_vars):
                f.write("\n# --- Email (optional) ---\n")
                for key in ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "NOTIFY_EMAIL"]:
                    if key in env_vars:
                        f.write(f"{key}={env_vars[key]}\n")

            if "WEBHOOK_URL" in env_vars:
                f.write("\n# --- Generic Webhook (optional) ---\n")
                f.write(f"WEBHOOK_URL={env_vars['WEBHOOK_URL']}\n")

        print_success(f"Configuration written to: {env_path}")

        # Set file permissions to owner-only read/write
        os.chmod(env_path, 0o600)
        print_success("File permissions set to 600 (owner read/write only)")

        print_header("Setup Complete!")
        print(f"{GREEN}✓ Your .env file is ready!{RESET}\n")
        print("Next steps:")
        print("  1. Test locally: python scripts/run_discord.py")
        print("  2. Or deploy to Railway (copy values to Railway Variables)")
        print(f"\n{YELLOW}IMPORTANT:{RESET} Never commit the .env file to git!")
        print(f"            It's in .gitignore, so git will ignore it.\n")

        return 0

    except Exception as e:
        print_error(f"Failed to write .env file: {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
