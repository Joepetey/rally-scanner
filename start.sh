#!/bin/bash
set -e

echo "Installing rally package..."
pip install -e ".[discord]"

echo "Creating models directory..."
mkdir -p /app/models

echo "Starting Discord bot..."
exec python3 scripts/run_discord.py
