#!/bin/bash
set -e

echo "=== Railway Deployment Start ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip --version)"

echo ""
echo "=== Checking Environment Variables ==="
if [ -z "$DISCORD_BOT_TOKEN" ]; then
    echo "ERROR: DISCORD_BOT_TOKEN is not set!"
    echo "Please set it in Railway environment variables"
    exit 1
fi
echo "✓ DISCORD_BOT_TOKEN is set"

echo ""
echo "=== Installing Package Dependencies ==="
pip install --no-cache-dir -e ".[discord]"

echo ""
echo "=== Setting Up Directories ==="
mkdir -p models
echo "✓ Models directory created/verified"

echo ""
echo "=== Starting Discord Bot ==="
exec python3 scripts/run_discord.py
