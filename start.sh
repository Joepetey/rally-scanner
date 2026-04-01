#!/bin/bash
set -e

echo "=== Railway Deployment Start ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip --version)"

echo ""
echo "=== Checking Environment Variables ==="
if [ -z "$DISCORD_BOT_TOKEN" ]; then
    echo "· DISCORD_BOT_TOKEN not set — Discord integration disabled"
else
    echo "✓ DISCORD_BOT_TOKEN is set"
fi

if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL is not set!"
    echo "Please add a PostgreSQL service to this Railway project"
    exit 1
fi
echo "✓ DATABASE_URL is set"

if [ "$ALPACA_AUTO_EXECUTE" = "1" ]; then
    if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
        echo "WARNING: ALPACA_AUTO_EXECUTE=1 but API keys not set"
    else
        echo "✓ Alpaca auto-execution enabled (paper=${ALPACA_PAPER_TRADE:-true})"
    fi
else
    echo "· Alpaca auto-execution disabled"
fi

echo ""
echo "=== Setting Up Directories ==="
mkdir -p models
echo "✓ Models directory created/verified"

echo ""
echo "=== Starting rally-scanner ==="
echo "✓ API server will listen on port ${PORT:-8080}"
exec python -c "from app import main_sync; main_sync()"
