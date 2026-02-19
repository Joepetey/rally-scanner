#!/usr/bin/env python
"""
Automated pipeline orchestrator for market-rally.

Schedule via cron:
    # Daily scan at 4:30 PM ET on weekdays
    30 16 * * 1-5 cd /home/michael/rillavoice_repository/market-rally && .venv/bin/python scripts/orchestrator.py scan

    # Weekly retrain on Sunday at 6 PM ET
    0 18 * * 0 cd /home/michael/rillavoice_repository/market-rally && .venv/bin/python scripts/orchestrator.py retrain

Commands:
    python scripts/orchestrator.py scan     # daily scan + alerts
    python scripts/orchestrator.py retrain  # weekly retrain + alerts
    python scripts/orchestrator.py auto     # auto-detect: Sun=retrain, Mon-Fri=scan
    python scripts/orchestrator.py health   # print model health report
"""

import argparse
import logging
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from rally.log import setup_logging
from rally.live.scanner import scan_all
from rally.live.retrain import retrain_all
from rally.trading.positions import load_positions
from rally.core.persistence import load_manifest
from rally.bot.notify import (
    notify_signals, notify_exits, notify_retrain_complete, notify_error,
)
from rally.trading.portfolio import update_daily_snapshot, record_closed_trades

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_model_health() -> dict:
    """Check model freshness and data availability."""
    manifest = load_manifest()
    now = datetime.now()

    stale, fresh = [], []
    for ticker, info in manifest.items():
        try:
            saved_at = datetime.fromisoformat(info["saved_at"])
            age_days = (now - saved_at).days
            if age_days > 14:
                stale.append((ticker, age_days))
            else:
                fresh.append(ticker)
        except (KeyError, ValueError):
            stale.append((ticker, 999))

    return {
        "total_count": len(manifest),
        "fresh_count": len(fresh),
        "stale_count": len(stale),
        "stale_tickers": sorted(stale, key=lambda x: -x[1])[:10],
        "oldest_model_days": max((d for _, d in stale), default=0),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> int:
    """Run daily scan + position update + notify."""
    logger.info("Starting daily scan...")
    t0 = time.time()

    # Health check
    health = check_model_health()
    if health["stale_count"] > health["total_count"] * 0.5:
        msg = (f"{health['stale_count']}/{health['total_count']} models are stale "
               f"(oldest: {health['oldest_model_days']} days)")
        logger.warning(msg)
        notify_error("Model staleness warning", msg)

    # Run scan with position tracking
    results = scan_all(
        config_name=args.config,
        show_positions=True,
    )

    # Extract signals
    signals = [r for r in results if r.get("signal")]
    if signals:
        logger.info(f"{len(signals)} new signals detected")
        notify_signals(signals)
    else:
        logger.info("No new signals")

    # Check for position exits
    positions = load_positions()
    closed = positions.get("closed_today", [])
    if closed:
        logger.info(f"{len(closed)} positions closed")
        notify_exits(closed)
        record_closed_trades(closed)

    # Update daily portfolio snapshot
    update_daily_snapshot(positions, results)

    elapsed = time.time() - t0
    logger.info(f"Scan complete: {len(signals)} signals, {len(closed)} exits ({elapsed:.1f}s)")
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    """Run weekly retrain + notify."""
    logger.info("Starting weekly retrain...")
    t0 = time.time()

    try:
        retrain_all(tickers=args.tickers)
    except Exception as e:
        logger.exception("Retrain failed")
        notify_error("Retrain failed", str(e))
        return 1

    elapsed = time.time() - t0
    health = check_model_health()
    notify_retrain_complete(health, elapsed)
    logger.info(f"Retrain complete: {health['fresh_count']} models ({elapsed:.1f}s)")
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """Auto-detect what to run based on day of week."""
    now = datetime.now()
    weekday = now.weekday()

    if weekday == 6:  # Sunday
        logger.info("Sunday — running weekly retrain")
        return cmd_retrain(args)
    elif weekday < 5:  # Monday-Friday
        logger.info(f"{now.strftime('%A')} — running daily scan")
        return cmd_scan(args)
    else:
        logger.info("Saturday — nothing to do")
        return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Print model health report."""
    health = check_model_health()

    print(f"\n{'='*60}")
    print(f"  MODEL HEALTH REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"  Total models:  {health['total_count']}")
    print(f"  Fresh (<14d):  {health['fresh_count']}")
    print(f"  Stale (>14d):  {health['stale_count']}")

    if health["stale_tickers"]:
        print(f"\n  Stalest models:")
        for ticker, age in health["stale_tickers"]:
            print(f"    {ticker:<8s} {age:>4d} days old")

    # Position summary
    positions = load_positions()
    open_pos = positions.get("positions", [])
    print(f"\n  Open positions: {len(open_pos)}")
    for p in open_pos:
        pnl = p.get("unrealized_pnl_pct", 0)
        sign = "+" if pnl >= 0 else ""
        print(f"    {p['ticker']:<8s} {sign}{pnl:.2f}%  ({p.get('bars_held', 0)} bars)")

    print(f"{'='*60}\n")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Market Rally orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Run daily scan + alerts")
    p_scan.add_argument("--config", default="conservative",
                        choices=["baseline", "conservative", "aggressive", "concentrated"])

    # retrain
    p_retrain = sub.add_parser("retrain", help="Run weekly retrain + alerts")
    p_retrain.add_argument("--tickers", nargs="+", default=None)

    # auto
    p_auto = sub.add_parser("auto", help="Auto-detect: Sun=retrain, Mon-Fri=scan")
    p_auto.add_argument("--config", default="conservative",
                        choices=["baseline", "conservative", "aggressive", "concentrated"])

    # health
    sub.add_parser("health", help="Print model health report")

    args = parser.parse_args()
    setup_logging(name="orchestrator")

    commands = {
        "scan": cmd_scan,
        "retrain": cmd_retrain,
        "auto": cmd_auto,
        "health": cmd_health,
    }

    try:
        return commands[args.command](args)
    except Exception:
        logger.exception(f"Orchestrator '{args.command}' failed")
        notify_error(f"Orchestrator {args.command} crashed",
                     "Check logs for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
