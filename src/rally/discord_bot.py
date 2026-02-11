"""Interactive Discord bot for market-rally — slash commands + rich embeds.

Reads shared disk state (positions.json, manifest.json, equity_history.csv)
for system queries. Writes to SQLite for per-user trade tracking.

Includes a built-in scheduler for scan/retrain so no external cron is needed
(useful for Railway/cloud deployment).
"""

import asyncio
import logging
import os
from datetime import datetime, time

import discord
from discord import app_commands
from discord.ext import commands, tasks

from .discord_db import (
    close_trade,
    ensure_user,
    get_open_trades,
    get_pnl_summary,
    get_trade_history,
    open_trade,
)
from .persistence import load_manifest
from .portfolio import load_equity_history, load_trade_journal
from .positions import load_positions

logger = logging.getLogger(__name__)

# Colors
GREEN = 0x00FF00
RED = 0xFF0000
BLUE = 0x0099FF
GOLD = 0xFFD700
GRAY = 0x95A5A6


def _ensure_caller(interaction: discord.Interaction) -> None:
    """Auto-register the calling Discord user."""
    ensure_user(interaction.user.id, str(interaction.user))


class RallyBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info(f"Synced {len(self.tree.get_commands())} slash commands")

    async def on_ready(self) -> None:
        logger.info(f"Bot connected as {self.user} (ID: {self.user.id})")
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="rally signals",
            )
        )


def make_bot(token: str) -> RallyBot:
    """Create and configure the bot with all slash commands."""
    bot = RallyBot()

    # ------------------------------------------------------------------
    # /signals — show latest scan signals
    # ------------------------------------------------------------------
    @bot.tree.command(name="signals", description="Show today's rally signals")
    async def cmd_signals(interaction: discord.Interaction) -> None:
        _ensure_caller(interaction)

        # Read last scan results from positions file (signals are in the scan output)
        # We check positions.json for the latest state
        state = load_positions()
        positions = state.get("positions", [])

        # We don't have a persistent "last signals" file, so show recent entries
        recent_entries = [
            p for p in positions
            if p.get("bars_held", 99) <= 1
        ]

        if not recent_entries:
            embed = discord.Embed(
                title="No Recent Signals",
                description="No new entry signals from the latest scan.",
                color=GRAY,
            )
            await interaction.response.send_message(embed=embed)
            return

        embed = discord.Embed(
            title=f"Recent Entries ({len(recent_entries)})",
            color=GREEN,
            timestamp=datetime.now(),
        )
        for p in recent_entries:
            pnl = p.get("unrealized_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            embed.add_field(
                name=p["ticker"],
                value=(
                    f"Entry: ${p['entry_price']:.2f}\n"
                    f"Stop: ${p.get('stop_price', 0):.2f}\n"
                    f"Target: ${p.get('target_price', 0):.2f}\n"
                    f"Size: {p.get('size', 0):.0%}\n"
                    f"PnL: {sign}{pnl:.2f}%"
                ),
                inline=True,
            )

        await interaction.response.send_message(embed=embed)

    # ------------------------------------------------------------------
    # /positions — show system open positions
    # ------------------------------------------------------------------
    @bot.tree.command(name="positions", description="Show system open positions")
    async def cmd_positions(interaction: discord.Interaction) -> None:
        _ensure_caller(interaction)

        state = load_positions()
        positions = state.get("positions", [])

        if not positions:
            embed = discord.Embed(
                title="No Open Positions",
                description="The system has no open positions.",
                color=GRAY,
            )
            await interaction.response.send_message(embed=embed)
            return

        total_exposure = sum(p.get("size", 0) for p in positions)
        embed = discord.Embed(
            title=f"Open Positions ({len(positions)})",
            description=f"Total exposure: {total_exposure:.0%}",
            color=BLUE,
            timestamp=datetime.now(),
        )

        for p in sorted(
            positions, key=lambda x: x.get("unrealized_pnl_pct", 0), reverse=True
        ):
            pnl = p.get("unrealized_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            embed.add_field(
                name=f"{p['ticker']} ({sign}{pnl:.2f}%)",
                value=(
                    f"Entry: ${p['entry_price']:.2f}\n"
                    f"Current: ${p.get('current_price', 0):.2f}\n"
                    f"Stop: ${p.get('stop_price', 0):.2f}\n"
                    f"Target: ${p.get('target_price', 0):.2f}\n"
                    f"Size: {p.get('size', 0):.0%} | {p.get('bars_held', 0)} bars"
                ),
                inline=True,
            )

        await interaction.response.send_message(embed=embed)

    # ------------------------------------------------------------------
    # /enter — record a trade
    # ------------------------------------------------------------------
    @bot.tree.command(name="enter", description="Record a new trade entry")
    @app_commands.describe(
        ticker="Ticker symbol (e.g. AAPL)",
        price="Entry price",
        size="Position size (default: 1.0)",
        notes="Optional notes",
    )
    async def cmd_enter(
        interaction: discord.Interaction,
        ticker: str,
        price: float,
        size: float = 1.0,
        notes: str | None = None,
    ) -> None:
        _ensure_caller(interaction)

        trade_id = open_trade(
            discord_id=interaction.user.id,
            ticker=ticker.upper(),
            entry_price=price,
            size=size,
            notes=notes,
        )

        embed = discord.Embed(
            title="Trade Entered",
            color=GREEN,
            timestamp=datetime.now(),
        )
        embed.add_field(name="Ticker", value=ticker.upper(), inline=True)
        embed.add_field(name="Entry Price", value=f"${price:.2f}", inline=True)
        embed.add_field(name="Size", value=f"{size}", inline=True)
        embed.add_field(name="Trade ID", value=str(trade_id), inline=True)
        if notes:
            embed.add_field(name="Notes", value=notes, inline=False)
        embed.set_footer(text=f"Recorded for {interaction.user}")

        await interaction.response.send_message(embed=embed)

    # ------------------------------------------------------------------
    # /exit — close a trade (FIFO)
    # ------------------------------------------------------------------
    @bot.tree.command(name="exit", description="Close your oldest open trade for a ticker")
    @app_commands.describe(
        ticker="Ticker symbol to close",
        price="Exit price",
        notes="Optional notes",
    )
    async def cmd_exit(
        interaction: discord.Interaction,
        ticker: str,
        price: float,
        notes: str | None = None,
    ) -> None:
        _ensure_caller(interaction)

        result = close_trade(
            discord_id=interaction.user.id,
            ticker=ticker.upper(),
            exit_price=price,
            notes=notes,
        )

        if result is None:
            embed = discord.Embed(
                title="No Open Trade Found",
                description=f"You have no open trades in **{ticker.upper()}**.",
                color=RED,
            )
            await interaction.response.send_message(embed=embed)
            return

        pnl = result["pnl_pct"]
        color = GREEN if pnl >= 0 else RED
        sign = "+" if pnl >= 0 else ""

        embed = discord.Embed(
            title=f"Trade Closed — {result['ticker']}",
            color=color,
            timestamp=datetime.now(),
        )
        embed.add_field(name="Entry", value=f"${result['entry_price']:.2f}", inline=True)
        embed.add_field(name="Exit", value=f"${result['exit_price']:.2f}", inline=True)
        embed.add_field(name="PnL", value=f"**{sign}{pnl:.2f}%**", inline=True)
        embed.add_field(
            name="Held",
            value=f"{result['entry_date']} → {result['exit_date']}",
            inline=False,
        )
        embed.set_footer(text=f"Trade #{result['id']} | {interaction.user}")

        await interaction.response.send_message(embed=embed)

    # ------------------------------------------------------------------
    # /history — show user's trade history
    # ------------------------------------------------------------------
    @bot.tree.command(name="history", description="Show your trade history")
    @app_commands.describe(
        ticker="Filter by ticker (optional)",
        limit="Number of trades to show (default: 10)",
    )
    async def cmd_history(
        interaction: discord.Interaction,
        ticker: str | None = None,
        limit: int = 10,
    ) -> None:
        _ensure_caller(interaction)

        trades = get_trade_history(
            discord_id=interaction.user.id,
            ticker=ticker,
            limit=limit,
        )

        if not trades:
            title = f"No trades found for {ticker.upper()}" if ticker else "No trade history"
            embed = discord.Embed(title=title, color=GRAY)
            await interaction.response.send_message(embed=embed)
            return

        title = f"Trade History — {ticker.upper()}" if ticker else "Trade History"
        embed = discord.Embed(
            title=title,
            description=f"Showing {len(trades)} trades",
            color=BLUE,
            timestamp=datetime.now(),
        )

        for t in trades:
            status = t["status"]
            if status == "closed":
                pnl = t["pnl_pct"] or 0
                sign = "+" if pnl >= 0 else ""
                value = (
                    f"Entry: ${t['entry_price']:.2f} ({t['entry_date']})\n"
                    f"Exit: ${t['exit_price']:.2f} ({t['exit_date']})\n"
                    f"PnL: **{sign}{pnl:.2f}%** | Size: {t['size']}"
                )
            else:
                value = (
                    f"Entry: ${t['entry_price']:.2f} ({t['entry_date']})\n"
                    f"Size: {t['size']} | **OPEN**"
                )

            if status == "closed":
                icon = "+" if (t.get("pnl_pct") or 0) >= 0 else "-"
            else:
                icon = "~"
            name = f"[{icon}] {t['ticker']}"
            embed.add_field(name=name, value=value, inline=False)

            if len(embed.fields) >= 25:
                break

        embed.set_footer(text=str(interaction.user))
        await interaction.response.send_message(embed=embed)

    # ------------------------------------------------------------------
    # /pnl — P&L summary
    # ------------------------------------------------------------------
    @bot.tree.command(name="pnl", description="Show your P&L summary")
    @app_commands.describe(
        period="Time period: all, 30d, 7d (default: all)",
    )
    @app_commands.choices(period=[
        app_commands.Choice(name="All time", value="all"),
        app_commands.Choice(name="Last 30 days", value="30d"),
        app_commands.Choice(name="Last 7 days", value="7d"),
    ])
    async def cmd_pnl(
        interaction: discord.Interaction,
        period: str = "all",
    ) -> None:
        _ensure_caller(interaction)

        days = None
        if period == "30d":
            days = 30
        elif period == "7d":
            days = 7

        summary = get_pnl_summary(discord_id=interaction.user.id, days=days)
        open_trades = get_open_trades(discord_id=interaction.user.id)

        period_label = {"all": "All Time", "30d": "Last 30 Days", "7d": "Last 7 Days"}[period]

        color = GREEN if summary["total_pnl"] >= 0 else RED
        sign = "+" if summary["total_pnl"] >= 0 else ""

        embed = discord.Embed(
            title=f"P&L Summary — {period_label}",
            color=color,
            timestamp=datetime.now(),
        )
        embed.add_field(
            name="Total PnL", value=f"**{sign}{summary['total_pnl']:.2f}%**", inline=True
        )
        embed.add_field(
            name="Avg PnL/Trade", value=f"{summary['avg_pnl']:+.2f}%", inline=True
        )
        embed.add_field(
            name="Win Rate", value=f"{summary['win_rate']:.0f}%", inline=True
        )
        embed.add_field(
            name="Closed Trades", value=str(summary["n_trades"]), inline=True
        )
        embed.add_field(
            name="Best Trade", value=f"+{summary['best_trade']:.2f}%", inline=True
        )
        embed.add_field(
            name="Worst Trade", value=f"{summary['worst_trade']:.2f}%", inline=True
        )
        embed.add_field(
            name="Open Trades", value=str(len(open_trades)), inline=True
        )

        embed.set_footer(text=str(interaction.user))
        await interaction.response.send_message(embed=embed)

    # ------------------------------------------------------------------
    # /health — model health report
    # ------------------------------------------------------------------
    @bot.tree.command(name="health", description="Show model health and system status")
    async def cmd_health(interaction: discord.Interaction) -> None:
        _ensure_caller(interaction)

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

        state = load_positions()
        positions = state.get("positions", [])

        color = GREEN if len(stale) == 0 else (GOLD if len(stale) < 5 else RED)

        embed = discord.Embed(
            title="System Health",
            color=color,
            timestamp=now,
        )
        embed.add_field(
            name="Total Models", value=str(len(manifest)), inline=True
        )
        embed.add_field(
            name="Fresh (<14d)", value=str(len(fresh)), inline=True
        )
        embed.add_field(
            name="Stale (>14d)", value=str(len(stale)), inline=True
        )
        embed.add_field(
            name="Open Positions", value=str(len(positions)), inline=True
        )

        if stale:
            stale_sorted = sorted(stale, key=lambda x: -x[1])[:10]
            stale_text = "\n".join(f"{t}: {d}d old" for t, d in stale_sorted)
            embed.add_field(
                name="Stalest Models", value=f"```\n{stale_text}\n```", inline=False
            )

        if positions:
            total_exposure = sum(p.get("size", 0) for p in positions)
            embed.add_field(
                name="Total Exposure", value=f"{total_exposure:.0%}", inline=True
            )

        await interaction.response.send_message(embed=embed)

    # ------------------------------------------------------------------
    # /portfolio — system equity history
    # ------------------------------------------------------------------
    @bot.tree.command(name="portfolio", description="Show system portfolio equity history")
    @app_commands.describe(days="Number of days to show (default: 30)")
    async def cmd_portfolio(
        interaction: discord.Interaction,
        days: int = 30,
    ) -> None:
        _ensure_caller(interaction)

        history = load_equity_history(days=days)
        trades = load_trade_journal(limit=10)

        embed = discord.Embed(
            title=f"Portfolio — Last {days} Days",
            color=BLUE,
            timestamp=datetime.now(),
        )

        if not history:
            embed.description = "No equity history yet. Run the orchestrator scan to start."
            await interaction.response.send_message(embed=embed)
            return

        # Summary from history
        latest = history[-1]
        embed.add_field(
            name="Date", value=latest.get("date", "?"), inline=True
        )
        embed.add_field(
            name="Positions", value=latest.get("n_positions", 0), inline=True
        )
        embed.add_field(
            name="Exposure",
            value=f"{float(latest.get('total_exposure', 0)):.0%}",
            inline=True,
        )
        embed.add_field(
            name="Signals Today", value=latest.get("n_signals_today", 0), inline=True
        )
        embed.add_field(
            name="Assets Scanned", value=latest.get("n_scanned", 0), inline=True
        )

        # Recent trades summary
        if trades:
            trade_lines = []
            for t in trades[-5:]:
                pnl = float(t.get("realized_pnl_pct", 0))
                sign = "+" if pnl >= 0 else ""
                trade_lines.append(
                    f"{t.get('ticker', '?')}: {sign}{pnl:.2f}% ({t.get('exit_reason', '?')})"
                )
            embed.add_field(
                name="Recent Closed Trades",
                value="\n".join(trade_lines),
                inline=False,
            )

        await interaction.response.send_message(embed=embed)

    # ------------------------------------------------------------------
    # Global error handler
    # ------------------------------------------------------------------
    @bot.tree.error
    async def on_app_command_error(
        interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        logger.error(f"Command error: {error}", exc_info=error)

        embed = discord.Embed(
            title="Command Error",
            description=str(error)[:4096],
            color=RED,
        )
        if interaction.response.is_done():
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=True)

    # ------------------------------------------------------------------
    # Built-in scheduler (replaces cron for cloud deployment)
    # Set ENABLE_SCHEDULER=1 in .env to activate
    # ------------------------------------------------------------------
    if os.environ.get("ENABLE_SCHEDULER", "").strip() in ("1", "true", "yes"):
        alert_channel_id = os.environ.get("DISCORD_CHANNEL_ID", "")

        async def _send_alert(embed: discord.Embed) -> None:
            """Send an embed to the alerts channel."""
            if not alert_channel_id:
                return
            channel = bot.get_channel(int(alert_channel_id))
            if channel:
                await channel.send(embed=embed)

        async def _run_scan() -> None:
            """Run the scan pipeline in a thread, then post results."""
            from .notify import _exit_embed, _signal_embed
            from .portfolio import record_closed_trades, update_daily_snapshot
            from .positions import load_positions
            from .scanner import scan_all

            logger.info("Scheduler: starting daily scan")
            results = await asyncio.to_thread(
                scan_all, None, True, "baseline"
            )
            if not results:
                return

            signals = [r for r in results if r.get("signal")]
            if signals:
                await _send_alert(discord.Embed.from_dict(_signal_embed(signals)))

            positions = load_positions()
            closed = positions.get("closed_today", [])
            if closed:
                await _send_alert(discord.Embed.from_dict(_exit_embed(closed)))
                record_closed_trades(closed)

            update_daily_snapshot(positions, results)
            logger.info(
                f"Scheduler: scan complete — {len(signals)} signals, "
                f"{len(closed)} exits"
            )

        async def _run_retrain() -> None:
            """Run retrain in a thread, then post results."""
            import time as _time

            from .notify import _retrain_embed
            from .retrain import retrain_all

            logger.info("Scheduler: starting weekly retrain")
            t0 = _time.time()
            await asyncio.to_thread(retrain_all)
            elapsed = _time.time() - t0

            from .persistence import load_manifest

            manifest = load_manifest()
            health = {
                "total_count": len(manifest),
                "fresh_count": len(manifest),
                "stale_count": 0,
            }
            await _send_alert(
                discord.Embed.from_dict(_retrain_embed(health, elapsed))
            )
            logger.info(f"Scheduler: retrain complete ({elapsed:.0f}s)")

        # Daily scan: 4:30 PM ET (21:30 UTC)
        @tasks.loop(time=time(hour=21, minute=30))
        async def scheduled_scan() -> None:
            weekday = datetime.utcnow().weekday()
            if weekday >= 5:  # skip weekends
                return
            try:
                await _run_scan()
            except Exception:
                logger.exception("Scheduled scan failed")

        # Weekly retrain: Sunday 6 PM ET (23:00 UTC)
        @tasks.loop(time=time(hour=23, minute=0))
        async def scheduled_retrain() -> None:
            if datetime.utcnow().weekday() != 6:  # Sunday only
                return
            try:
                await _run_retrain()
            except Exception:
                logger.exception("Scheduled retrain failed")

        @bot.event
        async def on_ready() -> None:
            logger.info(f"Bot connected as {bot.user} (ID: {bot.user.id})")
            await bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="rally signals",
                )
            )
            if not scheduled_scan.is_running():
                scheduled_scan.start()
                logger.info("Scheduler: daily scan armed (21:30 UTC / 4:30 PM ET)")
            if not scheduled_retrain.is_running():
                scheduled_retrain.start()
                logger.info("Scheduler: weekly retrain armed (Sun 23:00 UTC)")

    return bot
