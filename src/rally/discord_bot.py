"""Interactive Discord bot for market-rally — slash commands + rich embeds.

Reads shared disk state (positions.json, manifest.json, equity_history.csv)
for system queries. Writes to SQLite for per-user trade tracking.

Includes a built-in scheduler for scan/retrain so no external cron is needed
(useful for Railway/cloud deployment).
"""

import asyncio
import logging
import os
import zoneinfo
from datetime import datetime, time

import discord
from discord.ext import commands, tasks

from .claude_agent import process_message
from .config import PARAMS
from .discord_db import (
    ensure_user,
    get_conversation_history,
    save_conversation_history,
)
from .persistence import load_manifest
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


def _ensure_caller_from_message(message: discord.Message) -> None:
    """Auto-register user from message (not interaction)."""
    ensure_user(message.author.id, str(message.author))


def _get_system_recommendation(ticker: str) -> dict | None:
    """Look up the system's current recommendation for a ticker.

    Checks open positions first (already entered by the system), then
    falls back to the latest scan results cached in positions.json.
    Returns dict with size, stop_price, target_price, entry_price or None.
    """

    state = load_positions()

    # Check open positions (system already entered this trade)
    for p in state.get("positions", []):
        if p.get("ticker", "").upper() == ticker.upper():
            return {
                "entry_price": p.get("entry_price"),
                "size": p.get("size", 0),
                "stop_price": p.get("stop_price"),
                "target_price": p.get("target_price"),
            }

    return None


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
    """Create and configure the agentic Discord bot."""
    bot = RallyBot()

    # ------------------------------------------------------------------
    # Async task handlers for long-running operations
    # ------------------------------------------------------------------
    async def _run_retrain_task(
        channel: discord.abc.Messageable,
        tickers: list[str] | None = None
    ) -> None:
        """Run model retraining with progress updates."""
        import time

        from .retrain import retrain_all

        try:
            # Send initial message
            await channel.send(
                "🔄 **Starting model retraining...**\n"
                "This will take 10-30+ minutes depending on the number of tickers."
            )

            # Run retrain in thread and time it
            start_time = time.time()

            # Send progress update every 5 minutes
            async def send_updates():
                elapsed = 0
                while True:
                    await asyncio.sleep(300)  # 5 minutes
                    elapsed += 5
                    await channel.send(f"⏳ Still training... {elapsed} minutes elapsed")

            # Start update task
            update_task = asyncio.create_task(send_updates())

            try:
                # Run the actual retraining
                await asyncio.to_thread(retrain_all, tickers)
            finally:
                # Cancel update task
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass

            # Calculate elapsed time
            elapsed = time.time() - start_time
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)

            # Get model count
            manifest = load_manifest()
            model_count = len(manifest) if manifest else 0

            # Send completion message
            await channel.send(
                f"✅ **Retraining complete!**\n"
                f"• Trained: {model_count} models\n"
                f"• Time: {minutes}m {seconds}s\n"
                f"• You can now run scans to find signals!"
            )

        except Exception as e:
            logger.exception("Retrain task failed")
            await channel.send(f"❌ **Retraining failed:** {str(e)}")

    # ------------------------------------------------------------------
    # Message handler for natural language interaction via Claude
    # ------------------------------------------------------------------
    @bot.event
    async def on_message(message: discord.Message) -> None:
        """Handle natural language messages via Claude."""
        # Ignore bot's own messages
        if message.author == bot.user:
            return

        # Only respond to DMs or mentions
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = bot.user in message.mentions

        if not (is_dm or is_mentioned):
            return

        # Check if Claude is enabled
        if not os.environ.get("ANTHROPIC_API_KEY"):
            await message.channel.send(
                "💡 **Tip**: Set `ANTHROPIC_API_KEY` in `.env` to enable the agentic bot!"
            )
            return

        # Ensure user is registered
        _ensure_caller_from_message(message)

        # Get conversation history
        history = await asyncio.to_thread(
            get_conversation_history,
            message.author.id
        )

        # Show typing indicator
        async with message.channel.typing():
            try:
                # Remove mention from message if present
                content = message.content
                if bot.user.mentioned_in(message):
                    content = content.replace(f"<@{bot.user.id}>", "").strip()

                # Process message with Claude (runs in thread to avoid blocking)
                response_text, updated_history, async_tasks = await asyncio.to_thread(
                    process_message,
                    content,
                    message.author.id,
                    str(message.author),
                    history
                )

                # Save updated history
                await asyncio.to_thread(
                    save_conversation_history,
                    message.author.id,
                    updated_history
                )

                # Send response (split if too long for Discord's 2000 char limit)
                if len(response_text) <= 2000:
                    await message.channel.send(response_text)
                else:
                    # Split into chunks
                    chunks = [response_text[i:i+2000] for i in range(0, len(response_text), 2000)]
                    for chunk in chunks:
                        await message.channel.send(chunk)

                # Handle async tasks (long-running operations like retrain)
                for task in async_tasks:
                    if task.get("_async_task") == "retrain":
                        # Spawn retrain task in background
                        asyncio.create_task(_run_retrain_task(
                            message.channel,
                            task.get("tickers")
                        ))

            except Exception as e:
                logger.exception("Claude message processing failed")
                await message.channel.send(
                    f"⚠️ Error processing your request: {str(e)}"
                )

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

        async def _send_error_alert(task_name: str, error: Exception) -> None:
            """Send an error embed to the alerts channel."""
            import traceback

            from .notify import _error_embed
            tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            details = f"```\n{tb[-3500:]}\n```"
            embed = discord.Embed.from_dict(_error_embed(f"{task_name} Failed", details))
            await _send_alert(embed)

        async def _run_scan() -> None:
            """Run the scan pipeline in a thread, then post results."""
            from .notify import _exit_embed, _signal_embed
            from .portfolio import record_closed_trades, update_daily_snapshot
            from .positions import load_positions
            from .scanner import scan_all

            logger.info("Scheduler: starting daily scan")
            results = await asyncio.to_thread(
                scan_all, None, True, "conservative"
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

            # Auto-execute on Alpaca if enabled
            from .alpaca_executor import is_enabled as alpaca_enabled
            if alpaca_enabled():
                from .alpaca_executor import (
                    execute_entries,
                    execute_exits,
                    get_account_equity,
                )
                from .notify import _order_embed, _order_failure_embed

                # Filter out signals for tickers we already hold
                open_tickers = {
                    p["ticker"] for p in positions.get("positions", [])
                }
                new_signals = [
                    s for s in signals if s["ticker"] not in open_tickers
                ]

                # External system boundary — broker outage must not break scan
                try:
                    equity = await get_account_equity()
                    if new_signals:
                        entry_results = await execute_entries(new_signals, equity=equity)
                        await _store_order_ids(entry_results)
                        ok = [r for r in entry_results if r.success]
                        fail = [r for r in entry_results if not r.success]
                        if ok:
                            await _send_alert(
                                discord.Embed.from_dict(_order_embed(ok, equity))
                            )
                        if fail:
                            await _send_alert(
                                discord.Embed.from_dict(_order_failure_embed(fail))
                            )
                    if closed:
                        exit_results = await execute_exits(closed)
                        ok = [r for r in exit_results if r.success]
                        fail = [r for r in exit_results if not r.success]
                        if ok:
                            await _send_alert(
                                discord.Embed.from_dict(_order_embed(ok, equity))
                            )
                        if fail:
                            await _send_alert(
                                discord.Embed.from_dict(_order_failure_embed(fail))
                            )
                except Exception as e:
                    logger.exception("Alpaca execution failed")
                    await _send_error_alert("Alpaca Execution", e)

            logger.info(
                f"Scheduler: scan complete — {len(signals)} signals, "
                f"{len(closed)} exits"
            )

        async def _store_order_ids(results: list) -> None:
            """Save Alpaca order IDs + trailing stop IDs to positions.json."""
            from .positions import async_save_positions, load_positions
            state = load_positions()
            for result in results:
                if result.success and result.order_id:
                    for pos in state["positions"]:
                        if pos["ticker"] == result.ticker:
                            pos["order_id"] = result.order_id
                            if result.qty:
                                pos["qty"] = result.qty
                            if result.trail_order_id:
                                pos["trail_order_id"] = result.trail_order_id
                            break
            await async_save_positions(state)

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

        # -- Price alert configuration --
        _alert_interval = int(os.environ.get("PRICE_ALERT_INTERVAL_MINUTES", "15"))
        _alert_proximity_pct = float(os.environ.get("ALERT_PROXIMITY_PCT", "1.5"))
        _sent_alerts: set[tuple[str, str, str]] = set()  # (ticker, alert_type, date)

        _ET = zoneinfo.ZoneInfo("America/New_York")

        def _is_market_open() -> bool:
            """True if Mon-Fri 9:30 AM - 4:00 PM ET."""
            now = datetime.now(_ET)
            if now.weekday() >= 5:
                return False
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            return market_open <= now <= market_close

        async def _check_price_alerts() -> None:
            """Fetch live prices for open positions and alert on breaches."""
            from .alpaca_executor import is_enabled as alpaca_enabled
            from .data import fetch_quotes
            from .notify import _approaching_alert_embed, _price_alert_embed
            from .positions import load_positions

            if not _is_market_open():
                return

            state = load_positions()
            positions = state.get("positions", [])
            if not positions:
                return

            today = datetime.now(_ET).strftime("%Y-%m-%d")

            # Clear stale alerts from previous days
            stale = {k for k in _sent_alerts if k[2] != today}
            _sent_alerts.difference_update(stale)

            tickers = [p["ticker"] for p in positions]

            if alpaca_enabled():
                from .alpaca_executor import (
                    check_pending_fills,
                    get_snapshots,
                    place_trailing_stop,
                )
                from .positions import (
                    async_save_positions,
                    update_fill_prices,
                )
                from .positions import (
                    load_positions as reload_positions,
                )

                quotes = await get_snapshots(tickers)

                # Update entry prices for orders that filled since last check
                pending_ids = [p.get("order_id") for p in positions if p.get("order_id")]
                if pending_ids:
                    fills = await check_pending_fills(pending_ids)
                    if fills:
                        n_filled = update_fill_prices(fills)
                        if n_filled:
                            logger.info(f"Updated {n_filled} fill prices from Alpaca")

                # Place deferred trailing stops for filled entries without one
                fresh_state = reload_positions()
                for pos in fresh_state.get("positions", []):
                    if (not pos.get("order_id")        # fill confirmed (order_id removed)
                            and not pos.get("trail_order_id")  # no trailing stop yet
                            and pos.get("entry_price")):
                        atr_pct = pos.get("atr", 0) / pos["entry_price"] if pos.get("atr") else 0.02
                        trail_pct = round(max(1.5 * atr_pct * 100, 1.0), 2)
                        qty = pos.get("qty")
                        if not qty:
                            # Estimate qty from broker positions
                            try:
                                from .alpaca_executor import get_all_positions
                                broker_pos = await get_all_positions()
                                for bp in broker_pos:
                                    if bp["ticker"] == pos["ticker"]:
                                        qty = bp["qty"]
                                        pos["qty"] = qty
                                        break
                            except Exception:
                                pass
                        if not qty:
                            logger.warning(
                                "Cannot place trailing stop for %s: qty unknown",
                                pos["ticker"],
                            )
                            continue
                        trail_id = await place_trailing_stop(
                            pos["ticker"], qty, trail_pct,
                        )
                        if trail_id:
                            pos["trail_order_id"] = trail_id
                            logger.info(
                                "Deferred trailing stop placed for %s: %s",
                                pos["ticker"], trail_id,
                            )
                await async_save_positions(fresh_state)

                # Reload positions so breach checks use updated fill prices,
                # trailing stops, and qty values
                state = reload_positions()
                positions = state.get("positions", [])
            else:
                quotes = await asyncio.to_thread(fetch_quotes, tickers)

            breach_alerts: list[dict] = []
            approach_alerts: list[dict] = []

            for pos in positions:
                ticker = pos["ticker"]
                quote = quotes.get(ticker)
                if not quote or "error" in quote:
                    continue

                price = quote["price"]
                entry = pos["entry_price"]
                stop = pos.get("stop_price", 0)
                target = pos.get("target_price", 0)
                trailing = pos.get("trailing_stop", 0)
                effective_stop = max(stop, trailing)
                pnl_pct = round((price / entry - 1) * 100, 2) if entry else 0

                # Check stop breach
                if effective_stop > 0 and price <= effective_stop:
                    key = (ticker, "stop_breached", today)
                    if key not in _sent_alerts:
                        _sent_alerts.add(key)
                        level_name = "Trailing Stop" if trailing > stop else "Stop"
                        alert = {
                            "ticker": ticker,
                            "alert_type": "stop_breached",
                            "current_price": price,
                            "level_price": effective_stop,
                            "level_name": level_name,
                            "entry_price": entry,
                            "pnl_pct": pnl_pct,
                        }
                        # Execute exit immediately on Alpaca
                        if alpaca_enabled():
                            try:
                                from .alpaca_executor import execute_exit
                                from .positions import async_close_position
                                trail_oid = pos.get("trail_order_id")
                                result = await execute_exit(
                                    ticker, trail_order_id=trail_oid,
                                )
                                fill = result.fill_price or price
                                await async_close_position(ticker, fill, "stop")
                                alert["order_result"] = result.model_dump()
                            except Exception:
                                logger.exception(f"Alpaca exit failed for {ticker}")
                        breach_alerts.append(alert)

                # Check target breach
                elif target > 0 and price >= target:
                    key = (ticker, "target_breached", today)
                    if key not in _sent_alerts:
                        _sent_alerts.add(key)
                        alert = {
                            "ticker": ticker,
                            "alert_type": "target_breached",
                            "current_price": price,
                            "level_price": target,
                            "level_name": "Target",
                            "entry_price": entry,
                            "pnl_pct": pnl_pct,
                        }
                        # Execute exit immediately on Alpaca
                        if alpaca_enabled():
                            try:
                                from .alpaca_executor import execute_exit
                                from .positions import async_close_position
                                trail_oid = pos.get("trail_order_id")
                                result = await execute_exit(
                                    ticker, trail_order_id=trail_oid,
                                )
                                fill = result.fill_price or price
                                await async_close_position(ticker, fill, "profit_target")
                                alert["order_result"] = result.model_dump()
                            except Exception:
                                logger.exception(f"Alpaca exit failed for {ticker}")
                        breach_alerts.append(alert)

                # Check approaching stop
                elif _alert_proximity_pct > 0 and effective_stop > 0:
                    distance = (price / effective_stop - 1) * 100
                    if 0 < distance <= _alert_proximity_pct:
                        key = (ticker, "near_stop", today)
                        if key not in _sent_alerts:
                            _sent_alerts.add(key)
                            level_name = "Trailing Stop" if trailing > stop else "Stop"
                            approach_alerts.append({
                                "ticker": ticker,
                                "alert_type": "near_stop",
                                "current_price": price,
                                "level_price": effective_stop,
                                "level_name": level_name,
                                "distance_pct": round(distance, 1),
                                "entry_price": entry,
                                "pnl_pct": pnl_pct,
                            })

                # Check approaching target
                elif _alert_proximity_pct > 0 and target > 0:
                    distance = (target / price - 1) * 100
                    if 0 < distance <= _alert_proximity_pct:
                        key = (ticker, "near_target", today)
                        if key not in _sent_alerts:
                            _sent_alerts.add(key)
                            approach_alerts.append({
                                "ticker": ticker,
                                "alert_type": "near_target",
                                "current_price": price,
                                "level_price": target,
                                "level_name": "Target",
                                "distance_pct": round(distance, 1),
                                "entry_price": entry,
                                "pnl_pct": pnl_pct,
                            })

            if breach_alerts:
                await _send_alert(
                    discord.Embed.from_dict(_price_alert_embed(breach_alerts))
                )
            if approach_alerts:
                await _send_alert(
                    discord.Embed.from_dict(_approaching_alert_embed(approach_alerts))
                )

            n = len(breach_alerts) + len(approach_alerts)
            if n:
                logger.info(f"Price alerts: {len(breach_alerts)} breaches, "
                            f"{len(approach_alerts)} warnings")

        async def _reconcile() -> None:
            """Reconcile local positions with broker state."""
            from .alpaca_executor import (
                check_trail_stop_fills,
                get_all_positions,
            )
            from .alpaca_executor import (
                is_enabled as alpaca_enabled,
            )
            from .notify import _error_embed
            from .positions import get_trail_order_ids, reconcile_with_broker

            if not alpaca_enabled() or not _is_market_open():
                return

            broker_positions = await get_all_positions()
            trail_ids = get_trail_order_ids()
            trail_fills = await check_trail_stop_fills(trail_ids) if trail_ids else {}

            warnings = reconcile_with_broker(broker_positions, trail_fills)
            if warnings:
                details = "\n".join(f"- {w}" for w in warnings)
                embed = discord.Embed.from_dict(
                    _error_embed("Position Reconciliation", details)
                )
                # Use orange for warnings, not red
                embed.color = 0xFF8C00
                await _send_alert(embed)
                logger.info(f"Reconciliation: {len(warnings)} issues found")

        # -- Proactive state --
        _cached_regime_states: dict = {}
        _watchlist_tickers: list[str] = []
        _current_alert_interval = PARAMS.base_alert_interval
        _last_alert_check = datetime.min.replace(tzinfo=_ET)

        async def _check_regime_shifts() -> None:
            """Check HMM regime states and alert on transitions."""
            from .notify import _regime_shift_embed
            from .regime_monitor import check_regime_shifts, is_cascade

            if not PARAMS.regime_check_enabled or not _is_market_open():
                return

            transitions = await asyncio.to_thread(check_regime_shifts)
            if not transitions:
                return

            # Update cached regime states
            from .regime_monitor import get_regime_states
            nonlocal _cached_regime_states
            _cached_regime_states = get_regime_states()

            await _send_alert(
                discord.Embed.from_dict(_regime_shift_embed(transitions))
            )

            # Cascade: trigger early scan if many assets shifting
            if is_cascade(transitions):
                tickers_shifted = [t["ticker"] for t in transitions]
                logger.info(
                    "Regime cascade detected (%d shifts) — triggering early scan",
                    len(transitions),
                )
                await _send_alert(discord.Embed(
                    title="Regime Cascade — Early Scan Triggered",
                    description=(
                        f"{len(transitions)} regime shifts detected simultaneously.\n"
                        f"Tickers: {', '.join(tickers_shifted)}\n"
                        "Running full scan now..."
                    ),
                    color=0xFF4500,
                ))
                await _run_scan()

        async def _run_risk_evaluation() -> None:
            """Run proactive risk evaluation on current positions."""
            from .notify import _risk_action_embed
            from .risk_manager import evaluate, execute_actions

            if not PARAMS.proactive_risk_enabled:
                return

            state = load_positions()
            positions = state.get("positions", [])
            if not positions:
                return

            # Get equity
            try:
                from .alpaca_executor import get_account_equity
                from .alpaca_executor import is_enabled as alpaca_enabled
                if alpaca_enabled():
                    equity = await get_account_equity()
                else:
                    equity = 100_000  # default for non-Alpaca mode
            except Exception:
                equity = 100_000

            actions = await asyncio.to_thread(
                evaluate, equity, positions, _cached_regime_states,
            )

            if not actions:
                return

            results = await execute_actions(actions, positions)

            # Filter to meaningful results (skip no-ops)
            meaningful = [r for r in results if not r.get("skipped")]
            if meaningful:
                await _send_alert(
                    discord.Embed.from_dict(_risk_action_embed(meaningful))
                )
                logger.info(
                    "Risk evaluation: %d actions taken (%s)",
                    len(meaningful),
                    ", ".join(f"{r['ticker']}:{r['action']}" for r in meaningful),
                )

        async def _run_midday_scan() -> None:
            """Run a lightweight scan on watchlist tickers only."""
            from .notify import _signal_embed
            from .scanner import scan_watchlist

            if not PARAMS.midday_scans_enabled or not _is_market_open():
                return

            if not _watchlist_tickers:
                logger.debug("Mid-day scan: empty watchlist, skipping")
                return

            logger.info(
                "Mid-day scan: checking %d watchlist tickers",
                len(_watchlist_tickers),
            )
            results = await asyncio.to_thread(
                scan_watchlist, _watchlist_tickers,
            )

            signals = [r for r in results if r.get("signal")]
            if signals:
                embed = discord.Embed.from_dict(_signal_embed(signals))
                embed.title = f"Mid-day Signal ({len(signals)})"
                embed.set_footer(text="From watchlist mid-day scan")
                await _send_alert(embed)

        def _should_use_fast_alerts() -> bool:
            """Check if conditions warrant faster alert frequency."""
            if not PARAMS.adaptive_alerts_enabled:
                return False

            state = load_positions()
            positions = state.get("positions", [])

            # Check 1: any position within stop_proximity_pct of stop
            for pos in positions:
                price = pos.get("current_price", 0)
                stop = max(pos.get("stop_price", 0), pos.get("trailing_stop", 0))
                if stop > 0 and price > 0:
                    distance_pct = (price / stop - 1) * 100
                    if 0 < distance_pct <= PARAMS.stop_proximity_pct:
                        return True

            # Check 2: portfolio drawdown > Tier 1 threshold
            try:
                from .alpaca_executor import is_enabled as alpaca_enabled
                from .portfolio import compute_drawdown
                if alpaca_enabled():
                    # Use cached equity if available, skip if not
                    pass
                dd = compute_drawdown(100_000)
                if dd >= PARAMS.risk_tier1_dd:
                    return True
            except Exception:
                pass

            return False

        # Price alerts: runs every 1 minute, checks interval adaptively
        @tasks.loop(minutes=1)
        async def scheduled_price_alerts() -> None:
            nonlocal _current_alert_interval, _last_alert_check
            try:
                now = datetime.now(_ET)

                # Determine interval
                if PARAMS.adaptive_alerts_enabled:
                    fast = _should_use_fast_alerts()
                    new_interval = (
                        PARAMS.fast_alert_interval if fast
                        else PARAMS.base_alert_interval
                    )
                    if new_interval != _current_alert_interval:
                        logger.info(
                            "Alert frequency: %dm → %dm (%s mode)",
                            _current_alert_interval, new_interval,
                            "fast" if fast else "normal",
                        )
                        _current_alert_interval = new_interval

                # Check if enough time has elapsed since last check
                elapsed = (now - _last_alert_check).total_seconds() / 60
                if elapsed < _current_alert_interval:
                    return

                _last_alert_check = now
                await _check_price_alerts()

                # Run risk evaluation after price alerts
                await _run_risk_evaluation()
            except Exception as e:
                logger.exception("Price alert check failed")
                await _send_error_alert("Price Alerts", e)

        # Daily scan: 4:30 PM ET (21:30 UTC)
        @tasks.loop(time=time(hour=21, minute=30))
        async def scheduled_scan() -> None:
            nonlocal _watchlist_tickers
            weekday = datetime.utcnow().weekday()
            if weekday >= 5:  # skip weekends
                return
            try:
                await _run_scan()

                # Update watchlist for mid-day scans
                from .persistence import load_manifest
                manifest = load_manifest()
                if manifest:
                    _watchlist_tickers = sorted(manifest.keys())

                # Run risk evaluation after scan (picks up regime changes)
                await _run_risk_evaluation()
            except Exception as e:
                logger.exception("Scheduled scan failed")
                await _send_error_alert("Daily Scan", e)

        # Regime check: every 30 minutes during market hours
        @tasks.loop(minutes=30)
        async def scheduled_regime_check() -> None:
            try:
                await _check_regime_shifts()
            except Exception as e:
                logger.exception("Regime check failed")
                await _send_error_alert("Regime Check", e)

        # Mid-day scans: 2 PM and 3 PM ET (19:00, 20:00 UTC)
        @tasks.loop(time=[time(hour=19, minute=0), time(hour=20, minute=0)])
        async def scheduled_midday_scan() -> None:
            weekday = datetime.utcnow().weekday()
            if weekday >= 5:
                return
            try:
                await _run_midday_scan()
            except Exception as e:
                logger.exception("Mid-day scan failed")
                await _send_error_alert("Mid-day Scan", e)

        # Reconciliation: every 30 minutes during market hours
        @tasks.loop(minutes=30)
        async def scheduled_reconcile() -> None:
            try:
                await _reconcile()
            except Exception as e:
                logger.exception("Reconciliation failed")
                await _send_error_alert("Reconciliation", e)

        # Weekly retrain: Sunday 6 PM ET (23:00 UTC)
        @tasks.loop(time=time(hour=23, minute=0))
        async def scheduled_retrain() -> None:
            if datetime.utcnow().weekday() != 6:  # Sunday only
                return
            try:
                await _run_retrain()
            except Exception as e:
                logger.exception("Scheduled retrain failed")
                await _send_error_alert("Weekly Retrain", e)

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
            if not scheduled_price_alerts.is_running():
                scheduled_price_alerts.start()
                logger.info(
                    "Scheduler: price alerts armed "
                    f"(adaptive: {PARAMS.base_alert_interval}m / "
                    f"{PARAMS.fast_alert_interval}m fast)"
                )
            if not scheduled_reconcile.is_running():
                scheduled_reconcile.start()
                logger.info("Scheduler: reconciliation armed (every 30m during market hours)")
            if not scheduled_regime_check.is_running():
                scheduled_regime_check.start()
                logger.info("Scheduler: regime check armed (every 30m during market hours)")
            if not scheduled_midday_scan.is_running():
                scheduled_midday_scan.start()
                logger.info("Scheduler: mid-day scans armed (2 PM, 3 PM ET)")

            # Populate watchlist from manifest so mid-day scans work immediately
            try:
                from .persistence import load_manifest
                manifest = load_manifest()
                if manifest:
                    nonlocal _watchlist_tickers
                    _watchlist_tickers = sorted(manifest.keys())
                    logger.info(
                        "Loaded %d tickers into watchlist from manifest",
                        len(_watchlist_tickers),
                    )
            except Exception:
                logger.debug("No manifest found — watchlist empty until first scan")

            # Run initial scan if market is open (don't wait for 4:30 PM)
            if _is_market_open():
                logger.info("Market is open — running startup scan")
                try:
                    await _run_scan()
                except Exception as e:
                    logger.exception("Startup scan failed")
                    await _send_error_alert("Startup Scan", e)

    return bot
