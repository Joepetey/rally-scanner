"""Interactive Discord bot for market-rally — slash commands + rich embeds.

Reads position data from Alpaca + SQLite, manifest, and equity history.
Writes to SQLite for per-user trade tracking.

Includes a built-in scheduler for scan/retrain so no external cron is needed
(useful for Railway/cloud deployment).

Exposes an HTTP API on $PORT (default 8080) for local tools to query
positions from the Railway deployment.
"""

import asyncio
import json as _json
import logging
import os
import zoneinfo
from datetime import datetime, time

import discord
from aiohttp import web
from discord.ext import commands, tasks

from config import PARAMS
from core.persistence import load_manifest
from db.positions import load_positions, save_position_meta
from trading.positions import get_merged_positions

from db.conversations import get_conversation_history, save_conversation_history
from db.users import ensure_user

from .agent import process_message

logger = logging.getLogger(__name__)

# Colors
GREEN = 0x00FF00
RED = 0xFF0000
BLUE = 0x0099FF
GOLD = 0xFFD700
GRAY = 0x95A5A6


# ---------------------------------------------------------------------------
# HTTP API server (runs alongside the Discord bot)
# ---------------------------------------------------------------------------

async def _handle_positions(request: web.Request) -> web.Response:
    """Return merged Alpaca + DB positions as JSON."""
    api_key = os.environ.get("RALLY_API_KEY")
    if api_key:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {api_key}":
            return web.json_response({"error": "unauthorized"}, status=401)
    state = await get_merged_positions()
    return web.Response(
        text=_json.dumps(state), content_type="application/json",
    )


async def _handle_health(request: web.Request) -> web.Response:
    """Simple health check."""
    return web.json_response({"status": "ok"})


async def start_api_server() -> None:
    """Start the aiohttp API server alongside the Discord bot."""
    app = web.Application()
    app.router.add_get("/api/positions", _handle_positions)
    app.router.add_get("/health", _handle_health)
    port = int(os.environ.get("PORT", 8080))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("API server listening on port %d", port)


# ---------------------------------------------------------------------------

def _ensure_caller(interaction: discord.Interaction) -> None:
    """Auto-register the calling Discord user."""
    ensure_user(interaction.user.id, str(interaction.user))


def _ensure_caller_from_message(message: discord.Message) -> None:
    """Auto-register user from message (not interaction)."""
    ensure_user(message.author.id, str(message.author))



class RallyBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info("Synced %d slash commands", len(self.tree.get_commands()))

    async def on_ready(self) -> None:
        logger.info("Bot connected as %s (ID: %s)", self.user, self.user.id)
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="rally signals",
            )
        )
        asyncio.create_task(start_api_server())


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
        """Run model retraining with live progress updates."""
        import queue
        import time

        from pipeline.retrain import retrain_all

        try:
            await channel.send(
                "🔄 **Starting model retraining...**\n"
                "This will take 10-30+ minutes depending on the number of tickers."
            )

            start_time = time.time()
            progress_queue: queue.Queue[tuple[int, int, int, int]] = queue.Queue()

            def on_progress(done: int, total: int, success: int, failed: int) -> None:
                # Overwrite queue with latest snapshot so we always report current state
                try:
                    progress_queue.get_nowait()
                except queue.Empty:
                    pass
                progress_queue.put((done, total, success, failed))

            async def send_updates() -> None:
                while True:
                    await asyncio.sleep(300)  # 5 minutes
                    elapsed_min = int((time.time() - start_time) / 60)
                    try:
                        done, total, success, failed = progress_queue.get_nowait()
                        pct = int(done / total * 100) if total else 0
                        await channel.send(
                            f"⏳ **Retraining... {elapsed_min}m elapsed**\n"
                            f"• Progress: {done}/{total} ({pct}%)\n"
                            f"• Success: {success} | Failed: {failed}"
                        )
                    except queue.Empty:
                        await channel.send(
                            f"⏳ **Retraining... {elapsed_min}m elapsed** (fetching data)"
                        )

            update_task = asyncio.create_task(send_updates())

            try:
                await asyncio.to_thread(retrain_all, tickers, False, on_progress)
            finally:
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass

            elapsed = time.time() - start_time
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)

            # Get final counts from last progress snapshot
            try:
                done, total, success, failed = progress_queue.get_nowait()
            except queue.Empty:
                manifest = load_manifest()
                success = len(manifest) if manifest else 0
                failed = 0
                total = success

            await channel.send(
                f"✅ **Retraining complete!**\n"
                f"• Trained: {success}/{total} models\n"
                f"• Failed: {failed}\n"
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

        async def _send_alert(embed: discord.Embed, msg_type: str = "other") -> None:
            """Send an embed to the alerts channel and log to DB."""
            from db.events import log_discord_message
            if not alert_channel_id:
                return
            channel = bot.get_channel(int(alert_channel_id))
            if channel:
                await channel.send(embed=embed)
            log_discord_message(msg_type, embed.title, (embed.description or "")[:500] or None)

        async def _send_error_alert(task_name: str, error: Exception) -> None:
            """Send an error embed to the alerts channel."""
            import traceback

            from .notify import _error_embed
            tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            details = f"```\n{tb[-3500:]}\n```"
            embed = discord.Embed.from_dict(_error_embed(f"{task_name} Failed", details))
            await _send_alert(embed, "error")

        def _save_watchlist(results: list, positions: dict) -> None:
            """Persist all scan results for agent queries."""
            from datetime import date as _date

            from db.positions import save_watchlist as _db_save_watchlist

            open_tickers = {
                p["ticker"] for p in positions.get("positions", [])
            }
            watchlist = []
            for r in sorted(
                results,
                key=lambda x: x.get("p_rally", 0),
                reverse=True,
            ):
                if r.get("status") != "ok":
                    continue
                if r["ticker"] in open_tickers:
                    continue
                p_rally = r.get("p_rally", 0)
                watchlist.append({
                    "ticker": r["ticker"],
                    "p_rally": round(p_rally * 100, 1),
                    "p_rally_raw": r.get("p_rally_raw", 0),
                    "comp_score": r.get("comp_score", 0),
                    "fail_dn": r.get("fail_dn", 0),
                    "trend": r.get("trend", 0),
                    "golden_cross": r.get("golden_cross", 0),
                    "hmm_compressed": r.get("hmm_compressed", 0),
                    "rv_pctile": r.get("rv_pctile", 0),
                    "atr_pct": r.get("atr_pct", 0),
                    "macd_hist": r.get("macd_hist", 0),
                    "vol_ratio": r.get("vol_ratio", 1),
                    "vix_pctile": r.get("vix_pctile", 0),
                    "rsi": r.get("rsi", 0),
                    "close": r.get("close", 0),
                    "size": r.get("size", 0),
                    "signal": bool(r.get("signal")),
                })

            _db_save_watchlist(watchlist, scan_date=_date.today())

        async def _run_scan() -> None:
            """Run the scan pipeline in a thread, then post results."""
            import time as _time

            from db.events import finish_scheduler_event, log_scheduler_event
            from pipeline.scanner import scan_all
            from db.portfolio import record_closed_trades, update_daily_snapshot
            from core.persistence import load_manifest
            from trading.positions import update_existing_positions

            from .notify import _exit_embed, _positions_embed, _signal_embed

            logger.info("Scheduler: starting daily scan")
            _scan_event_id = log_scheduler_event("scan")
            _t0 = _time.time()

            manifest = await asyncio.to_thread(load_manifest)
            if not manifest:
                duration = round(_time.time() - _t0, 1)
                finish_scheduler_event(_scan_event_id, "error", n_signals=0, n_exits=0,
                                       duration_s=duration)
                embed = discord.Embed(
                    title="⚠️ Daily Scan Skipped — No Trained Models",
                    description=(
                        "No trained models were found on the volume.\n"
                        "Run retrain before the next scan: ask me to **retrain** or trigger it manually."
                    ),
                    color=discord.Color.orange(),
                )
                await _send_alert(embed, "scan_error")
                return

            results = await asyncio.to_thread(
                scan_all, None, False, "conservative"
            )
            if not results:
                finish_scheduler_event(_scan_event_id, "success", n_signals=0, n_exits=0,
                                       duration_s=round(_time.time() - _t0, 1))
                return

            all_signals = [r for r in results if r.get("signal")]

            # Get positions WITH full metadata (trail_order_id, target_order_id) before
            # any exits are detected, so order IDs are available for OCO cancellation.
            from integrations.alpaca.executor import is_enabled as alpaca_enabled
            positions = await get_merged_positions()

            # Detect exits + update bars_held / trailing_stop for open positions.
            # When Alpaca is enabled we defer DB commits (commit_exits=False) so we can
            # confirm the broker order before removing the position from system_positions.
            # Without Alpaca there is no broker to confirm, so we commit immediately.
            positions = await asyncio.to_thread(
                update_existing_positions, positions, results,
                not alpaca_enabled(),  # commit_exits
            )
            closed = positions.get("closed_today", [])

            if closed and not alpaca_enabled():
                # Non-Alpaca path: exits already committed to DB, notify now.
                await _send_alert(discord.Embed.from_dict(_exit_embed(closed)), "exit")
                record_closed_trades(closed)

            update_daily_snapshot(positions, results)

            # Persist watchlist (near-signal tickers) for agent queries
            _save_watchlist(results, positions)

            # Replace current-state snapshot tables (latest scan only)
            from db.positions import save_latest_scan as _save_latest_scan
            await asyncio.to_thread(_save_latest_scan, results, positions)

            # Filter out signals for tickers we already hold
            open_tickers = {
                p["ticker"] for p in positions.get("positions", [])
            }
            signals = [
                s for s in all_signals if s["ticker"] not in open_tickers
            ]

            # Only alert on genuinely new signals
            if signals:
                await _send_alert(discord.Embed.from_dict(_signal_embed(signals)), "signal")

            # Always show open positions summary
            open_pos = positions.get("positions", [])
            await _send_alert(discord.Embed.from_dict(_positions_embed(open_pos)), "positions")

            # Auto-execute on Alpaca if enabled — only add positions after fill
            if alpaca_enabled():
                from db.events import log_order
                from trading.positions import add_signal_positions, process_signal_queue

                from integrations.alpaca.executor import (
                    execute_entries,
                    execute_exits,
                    get_account_equity,
                )
                from .notify import _cash_parking_embed, _order_embed, _order_failure_embed

                # External system boundary — broker outage must not break scan
                try:
                    equity = await get_account_equity()

                    # Cash parking: sell only enough SGOV to cover the capital gap
                    from integrations.alpaca.cash_parking import is_enabled as sgov_enabled
                    if sgov_enabled() and signals:
                        from integrations.alpaca.cash_parking import sell_sgov
                        from trading.positions import get_total_exposure
                        _available = PARAMS.max_portfolio_exposure - get_total_exposure()
                        _needed = sum(s.get("size", 0) for s in signals)
                        _gap = max(_needed - _available, 0.0)
                        sgov_sell = await sell_sgov(equity, _gap)
                        if sgov_sell and sgov_sell.success:
                            await _send_alert(
                                discord.Embed.from_dict(
                                    _cash_parking_embed("unpark", sgov_sell, equity, _gap)
                                ), "order"
                            )

                    if signals:
                        entry_results = await execute_entries(signals, equity=equity)
                        ok = [r for r in entry_results if r.success]
                        fail = [r for r in entry_results if not r.success and not r.skipped]

                        for r in entry_results:
                            log_order(r.ticker, "buy", "market", r.qty, "entry",
                                      r.order_id, "filled" if r.success else "failed",
                                      r.fill_price, r.error)

                        # Add only filled signals to DB
                        if ok:
                            size_map = {r.ticker: r.actual_size for r in ok if r.actual_size is not None}
                            filled_tickers = {r.ticker for r in ok}
                            filled_signals = [
                                {**s, "size": size_map.get(s["ticker"], s["size"])}
                                for s in signals if s["ticker"] in filled_tickers
                            ]
                            fresh_positions = load_positions()
                            add_signal_positions(fresh_positions, filled_signals)
                            await _store_order_ids(entry_results)
                            await _send_alert(
                                discord.Embed.from_dict(_order_embed(ok, equity)), "order"
                            )
                        if fail:
                            await _send_alert(
                                discord.Embed.from_dict(_order_failure_embed(fail)), "order_failure"
                            )
                    if closed:
                        exit_results = await execute_exits(closed)
                        ok = [r for r in exit_results if r.success]
                        fail = [r for r in exit_results if not r.success]

                        for r in exit_results:
                            log_order(r.ticker, "sell", "market", r.qty, "exit_scan",
                                      r.order_id, "filled" if r.success else "failed",
                                      r.fill_price, r.error)

                        # Commit exits to DB only after the broker confirms the close.
                        # Positions that failed stay in system_positions so they remain
                        # managed (stops, targets, time stop) on subsequent scans.
                        if ok:
                            from db.positions import (
                                delete_position_meta as _del_meta,
                                record_closed_position as _rec_closed,
                            )
                            ok_tickers = {r.ticker for r in ok}
                            confirmed_closed = [
                                p for p in closed if p["ticker"] in ok_tickers
                            ]
                            for pos in confirmed_closed:
                                _del_meta(pos["ticker"])
                                _rec_closed(pos)
                            await _send_alert(
                                discord.Embed.from_dict(_exit_embed(confirmed_closed)), "exit"
                            )
                            record_closed_trades(confirmed_closed)
                            await _send_alert(
                                discord.Embed.from_dict(_order_embed(ok, equity)), "order"
                            )
                        if fail:
                            await _send_alert(
                                discord.Embed.from_dict(_order_failure_embed(fail)), "order_failure"
                            )

                    # --- Signal queue: re-attempt queued signals freed by today's closes ---
                    queued = await asyncio.to_thread(process_signal_queue)
                    if queued:
                        logger.info("Processing %d queued signals", len(queued))
                        q_results = await execute_entries(queued, equity=equity)
                        q_ok = [r for r in q_results if r.success]
                        if q_ok:
                            q_size_map = {r.ticker: r.actual_size for r in q_ok if r.actual_size is not None}
                            filled_tickers = {r.ticker for r in q_ok}
                            filled_queued = [
                                {**s, "size": q_size_map.get(s["ticker"], s["size"])}
                                for s in queued if s["ticker"] in filled_tickers
                            ]
                            add_signal_positions(load_positions(), filled_queued)
                            await _store_order_ids(q_results)
                            await _send_alert(
                                discord.Embed.from_dict(_order_embed(q_ok, equity)), "order"
                            )

                    # Cash parking: park remaining idle capital in SGOV
                    if sgov_enabled():
                        from integrations.alpaca.cash_parking import buy_sgov
                        from trading.positions import get_total_exposure
                        idle = PARAMS.max_portfolio_exposure - get_total_exposure()
                        sgov_buy = await buy_sgov(equity, idle)
                        if sgov_buy and sgov_buy.success:
                            await _send_alert(
                                discord.Embed.from_dict(
                                    _cash_parking_embed("park", sgov_buy, equity, idle)
                                ), "order"
                            )

                except Exception as e:
                    logger.exception("Alpaca execution failed")
                    await _send_error_alert("Alpaca Execution", e)

            # --- Opportunity cost: update outcomes for past skipped signals ---
            from trading.positions import update_skipped_outcomes
            await asyncio.to_thread(update_skipped_outcomes, results)

            finish_scheduler_event(_scan_event_id, "success", n_signals=len(signals),
                                   n_exits=len(closed), duration_s=round(_time.time() - _t0, 1))
            logger.info(
                f"Scheduler: scan complete — {len(signals)} signals, "
                f"{len(closed)} exits"
            )

        async def _store_order_ids(results: list) -> None:
            """Save Alpaca order IDs + trailing stop IDs to positions.json."""
            from db.positions import load_positions
            from trading.positions import async_save_positions
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

            from db.events import finish_scheduler_event, log_scheduler_event
            from pipeline.retrain import retrain_all

            from .notify import _retrain_embed

            logger.info("Scheduler: starting weekly retrain")
            _retrain_event_id = log_scheduler_event("retrain")
            t0 = _time.time()
            await asyncio.to_thread(retrain_all)
            elapsed = _time.time() - t0

            from core.persistence import load_manifest

            manifest = load_manifest()
            health = {
                "total_count": len(manifest),
                "fresh_count": len(manifest),
                "stale_count": 0,
            }
            await _send_alert(
                discord.Embed.from_dict(_retrain_embed(health, elapsed)), "retrain"
            )
            finish_scheduler_event(_retrain_event_id, "success", duration_s=round(elapsed, 1))
            logger.info("Scheduler: retrain complete (%.0fs)", elapsed)

        # -- Price alert configuration --
        _alert_interval = int(os.environ.get("PRICE_ALERT_INTERVAL_MINUTES", "15"))
        _alert_proximity_pct = float(os.environ.get("ALERT_PROXIMITY_PCT", "1.5"))

        _ET = zoneinfo.ZoneInfo("America/New_York")

        def _is_market_open() -> bool:
            """True if Mon-Fri 9:30 AM - 4:00 PM ET."""
            now = datetime.now(_ET)
            if now.weekday() >= 5:
                return False
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            return market_open <= now <= market_close

        async def _execute_breach_exit(
            ticker: str, pos: dict, price: float, exit_reason: str,
        ) -> dict | None:
            """Cancel any open exit orders then execute a market sell. Returns order result dict or None."""
            from db.events import log_order
            from trading.positions import async_close_position

            from integrations.alpaca.executor import cancel_exit_orders, execute_exit

            # Cancel standing limit/stop orders before firing market sell
            await cancel_exit_orders(
                pos.get("target_order_id"), pos.get("trail_order_id"),
            )

            try:
                result = await execute_exit(ticker)
                fill = result.fill_price or price
                await async_close_position(ticker, fill, exit_reason)
                log_order(
                    ticker, "sell", "market", result.qty,
                    f"exit_{exit_reason}", result.order_id,
                    "filled" if result.success else "failed",
                    result.fill_price, result.error,
                )
                return result.model_dump()
            except Exception:
                logger.exception("Alpaca exit failed for %s", ticker)
                return None

        async def _check_price_alerts() -> None:
            """Fetch live prices for open positions and alert on breaches."""
            from core.data import fetch_quotes

            from integrations.alpaca.executor import is_enabled as alpaca_enabled
            from .notify import _approaching_alert_embed, _price_alert_embed

            if not _is_market_open():
                return

            state = await get_merged_positions()
            positions = state.get("positions", [])
            if not positions:
                return

            from db.events import log_price_alert

            today = datetime.now(_ET).strftime("%Y-%m-%d")

            tickers = [p["ticker"] for p in positions]

            if alpaca_enabled():
                from db.positions import load_positions as reload_positions
                from trading.positions import (
                    async_save_positions,
                    update_fill_prices,
                )

                from integrations.alpaca.executor import (
                    check_pending_fills,
                    get_snapshots,
                    place_exit_orders,
                )

                quotes = await get_snapshots(tickers)

                # Update entry prices for orders that filled since last check
                pending_ids = [p.get("order_id") for p in positions if p.get("order_id")]
                if pending_ids:
                    fills = await check_pending_fills(pending_ids)
                    if fills:
                        from db.events import update_order_fill
                        for oid, fp in fills.items():
                            update_order_fill(oid, fp)
                        n_filled = await update_fill_prices(fills)
                        if n_filled:
                            logger.info("Updated %d fill prices from Alpaca", n_filled)

                # Place exit orders for newly-filled entries that don't have them yet
                fresh_state = reload_positions()
                for pos in fresh_state.get("positions", []):
                    if (not pos.get("order_id")               # fill confirmed
                            and not pos.get("target_order_id")  # no limit sell yet
                            and not pos.get("trail_order_id")   # no stop sell yet
                            and pos.get("target_price") and pos.get("stop_price")
                            and pos.get("qty")):
                        effective_stop = max(
                            pos.get("stop_price", 0), pos.get("trailing_stop", 0),
                        )
                        t_oid, s_oid = await place_exit_orders(
                            pos["ticker"], pos["qty"],
                            pos["target_price"], effective_stop,
                        )
                        pos["target_order_id"] = t_oid
                        pos["trail_order_id"] = s_oid
                        if t_oid or s_oid:
                            from db.events import log_order
                            if t_oid:
                                log_order(pos["ticker"], "sell", "limit",
                                          pos["qty"], "exit_target", t_oid, "pending")
                            if s_oid:
                                log_order(pos["ticker"], "sell", "stop",
                                          pos["qty"], "exit_stop", s_oid, "pending")
                            logger.info(
                                "Exit orders placed for %s: target=%s stop=%s",
                                pos["ticker"], t_oid, s_oid,
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
                pnl_pct = round((price / entry - 1) * 100, 2) if entry else 0

                # Profit lock: raise hard stop floor intraday once lock level is touched
                if PARAMS.profit_lock_pct > 0 and entry:
                    lock_price = round(entry * (1 + PARAMS.profit_lock_pct), 4)
                    if price >= lock_price and stop < lock_price:
                        pos["stop_price"] = lock_price
                        stop = lock_price
                        await asyncio.to_thread(save_position_meta, pos)

                effective_stop = max(stop, trailing)

                # Check stop breach
                if effective_stop > 0 and price <= effective_stop:
                    if log_price_alert(today, ticker, "stop_breached", price, effective_stop, entry, pnl_pct):
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
                        if alpaca_enabled():
                            order_result = await _execute_breach_exit(ticker, pos, price, "stop")
                            if order_result:
                                alert["order_result"] = order_result
                        breach_alerts.append(alert)

                # Check target breach
                elif target > 0 and price >= target:
                    if log_price_alert(today, ticker, "target_breached", price, target, entry, pnl_pct):
                        alert = {
                            "ticker": ticker,
                            "alert_type": "target_breached",
                            "current_price": price,
                            "level_price": target,
                            "level_name": "Target",
                            "entry_price": entry,
                            "pnl_pct": pnl_pct,
                        }
                        if alpaca_enabled():
                            order_result = await _execute_breach_exit(ticker, pos, price, "profit_target")
                            if order_result:
                                alert["order_result"] = order_result
                        breach_alerts.append(alert)

                # Check approaching stop
                elif _alert_proximity_pct > 0 and effective_stop > 0:
                    distance = (price / effective_stop - 1) * 100
                    if 0 < distance <= _alert_proximity_pct:
                        if log_price_alert(today, ticker, "near_stop", price, effective_stop, entry, pnl_pct):
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
                        if log_price_alert(today, ticker, "near_target", price, target, entry, pnl_pct):
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
                    discord.Embed.from_dict(_price_alert_embed(breach_alerts)), "price_alert"
                )
            if approach_alerts:
                await _send_alert(
                    discord.Embed.from_dict(_approaching_alert_embed(approach_alerts)), "price_alert"
                )

            n = len(breach_alerts) + len(approach_alerts)
            if n:
                logger.info(
                    "Price alerts: %d breaches, %d warnings",
                    len(breach_alerts), len(approach_alerts),
                )


        async def _reconcile() -> None:
            """Check whether any broker exit orders (limit/stop) have filled and sync the DB."""
            from trading.positions import async_close_position

            from integrations.alpaca.executor import check_exit_fills
            from integrations.alpaca.executor import is_enabled as alpaca_enabled

            if not alpaca_enabled() or not _is_market_open():
                return

            state = await get_merged_positions()
            positions = state.get("positions", [])
            filled = await check_exit_fills(positions)
            for fill in filled:
                ticker = fill["ticker"]
                fill_price = fill["fill_price"]
                exit_reason = fill["exit_reason"]
                logger.info(
                    "Broker exit filled for %s at %.2f (%s) — syncing DB",
                    ticker, fill_price, exit_reason,
                )
                await async_close_position(ticker, fill_price, exit_reason)

        # -- Proactive state --
        _cached_regime_states: dict = {}
        _watchlist_tickers: list[str] = []
        _current_alert_interval = PARAMS.base_alert_interval
        _last_alert_check = datetime.min.replace(tzinfo=_ET)

        async def _check_regime_shifts() -> None:
            """Check HMM regime states and alert on transitions."""
            from trading.regime_monitor import check_regime_shifts, is_cascade

            from .notify import _regime_shift_embed

            if not PARAMS.regime_check_enabled or not _is_market_open():
                return

            transitions = await asyncio.to_thread(check_regime_shifts)
            if not transitions:
                return

            # Update cached regime states
            from trading.regime_monitor import get_regime_states
            nonlocal _cached_regime_states
            _cached_regime_states = get_regime_states()

            await _send_alert(
                discord.Embed.from_dict(_regime_shift_embed(transitions)), "regime_shift"
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
                ), "regime_shift")
                await _run_scan()

        async def _run_risk_evaluation() -> None:
            """Run proactive risk evaluation on current positions."""
            from trading.risk_manager import evaluate, execute_actions

            from .notify import _risk_action_embed

            if not PARAMS.proactive_risk_enabled:
                return

            state = await get_merged_positions()
            positions = state.get("positions", [])
            if not positions:
                return

            # Get equity
            try:
                from integrations.alpaca.executor import get_account_equity
                from integrations.alpaca.executor import is_enabled as alpaca_enabled
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
                    discord.Embed.from_dict(_risk_action_embed(meaningful)), "risk_action"
                )
                logger.info(
                    "Risk evaluation: %d actions taken (%s)",
                    len(meaningful),
                    ", ".join(f"{r['ticker']}:{r['action']}" for r in meaningful),
                )

        async def _run_midday_scan() -> None:
            """Run a lightweight scan on watchlist tickers only."""
            from pipeline.scanner import scan_watchlist

            from .notify import _signal_embed

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

            all_signals = [r for r in results if r.get("signal")]

            # Filter out tickers we already hold (Alpaca = source of truth)
            merged = await get_merged_positions()
            open_tickers = {
                p["ticker"]
                for p in merged.get("positions", [])
            }
            signals = [
                s for s in all_signals if s["ticker"] not in open_tickers
            ]

            if signals:
                embed = discord.Embed.from_dict(_signal_embed(signals))
                embed.title = f"Mid-day Signal ({len(signals)})"
                embed.set_footer(text="From watchlist mid-day scan")
                await _send_alert(embed, "signal")

        async def _should_use_fast_alerts() -> bool:
            """Check if conditions warrant faster alert frequency."""
            if not PARAMS.adaptive_alerts_enabled:
                return False

            state = await get_merged_positions()
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
                from trading.risk_manager import compute_drawdown

                from integrations.alpaca.executor import is_enabled as alpaca_enabled
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
                    fast = await _should_use_fast_alerts()
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

        # Morning scan: 9:00 AM ET — signals ready before market open
        # Alpaca DAY orders submitted pre-market queue and fill at 9:30 AM open.
        @tasks.loop(time=time(hour=9, minute=0, tzinfo=_ET))
        async def scheduled_morning_scan() -> None:
            nonlocal _watchlist_tickers
            weekday = datetime.utcnow().weekday()
            if weekday >= 5:  # skip weekends
                return
            if not PARAMS.morning_scan_enabled:
                return
            try:
                await _run_scan()

                from core.persistence import load_manifest
                manifest = load_manifest()
                if manifest:
                    _watchlist_tickers = sorted(manifest.keys())
            except Exception as e:
                logger.exception("Morning scan failed")
                await _send_error_alert("Morning Scan", e)

        # Daily scan: 4:30 PM ET
        @tasks.loop(time=time(hour=16, minute=30, tzinfo=_ET))
        async def scheduled_scan() -> None:
            nonlocal _watchlist_tickers
            weekday = datetime.utcnow().weekday()
            if weekday >= 5:  # skip weekends
                return
            try:
                await _run_scan()

                # Update watchlist for mid-day scans
                from core.persistence import load_manifest
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

        # Mid-day scans: 11 AM and 1 PM ET
        @tasks.loop(time=[time(hour=11, minute=0, tzinfo=_ET), time(hour=13, minute=0, tzinfo=_ET)])
        async def scheduled_midday_scan() -> None:
            weekday = datetime.utcnow().weekday()
            if weekday >= 5:
                return
            try:
                await _run_midday_scan()
            except Exception as e:
                logger.exception("Mid-day scan failed")
                await _send_error_alert("Mid-day Scan", e)

        # Reconciliation: every 15 minutes during market hours — sync DB when
        # broker exit orders (limit/stop) fill without our loop catching it first.
        @tasks.loop(minutes=15)
        async def scheduled_reconcile() -> None:
            try:
                await _reconcile()
            except Exception as e:
                logger.exception("Reconciliation failed")
                await _send_error_alert("Reconciliation", e)

        # Weekly retrain: Sunday 6 PM ET, followed by a scan so
        # Monday signals are ready before market open.
        @tasks.loop(time=time(hour=18, minute=0, tzinfo=_ET))
        async def scheduled_retrain() -> None:
            if datetime.utcnow().weekday() != 6:  # Sunday only
                return
            try:
                await _run_retrain()
                await _run_scan()
            except Exception as e:
                logger.exception("Scheduled retrain failed")
                await _send_error_alert("Weekly Retrain", e)

        @bot.event
        async def on_ready() -> None:
            logger.info("Bot connected as %s (ID: %s)", bot.user, bot.user.id)
            await bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="rally signals",
                )
            )
            if not scheduled_morning_scan.is_running():
                scheduled_morning_scan.start()
                logger.info("Scheduler: morning scan armed (9:00 AM ET)")
            if not scheduled_scan.is_running():
                scheduled_scan.start()
                logger.info("Scheduler: daily scan armed (4:30 PM ET)")
            if not scheduled_retrain.is_running():
                scheduled_retrain.start()
                logger.info("Scheduler: weekly retrain armed (Sun 6:00 PM ET)")
            if not scheduled_price_alerts.is_running():
                scheduled_price_alerts.start()
                logger.info(
                    "Scheduler: price alerts armed "
                    f"(adaptive: {PARAMS.base_alert_interval}m / "
                    f"{PARAMS.fast_alert_interval}m fast)"
                )
            if not scheduled_reconcile.is_running():
                scheduled_reconcile.start()
                logger.info("Scheduler: reconciliation armed (every 15m during market hours)")
            if not scheduled_regime_check.is_running():
                scheduled_regime_check.start()
                logger.info("Scheduler: regime check armed (every 30m during market hours)")
            if not scheduled_midday_scan.is_running():
                scheduled_midday_scan.start()
                logger.info("Scheduler: mid-day scans armed (11 AM, 1 PM ET)")

            # Populate watchlist from manifest so mid-day scans work immediately
            try:
                from core.persistence import load_manifest
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

            # No auto-scan on restart to avoid unintended entries
            if _is_market_open():
                logger.info(
                    "Market is open — skipping startup scan "
                    "(next scheduled scan at 4:30 PM ET; morning scan runs at 9:00 AM ET)"
                )

    return bot
