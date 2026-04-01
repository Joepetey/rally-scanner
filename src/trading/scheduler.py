"""
TradingScheduler — decoupled orchestration of all trading tasks.

Owns AlertEngine + AlpacaStreamManager. Emits typed events via callback.
Zero Discord imports.
"""

import asyncio
import logging
import time as _time
import zoneinfo
from collections.abc import Awaitable, Callable
from datetime import date as _date
from datetime import datetime

import sentry_sdk

from config import ASSETS, PARAMS
from core.data import fetch_quotes
from core.persistence import load_manifest
from db.events import finish_scheduler_event, log_order, log_scheduler_event
from db.portfolio import record_closed_trades, update_daily_snapshot
from db.positions import (
    delete_position_meta as _del_meta,
)
from db.positions import (
    get_recently_closed_tickers,
    load_current_signals,
    load_positions,
)
from db.positions import (
    load_position_meta as _load_position_meta,
)
from db.positions import (
    record_closed_position as _rec_closed,
)
from db.positions import (
    save_latest_scan as _save_latest_scan,
)
from db.positions import (
    save_position_meta as _save_position_meta,
)
from db.positions import (
    save_watchlist as _db_save_watchlist,
)
from integrations.alpaca.broker import get_all_positions
from integrations.alpaca.cash_parking import (
    _TICKER as PARKING_TICKER,
)
from integrations.alpaca.cash_parking import (
    buy_sgov,
    sell_sgov,
)
from integrations.alpaca.cash_parking import (
    is_enabled as sgov_enabled,
)
from integrations.alpaca.executor import (
    check_exit_fills,
    execute_entries,
    execute_exits,
    get_account_equity,
    get_snapshots,
)
from integrations.alpaca.executor import (
    is_enabled as alpaca_enabled,
)
from integrations.alpaca.stream import AlpacaStreamManager, is_stream_enabled
from pipeline.retrain import retrain_all
from pipeline.scanner import scan_all, scan_watchlist
from trading.engine import (
    AlertEngine,
    AlertEvent,
    ExitResult,
    HousekeepingResult,
    RegimeEvent,
    RetrainResult,
    RiskActionEvent,
    ScanResult,
    StreamDegradedEvent,
    StreamRecoveredEvent,
    WatchlistEvent,
)
from trading.positions import (
    add_signal_positions,
    async_close_position,
    async_save_positions,
    get_total_exposure,
    process_signal_queue,
    sync_positions_from_alpaca,
    update_existing_positions,
    update_position_for_price,
    update_skipped_outcomes,
)
from trading.regime_monitor import check_regime_shifts, get_regime_states, is_cascade
from trading.risk_manager import compute_drawdown, evaluate, execute_actions

_ET = zoneinfo.ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)

# All event types the scheduler can emit
TradingEvent = (
    AlertEvent | ExitResult | HousekeepingResult |
    ScanResult | WatchlistEvent | RegimeEvent | RetrainResult | RiskActionEvent |
    StreamDegradedEvent | StreamRecoveredEvent
)


class TradingScheduler:
    """Owns AlertEngine + AlpacaStreamManager, emits typed events via callback.

    Usage::

        scheduler = TradingScheduler(on_event=_handle_trading_event)
        await scheduler.start()   # in on_ready
        await scheduler.stop()    # in bot.close()
    """

    def __init__(
        self,
        on_event: Callable[[TradingEvent], Awaitable[None]],
    ) -> None:
        self._on_event = on_event
        self._engine = AlertEngine()
        self._stream: AlpacaStreamManager | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []

        # Shared state
        self._regime_states: dict = {}
        self._watchlist_tickers: list[str] = []
        self._current_alert_interval = PARAMS.base_alert_interval
        self._last_alert_check = datetime.min.replace(tzinfo=_ET)

        # Daily dedup: store ISO date string of last run
        self._ran_morning_scan: str = ""
        self._ran_morning_execute: str = ""
        self._ran_midday_1: str = ""
        self._ran_midday_2: str = ""
        self._ran_retrain: str = ""
        # Weekend crypto scan dedup: "{date}_{6h_slot}" (slot 0=midnight,1=6am,2=noon,3=6pm)
        self._ran_weekend_scan: str = ""

        # Pre-market scan results, consumed by market-open execution
        self._pending_signals: list[dict] = []
        self._pending_exits: list[dict] = []
        self._pending_scan_results: list[dict] = []
        self._pending_positions_embed: list[dict] = []

        # In-progress scan guard: prevent concurrent scans of the same type
        self._scan_in_progress: dict[str, bool] = {}

        # Concurrent exit guard: prevent double-exit for the same ticker
        self._exiting_tickers: set[str] = set()
        self._exit_lock: asyncio.Lock  # assigned in start()

        # Snapshot fetch guard: prevent housekeeping IEX fallback and polling
        # from calling get_snapshots concurrently when is_connected flips state
        self._snapshot_lock: asyncio.Lock  # assigned in start()

        # Housekeeping cycle counter for IEX coverage fallback (every 2 cycles ≈ 2 min)
        self._housekeeping_cycles: int = 0

        # Stream health monitoring: consecutive market-hours cycles where stream is disconnected
        self._stream_degraded_cycles: int = 0
        self._stream_alert_sent: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start stream (if enabled) and all housekeeping loops."""
        self._loop = asyncio.get_event_loop()
        # Locks must be created after the event loop is running to avoid
        # cross-loop errors if the scheduler is instantiated in a different
        # async context than where it's used (e.g. per-test event loops).
        self._exit_lock = asyncio.Lock()
        self._snapshot_lock = asyncio.Lock()

        # Start stream if Alpaca + streaming both enabled
        if alpaca_enabled() and is_stream_enabled():
            try:
                positions = load_positions()
                tickers = {p["ticker"] for p in positions.get("positions", [])}
                self._stream = AlpacaStreamManager(
                    on_trade=self._on_stream_trade,
                    excluded={PARKING_TICKER},
                )
                self._stream.start(tickers)
                logger.info("AlpacaStreamManager started")
            except Exception as e:
                logger.warning("Stream failed to start: %s — running in polling-only mode", e)
                self._stream = None

        # Populate watchlist from manifest so mid-day scans work immediately
        try:
            manifest = load_manifest()
            if manifest:
                self._watchlist_tickers = sorted(manifest.keys())
                logger.info(
                    "Loaded %d tickers into watchlist from manifest",
                    len(self._watchlist_tickers),
                )
        except Exception:
            logger.debug("No manifest found — watchlist empty until first scan")

        self._tasks = [
            asyncio.create_task(self._housekeeping_loop(), name="housekeeping"),
            asyncio.create_task(self._polling_loop(), name="polling"),
            asyncio.create_task(self._reconcile_loop(), name="reconcile"),
            asyncio.create_task(self._regime_loop(), name="regime"),
            asyncio.create_task(self._scan_loop(), name="scan"),
            asyncio.create_task(self._retrain_loop(), name="retrain"),
        ]
        for task in self._tasks:
            task.add_done_callback(self._task_sentinel)
        logger.info(
            "TradingScheduler started (%d loops, stream=%s)",
            len(self._tasks),
            self._stream is not None,
        )

    async def stop(self) -> None:
        """Stop stream and cancel all background tasks."""
        if self._stream:
            self._stream.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("TradingScheduler stopped")

    # ------------------------------------------------------------------
    # Public actions (callable from slash commands / agent)
    # ------------------------------------------------------------------

    async def run_daily_scan(
        self, scan_type: str = "daily", tickers: list[str] | None = None,
    ) -> None:
        """Run the full scan pipeline and emit a ScanResult event.

        Args:
            scan_type: label for the scan (daily, morning, weekend_crypto, etc.)
            tickers: optional list of tickers to restrict the scan to; None = all.
        """
        logger.info("Scheduler: starting %s scan", scan_type)
        _event_id = log_scheduler_event("scan")
        t0 = _time.time()

        manifest = load_manifest()
        if not manifest:
            finish_scheduler_event(_event_id, "error", n_signals=0, n_exits=0, duration_s=0)
            await self._on_event(ScanResult(
                signals=[], exits=[], orders=[],
                positions_summary={},
                scan_type=scan_type,
                error="No trained models found",
            ))
            return

        results = await asyncio.to_thread(scan_all, tickers, False, "conservative")
        if not results:
            finish_scheduler_event(_event_id, "success", n_signals=0, n_exits=0,
                                   duration_s=round(_time.time() - t0, 1))
            return

        all_signals = [r for r in results if r.get("signal")]

        if alpaca_enabled():
            try:
                _equity = await get_account_equity()
                await sync_positions_from_alpaca(equity=_equity)

            except Exception:
                logger.exception("Alpaca sync failed — using cached positions")
        positions = load_positions()

        # Detect exits; defer DB commit if Alpaca will confirm them
        positions = await asyncio.to_thread(
            update_existing_positions, positions, results, not alpaca_enabled(),
        )
        closed = positions.get("closed_today", [])

        if closed and not alpaca_enabled():
            record_closed_trades(closed)

        update_daily_snapshot(positions, results)
        self._save_watchlist(results, positions)
        await asyncio.to_thread(_save_latest_scan, results, positions)

        # Refresh watchlist for mid-day scans
        if manifest:
            self._watchlist_tickers = sorted(manifest.keys())

        open_tickers = {p["ticker"] for p in positions.get("positions", [])}
        cooldown_tickers: set[str] = set()
        if PARAMS.cooldown_days > 0:
            cooldown_tickers = get_recently_closed_tickers(PARAMS.cooldown_days)
            cooled = [s["ticker"] for s in all_signals
                      if s["ticker"] not in open_tickers and s["ticker"] in cooldown_tickers]
            if cooled:
                logger.info("Cooldown filter (%dd): skipping %s", PARAMS.cooldown_days, cooled)
        signals = [s for s in all_signals
                   if s["ticker"] not in open_tickers and s["ticker"] not in cooldown_tickers]

        # Snapshot positions BEFORE entries so the embed reflects existing
        # open positions only — new entries are shown in the orders/signals section.
        positions_for_embed = list(positions.get("positions", []))

        orders: list[dict] = []
        confirmed_exits: list[dict] = list(closed) if not alpaca_enabled() else []
        scan_equity: float = 0.0

        if alpaca_enabled():
            try:
                equity = await get_account_equity()
                scan_equity = equity

                # Sell SGOV to cover capital gap for new entries
                if sgov_enabled() and signals:
                    _available = PARAMS.max_portfolio_exposure - get_total_exposure()
                    _needed = sum(s.get("size", 0) for s in signals)
                    _gap = max(_needed - _available, 0.0)
                    sgov_sell = await sell_sgov(equity, _gap)
                    if sgov_sell and sgov_sell.success:
                        orders.append({"type": "sgov_sell", "result": sgov_sell.model_dump()})

                if signals:
                    entry_results = await self._execute_and_log_entries(signals, equity)
                    orders.extend([r.model_dump() for r in entry_results])

                if closed:
                    exit_results = await execute_exits(closed)
                    ok_exit = [r for r in exit_results if r.success]
                    for r in exit_results:
                        log_order(r.ticker, "sell", "market", r.qty, "exit_scan",
                                  r.order_id, "filled" if r.success else "failed",
                                  r.fill_price, r.error)
                    if ok_exit:
                        ok_tickers = {r.ticker for r in ok_exit}
                        # Confirm at the broker level before deleting DB records.
                        # success=True only means the close_position() API call succeeded;
                        # if an OCO leg still held the shares, the position may still be
                        # open at Alpaca. Ghost positions would be re-inserted by
                        # sync_positions_from_alpaca() with wrong risk params.
                        broker_positions = await get_all_positions()
                        still_open = {p["ticker"] for p in broker_positions}
                        ghost_tickers = ok_tickers & still_open
                        if ghost_tickers:
                            logger.warning(
                                "Exit API success but position still open at broker "
                                "— skipping DB delete to avoid ghost position: %s",
                                ghost_tickers,
                            )
                        confirmed_exits = [
                            p for p in closed
                            if p["ticker"] in ok_tickers and p["ticker"] not in still_open
                        ]
                        for pos in confirmed_exits:
                            _del_meta(pos["ticker"])
                            _rec_closed(pos)
                        record_closed_trades(confirmed_exits)

                    orders.extend([r.model_dump() for r in exit_results])

                # Re-attempt queued signals freed by today's closes
                q_results = await self._execute_queued_entries(equity)
                orders.extend([r.model_dump() for r in q_results])

                # Park idle capital in SGOV
                if sgov_enabled():
                    idle = PARAMS.max_portfolio_exposure - get_total_exposure()
                    sgov_buy = await buy_sgov(equity, idle)
                    if sgov_buy and sgov_buy.success:
                        orders.append({"type": "sgov_buy", "result": sgov_buy.model_dump()})

            except Exception:
                logger.exception("Alpaca execution failed during %s scan", scan_type)

        await asyncio.to_thread(update_skipped_outcomes, results)

        duration = round(_time.time() - t0, 1)
        finish_scheduler_event(_event_id, "success", n_signals=len(signals),
                               n_exits=len(confirmed_exits), duration_s=duration)
        logger.info(
            "Scan complete — %d signals, %d exits (%.0fs)",
            len(signals), len(confirmed_exits), duration,
        )

        await self._on_event(ScanResult(
            signals=signals,
            exits=confirmed_exits,
            orders=orders,
            positions_summary={"positions": positions_for_embed},
            scan_type=scan_type,
            equity=scan_equity,
        ))

        # Post-scan risk evaluation picks up any regime changes
        await self.run_risk_evaluation()

    async def run_premarket_scan(self) -> None:
        """Run full-universe scan at 9:00 AM ET and store signals for market-open execution."""
        logger.info("Scheduler: starting pre-market scan")
        _event_id = log_scheduler_event("scan")
        t0 = _time.time()

        manifest = load_manifest()
        if not manifest:
            finish_scheduler_event(_event_id, "error", n_signals=0, n_exits=0, duration_s=0)
            await self._on_event(ScanResult(
                signals=[], exits=[], orders=[],
                positions_summary={},
                scan_type="premarket",
                error="No trained models found",
            ))
            return

        results = await asyncio.to_thread(scan_all, None, False, "conservative")
        if not results:
            finish_scheduler_event(_event_id, "success", n_signals=0, n_exits=0,
                                   duration_s=round(_time.time() - t0, 1))
            return

        all_signals = [r for r in results if r.get("signal")]

        if alpaca_enabled():
            try:
                _equity = await get_account_equity()
                await sync_positions_from_alpaca(equity=_equity)
            except Exception:
                logger.exception("Alpaca sync failed — using cached positions")
        positions = load_positions()

        # Detect exits; defer DB commit — execution happens at 9:30
        positions = await asyncio.to_thread(
            update_existing_positions, positions, results, not alpaca_enabled(),
        )
        closed = positions.get("closed_today", [])

        if closed and not alpaca_enabled():
            record_closed_trades(closed)

        update_daily_snapshot(positions, results)
        self._save_watchlist(results, positions)
        await asyncio.to_thread(_save_latest_scan, results, positions)

        if manifest:
            self._watchlist_tickers = sorted(manifest.keys())

        open_tickers = {p["ticker"] for p in positions.get("positions", [])}
        cooldown_tickers: set[str] = set()
        if PARAMS.cooldown_days > 0:
            cooldown_tickers = get_recently_closed_tickers(PARAMS.cooldown_days)
            cooled = [s["ticker"] for s in all_signals
                      if s["ticker"] not in open_tickers and s["ticker"] in cooldown_tickers]
            if cooled:
                logger.info("Cooldown filter (%dd): skipping %s", PARAMS.cooldown_days, cooled)
        signals = [s for s in all_signals
                   if s["ticker"] not in open_tickers and s["ticker"] not in cooldown_tickers]

        positions_for_embed = list(positions.get("positions", []))

        # Store for market-open execution
        self._pending_signals = signals
        self._pending_exits = list(closed) if alpaca_enabled() else []
        self._pending_scan_results = results
        self._pending_positions_embed = positions_for_embed

        duration = round(_time.time() - t0, 1)
        confirmed_exits = list(closed) if not alpaca_enabled() else []
        finish_scheduler_event(_event_id, "success", n_signals=len(signals),
                               n_exits=len(confirmed_exits), duration_s=duration)
        logger.info(
            "Pre-market scan complete — %d signals, %d pending exits (%.0fs)",
            len(signals), len(self._pending_exits), duration,
        )

        await self._on_event(ScanResult(
            signals=signals,
            exits=confirmed_exits,
            orders=[],
            positions_summary={"positions": positions_for_embed},
            scan_type="premarket",
        ))

    async def run_market_open_execute(self) -> None:
        """Execute pending signals and exits at market open (9:30 AM ET)."""
        # Restart fallback: reload from DB if in-memory state was lost
        if not self._pending_signals and not self._pending_exits:
            if self._ran_morning_scan == _date.today().isoformat():
                db_signals = load_current_signals()
                if db_signals:
                    logger.info(
                        "Restart fallback: loaded %d signals from current_signals table",
                        len(db_signals),
                    )
                    self._pending_signals = db_signals

        if not self._pending_signals and not self._pending_exits:
            logger.info("Market-open execute: nothing pending, skipping")
            return

        logger.info(
            "Scheduler: executing %d entries + %d exits at market open",
            len(self._pending_signals), len(self._pending_exits),
        )
        _event_id = log_scheduler_event("execute")
        t0 = _time.time()

        signals = self._pending_signals
        closed = self._pending_exits
        results = self._pending_scan_results
        positions_for_embed = self._pending_positions_embed

        orders: list[dict] = []
        confirmed_exits: list[dict] = list(closed) if not alpaca_enabled() else []
        scan_equity: float = 0.0

        if alpaca_enabled():
            try:
                # Re-sync in case of pre-market changes
                equity = await get_account_equity()
                scan_equity = equity
                await sync_positions_from_alpaca(equity=equity)

                # Re-filter against current open positions (safety)
                open_tickers = {
                    p["ticker"] for p in load_positions().get("positions", [])
                }
                signals = [s for s in signals if s["ticker"] not in open_tickers]

                # Sell SGOV to cover capital gap for new entries
                if sgov_enabled() and signals:
                    _available = PARAMS.max_portfolio_exposure - get_total_exposure()
                    _needed = sum(s.get("size", 0) for s in signals)
                    _gap = max(_needed - _available, 0.0)
                    sgov_sell = await sell_sgov(equity, _gap)
                    if sgov_sell and sgov_sell.success:
                        orders.append({"type": "sgov_sell", "result": sgov_sell.model_dump()})

                if signals:
                    entry_results = await self._execute_and_log_entries(signals, equity)
                    orders.extend([r.model_dump() for r in entry_results])

                if closed:
                    exit_results = await execute_exits(closed)
                    ok_exit = [r for r in exit_results if r.success]
                    for r in exit_results:
                        log_order(r.ticker, "sell", "market", r.qty, "exit_scan",
                                  r.order_id, "filled" if r.success else "failed",
                                  r.fill_price, r.error)
                    if ok_exit:
                        ok_tickers = {r.ticker for r in ok_exit}
                        broker_positions = await get_all_positions()
                        still_open = {p["ticker"] for p in broker_positions}
                        ghost_tickers = ok_tickers & still_open
                        if ghost_tickers:
                            logger.warning(
                                "Exit API success but position still open at broker "
                                "— skipping DB delete to avoid ghost position: %s",
                                ghost_tickers,
                            )
                        confirmed_exits = [
                            p for p in closed
                            if p["ticker"] in ok_tickers and p["ticker"] not in still_open
                        ]
                        for pos in confirmed_exits:
                            _del_meta(pos["ticker"])
                            _rec_closed(pos)
                        record_closed_trades(confirmed_exits)

                    orders.extend([r.model_dump() for r in exit_results])

                # Re-attempt queued signals freed by today's closes
                q_results = await self._execute_queued_entries(equity)
                orders.extend([r.model_dump() for r in q_results])

                # Park idle capital in SGOV
                if sgov_enabled():
                    idle = PARAMS.max_portfolio_exposure - get_total_exposure()
                    sgov_buy = await buy_sgov(equity, idle)
                    if sgov_buy and sgov_buy.success:
                        orders.append({"type": "sgov_buy", "result": sgov_buy.model_dump()})

            except Exception:
                logger.exception("Alpaca execution failed during market-open execute")

        if results:
            await asyncio.to_thread(update_skipped_outcomes, results)

        duration = round(_time.time() - t0, 1)
        finish_scheduler_event(_event_id, "success", n_signals=len(signals),
                               n_exits=len(confirmed_exits), duration_s=duration)
        logger.info(
            "Market-open execute complete — %d signals, %d exits (%.0fs)",
            len(signals), len(confirmed_exits), duration,
        )

        await self._on_event(ScanResult(
            signals=signals,
            exits=confirmed_exits,
            orders=orders,
            positions_summary={"positions": positions_for_embed},
            scan_type="morning",
            equity=scan_equity,
        ))

        # Clear pending state
        self._pending_signals = []
        self._pending_exits = []
        self._pending_scan_results = []
        self._pending_positions_embed = []

        await self.run_risk_evaluation()

    async def run_midday_scan(self) -> None:
        """Run a lightweight scan on watchlist tickers only."""
        if not PARAMS.midday_scans_enabled or not self._engine.is_market_open():
            return
        if not self._watchlist_tickers:
            logger.debug("Mid-day scan: empty watchlist, skipping")
            return

        logger.info("Mid-day scan: checking %d watchlist tickers", len(self._watchlist_tickers))
        results = await asyncio.to_thread(scan_watchlist, self._watchlist_tickers)
        all_signals = [r for r in results if r.get("signal")]

        open_tickers = {p["ticker"] for p in load_positions().get("positions", [])}
        signals = [s for s in all_signals if s["ticker"] not in open_tickers]

        await self._on_event(WatchlistEvent(signals=signals, scan_type="midday"))

    async def run_regime_check(self) -> None:
        """Check HMM regime states and emit a RegimeEvent on transitions."""
        if not PARAMS.regime_check_enabled or not self._engine.is_market_open():
            return

        transitions = await asyncio.to_thread(check_regime_shifts)
        if not transitions:
            return

        self._regime_states = get_regime_states()
        cascade = is_cascade(transitions)

        await self._on_event(RegimeEvent(transitions=transitions, cascade_triggered=cascade))

        if cascade:
            logger.info(
                "Regime cascade detected (%d shifts) — triggering early scan",
                len(transitions),
            )
            await self.run_daily_scan("cascade")

    async def run_retrain(self) -> None:
        """Run model retraining and emit a RetrainResult event, then scan."""
        logger.info("Scheduler: starting weekly retrain")
        _event_id = log_scheduler_event("retrain")
        t0 = _time.time()
        await asyncio.to_thread(retrain_all)
        elapsed = _time.time() - t0

        manifest = load_manifest()
        finish_scheduler_event(_event_id, "success", duration_s=round(elapsed, 1))
        logger.info("Scheduler: retrain complete (%.0fs)", elapsed)

        await self._on_event(RetrainResult(
            tickers_retrained=sorted(manifest.keys()) if manifest else [],
            duration_seconds=round(elapsed, 1),
            manifest_size=len(manifest) if manifest else 0,
        ))

        # Scan so Monday signals are ready before market open
        await self.run_daily_scan("post_retrain")

    async def run_risk_evaluation(self) -> None:
        """Run proactive risk evaluation on current positions."""
        if not PARAMS.proactive_risk_enabled:
            return

        state = load_positions()
        positions = state.get("positions", [])
        if not positions:
            return

        try:
            equity = await get_account_equity() if alpaca_enabled() else 100_000
        except Exception:
            logger.warning("Equity fetch failed, defaulting to $100k", exc_info=True)
            equity = 100_000

        actions = await asyncio.to_thread(evaluate, equity, positions, self._regime_states)
        if not actions:
            return

        results = await execute_actions(actions, positions)
        meaningful = [r for r in results if not r.get("skipped")]
        if meaningful:
            await self._on_event(RiskActionEvent(actions=meaningful))
            logger.info(
                "Risk evaluation: %d actions (%s)",
                len(meaningful),
                ", ".join(f"{r['ticker']}:{r['action']}" for r in meaningful),
            )

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _housekeeping_loop(self) -> None:
        """1-minute timer: fill checks, OCO placement, position sync.

        Also handles stale-price warnings and IEX coverage fallback every
        2 cycles (~2 min) when the stream is connected but a ticker has had
        no trade event.
        """
        while True:
            await asyncio.sleep(60)
            if not self._engine.is_market_open() and not self._has_open_crypto_positions():
                continue
            try:
                self._housekeeping_cycles += 1
                state = load_positions()
                positions = state.get("positions", [])
                result = await self._engine.run_housekeeping(positions)

                if result.fills_confirmed or result.orders_placed:
                    await self._on_event(result)
                    if self._stream:
                        all_pos = load_positions().get("positions", [])
                        self._stream.update_subscriptions({p["ticker"] for p in all_pos})

                # Stale price check + IEX REST fallback every 2 housekeeping cycles
                if (
                    self._stream and self._stream.is_connected
                    and self._housekeeping_cycles % 2 == 0
                ):
                    new_stale, known_stale, never_traded = self._stream.get_stale_tickers(
                        stale_seconds=300.0,
                    )
                    stale = new_stale + known_stale + never_traded
                    if new_stale:
                        logger.warning(
                            "No stream trade in 5 min for %d ticker(s): %s — "
                            "possible subscription issue",
                            len(new_stale), sorted(new_stale),
                        )
                    if never_traded:
                        logger.info(
                            "No stream trade yet for %d ticker(s): %s — "
                            "likely low-volume",
                            len(never_traded), sorted(never_traded),
                        )
                    if known_stale:
                        logger.debug(
                            "Still stale (low-volume expected): %s",
                            sorted(known_stale),
                        )
                    # IEX fallback: evaluate stale tickers via REST snapshot
                    if stale and alpaca_enabled():
                        try:
                            async with self._snapshot_lock:
                                snapshots = await get_snapshots(stale)
                            fresh_state = load_positions()
                            fresh_positions = fresh_state.get("positions", [])
                            events = await self._engine.check_prices(
                                [p for p in fresh_positions if p["ticker"] in snapshots],
                                snapshots,
                            )
                            for event in events:
                                await self._on_event(event)
                                if event.alert_type in ("stop_breached", "target_breached"):
                                    pos = next(
                                        (p for p in fresh_positions if p["ticker"] == event.ticker),
                                        None,
                                    )
                                    if pos:
                                        await self._execute_breach_guarded(
                                            event.ticker, pos,
                                            event.current_price, event.alert_type,
                                        )
                        except Exception:
                            logger.exception("IEX fallback evaluation failed")

                # Stream degradation monitoring: alert after 5 consecutive disconnected cycles
                # (~5 min)
                _DEGRADE_THRESHOLD = 5
                if self._stream:
                    if not self._stream.is_connected:
                        self._stream_degraded_cycles += 1
                        if (
                            self._stream_degraded_cycles >= _DEGRADE_THRESHOLD
                            and not self._stream_alert_sent
                        ):
                            self._stream_alert_sent = True
                            logger.warning(
                                "Stream has been disconnected for %d min"
                                " — emitting degradation alert",
                                self._stream_degraded_cycles,
                            )
                            await self._on_event(StreamDegradedEvent(
                                disconnected_minutes=self._stream_degraded_cycles,
                            ))
                    else:
                        if self._stream_alert_sent:
                            logger.info(
                                "Stream reconnected after %d min — emitting recovery alert",
                                self._stream_degraded_cycles,
                            )
                            await self._on_event(StreamRecoveredEvent(
                                downtime_minutes=self._stream_degraded_cycles,
                            ))
                            self._stream_alert_sent = False
                        self._stream_degraded_cycles = 0

            except Exception:
                logger.exception("Housekeeping loop error")

    async def _polling_loop(self) -> None:
        """Price alert polling — active when stream is disconnected or disabled."""
        while True:
            await asyncio.sleep(60)
            if not self._engine.is_market_open() and not self._has_open_crypto_positions():
                continue
            # Skip polling while stream is connected
            if self._stream and self._stream.is_connected:
                continue
            try:
                now = datetime.now(_ET)
                elapsed = (now - self._last_alert_check).total_seconds() / 60
                if elapsed < self._current_alert_interval:
                    continue
                self._last_alert_check = now

                state = load_positions()
                positions = state.get("positions", [])
                if not positions:
                    continue

                tickers = [p["ticker"] for p in positions]
                if alpaca_enabled():
                    async with self._snapshot_lock:
                        quotes = await get_snapshots(tickers)
                else:
                    quotes = await asyncio.to_thread(fetch_quotes, tickers)

                events = await self._engine.check_prices(positions, quotes)
                for event in events:
                    await self._on_event(event)
                    if (
                        event.alert_type in ("stop_breached", "target_breached")
                        and alpaca_enabled()
                    ):
                        # Reload from DB so execute_breach sees current
                        # exit order IDs (MIC-102).
                        pos = _load_position_meta(event.ticker)
                        if not pos:
                            pos = next((p for p in positions if p["ticker"] == event.ticker), None)
                        if pos:
                            await self._execute_breach_guarded(
                                event.ticker, pos, event.current_price, event.alert_type,
                            )

                # Update adaptive interval
                if PARAMS.adaptive_alerts_enabled:
                    fast = await self._should_use_fast_alerts(positions, quotes)
                    self._current_alert_interval = (
                        PARAMS.fast_alert_interval if fast else PARAMS.base_alert_interval
                    )
            except Exception:
                logger.exception("Polling loop error")

    async def _reconcile_loop(self) -> None:
        """15-minute timer: check broker-side OCO fills and sync DB."""
        while True:
            await asyncio.sleep(900)
            if not alpaca_enabled() or not self._engine.is_market_open():
                continue
            try:
                state = load_positions()
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
                    pos = await async_close_position(ticker, fill_price, exit_reason)
                    if pos:

                        record_closed_trades([pos])
                        await self._on_event(ExitResult(
                            ticker=ticker,
                            exit_reason=exit_reason,
                            fill_price=fill_price,
                            realized_pnl_pct=pos.get("realized_pnl_pct"),
                            bars_held=pos.get("bars_held"),
                        ))
                        if self._stream:
                            all_pos = load_positions().get("positions", [])
                            self._stream.update_subscriptions({p["ticker"] for p in all_pos})
            except Exception:
                logger.exception("Reconciliation loop error")

    async def _regime_loop(self) -> None:
        """30-minute timer: HMM regime check."""
        while True:
            await asyncio.sleep(1800)
            try:
                await self.run_regime_check()
            except Exception:
                logger.exception("Regime loop error")

    async def _scan_loop(self) -> None:
        """Time-based scan loop.

        Weekdays: pre-market scan (9:00 AM), market-open execute (9:30 AM),
                  midday watchlist scans (11 AM, 1 PM) ET.
        Weekends: crypto-only scan every 6 hours (0, 6, 12, 18 ET).
        """
        while True:
            await asyncio.sleep(30)
            now = datetime.now(_ET)
            today = now.date().isoformat()

            if now.weekday() >= 5:
                # Weekend: crypto-only scan every 6 hours
                try:
                    await self._maybe_run_weekend_crypto_scan(now, today)
                except Exception:
                    logger.exception("Weekend crypto scan error")
                continue

            # Weekday scans
            try:
                # 9:00 AM ET — pre-market scan (full universe)
                if PARAMS.morning_scan_enabled and now.hour == 9 and now.minute < 5:
                    if (
                        self._ran_morning_scan != today
                        and not self._scan_in_progress.get("morning")
                    ):
                        self._ran_morning_scan = today
                        self._scan_in_progress["morning"] = True
                        try:
                            async with asyncio.timeout(600):
                                await self.run_premarket_scan()
                        finally:
                            self._scan_in_progress["morning"] = False
                # 9:30 AM ET — execute entries/exits at market open
                elif PARAMS.morning_scan_enabled and now.hour == 9 and 30 <= now.minute < 35:
                    if (
                        self._ran_morning_execute != today
                        and not self._scan_in_progress.get("execute")
                    ):
                        self._ran_morning_execute = today
                        self._scan_in_progress["execute"] = True
                        try:
                            async with asyncio.timeout(300):
                                await self.run_market_open_execute()
                        finally:
                            self._scan_in_progress["execute"] = False
                elif PARAMS.midday_scans_enabled and now.hour == 11 and now.minute < 5:
                    if self._ran_midday_1 != today and not self._scan_in_progress.get("midday_1"):
                        self._ran_midday_1 = today
                        self._scan_in_progress["midday_1"] = True
                        try:
                            async with asyncio.timeout(600):
                                await self.run_midday_scan()
                        finally:
                            self._scan_in_progress["midday_1"] = False
                elif PARAMS.midday_scans_enabled and now.hour == 13 and now.minute < 5:
                    if self._ran_midday_2 != today and not self._scan_in_progress.get("midday_2"):
                        self._ran_midday_2 = today
                        self._scan_in_progress["midday_2"] = True
                        try:
                            async with asyncio.timeout(600):
                                await self.run_midday_scan()
                        finally:
                            self._scan_in_progress["midday_2"] = False
            except Exception:
                logger.exception("Scan loop error")

    async def _maybe_run_weekend_crypto_scan(self, now: datetime, today: str) -> None:
        """Run a crypto-only scan every 6 hours on weekends."""
        from config import ASSETS

        # Fire at hours 0, 6, 12, 18 ET within a 5-minute window
        if now.hour % 6 != 0 or now.minute >= 5:
            return

        run_key = f"{today}_{now.hour}"
        if self._ran_weekend_scan == run_key:
            return

        crypto_tickers = [t for t, cfg in ASSETS.items() if cfg.asset_class == "crypto"]
        if not crypto_tickers:
            return

        # Only run if at least one crypto ticker has a trained model
        manifest = load_manifest()
        if not manifest or not any(t in manifest for t in crypto_tickers):
            logger.debug("Weekend crypto scan: no trained crypto models found, skipping")
            return

        if self._scan_in_progress.get("weekend_crypto"):
            return
        self._ran_weekend_scan = run_key
        logger.info(
            "Weekend crypto scan: %s (tickers=%s)",
            run_key, sorted(crypto_tickers),
        )
        self._scan_in_progress["weekend_crypto"] = True
        try:
            async with asyncio.timeout(600):
                await self.run_daily_scan("weekend_crypto", tickers=crypto_tickers)
        finally:
            self._scan_in_progress["weekend_crypto"] = False

    async def _retrain_loop(self) -> None:
        """Time-based: weekly retrain Sunday >= 6 PM ET.

        Uses catch-up semantics: if the container restarts after 18:00,
        the retrain fires on the first loop iteration instead of being
        silently skipped until next week.
        """
        while True:
            await asyncio.sleep(60)
            now = datetime.now(_ET)
            if now.weekday() != 6:  # Sunday only
                continue
            today = now.date().isoformat()
            if now.hour >= 18 and self._ran_retrain != today:
                try:
                    self._ran_retrain = today
                    await self.run_retrain()
                except Exception:
                    logger.exception("Retrain loop error")

    # ------------------------------------------------------------------
    # Stream callback (sync — called from stream daemon thread)
    # ------------------------------------------------------------------

    def _on_stream_trade(self, ticker: str, price: float) -> None:
        """Called from stream thread per throttled trade event.

        Evaluates the ticker synchronously then dispatches async work
        back to the main event loop via run_coroutine_threadsafe.
        """
        if not self._loop:
            return
        is_crypto = bool(ASSETS.get(ticker) and ASSETS[ticker].asset_class == "crypto")
        if not is_crypto and not self._engine.is_market_open():
            return

        pos = _load_position_meta(ticker)
        if not pos:
            return

        if update_position_for_price(pos, price):
            _save_position_meta(pos)

        event = self._engine.evaluate_single_ticker(ticker, price, pos)
        if event:
            asyncio.run_coroutine_threadsafe(
                self._handle_stream_alert(event, pos, price),
                self._loop,
            )

    async def _handle_stream_alert(
        self, event: AlertEvent, pos: dict, price: float,
    ) -> None:
        """Process a stream-triggered alert on the main event loop."""
        await self._on_event(event)

        if event.alert_type in ("stop_breached", "target_breached") and alpaca_enabled():
            # Reload from DB so execute_breach sees current exit order IDs
            # (pos was loaded on the stream thread and may be stale by now).
            fresh = _load_position_meta(event.ticker)
            if fresh is not None:
                pos = fresh
            await self._execute_breach_guarded(event.ticker, pos, price, event.alert_type)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _task_sentinel(self, task: asyncio.Task) -> None:
        """done_callback: log critical if a background task dies unexpectedly.

        Each loop has an inner except Exception that should catch everything
        normal, but a BaseException (e.g. SystemExit, KeyboardInterrupt) or an
        upstream bug can escape and kill the task silently. This fires once,
        after the task is done, and makes the failure visible in logs + Sentry.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        logger.critical(
            "Background task '%s' died unexpectedly — scheduler loop is no longer running: %s",
            task.get_name(), exc,
            exc_info=exc,
        )
        sentry_sdk.capture_exception(exc)

    async def _execute_breach_guarded(
        self, ticker: str, pos: dict, price: float, alert_type: str,
    ) -> None:
        """Execute a breach exit with concurrent-exit guard.

        Prevents race between stream-triggered exit and reconciliation-triggered
        exit for the same ticker running simultaneously.
        """
        async with self._exit_lock:
            if ticker in self._exiting_tickers:
                logger.info("Exit already in progress for %s — skipping duplicate", ticker)
                return
            self._exiting_tickers.add(ticker)

        try:
            reason = alert_type.replace("_breached", "")
            exit_result = await self._engine.execute_breach(ticker, pos, price, reason)
            if exit_result:

                await self._on_event(exit_result)
                await self.run_risk_evaluation()
                if self._stream:
                    all_pos = load_positions().get("positions", [])
                    self._stream.update_subscriptions({p["ticker"] for p in all_pos})
        finally:
            async with self._exit_lock:
                self._exiting_tickers.discard(ticker)

    async def _should_use_fast_alerts(self, positions: list[dict], quotes: dict[str, dict]) -> bool:
        """True if any position is near its stop or portfolio is drawing down."""
        for pos in positions:
            ticker = pos["ticker"]
            price = quotes.get(ticker, {}).get("price", 0)
            stop = max(pos.get("stop_price", 0), pos.get("trailing_stop", 0))
            if stop > 0 and price > 0:
                distance_pct = (price / stop - 1) * 100
                if 0 < distance_pct <= PARAMS.stop_proximity_pct:
                    return True
        try:
            dd = compute_drawdown(100_000)
            if dd >= PARAMS.risk_tier1_dd:
                return True
        except Exception:
            logger.warning("Drawdown computation failed in _should_use_fast_alerts", exc_info=True)
        return False

    async def _store_order_ids(self, results: list) -> None:
        """Persist Alpaca order IDs to position records after entry fills."""
        state = load_positions()
        for result in results:
            if result.success and result.order_id:
                for pos in state["positions"]:
                    if pos["ticker"] == result.ticker:
                        if not pos.get("order_id"):
                            pos["order_id"] = result.order_id
                        if result.qty:
                            pos["qty"] = result.qty
                        if result.trail_order_id:
                            pos["trail_order_id"] = result.trail_order_id
                        break
        await async_save_positions(state)

    def _has_open_crypto_positions(self) -> bool:
        """True if any open position is a crypto asset."""
        from config import ASSETS
        positions = load_positions().get("positions", [])
        return any(
            ASSETS.get(p["ticker"]) and ASSETS[p["ticker"]].asset_class == "crypto"
            for p in positions
        )

    def _save_watchlist(self, results: list, positions: dict) -> None:
        """Persist scan results as watchlist for agent queries."""
        open_tickers = {p["ticker"] for p in positions.get("positions", [])}
        watchlist = []
        for r in sorted(results, key=lambda x: x.get("p_rally", 0), reverse=True):
            if r.get("status") != "ok":
                continue
            if r["ticker"] in open_tickers:
                continue
            watchlist.append({
                "ticker": r["ticker"],
                "p_rally": round(r.get("p_rally", 0) * 100, 1),
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

    async def _execute_and_log_entries(self, signals: list[dict], equity: float) -> list:
        """Execute entry orders, log each result, and update positions + stream subscriptions."""
        entry_results = await execute_entries(signals, equity=equity)
        ok = [r for r in entry_results if r.success]
        for r in entry_results:
            log_order(r.ticker, "buy", "market", r.qty, "entry",
                      r.order_id, "filled" if r.success else "failed",
                      r.fill_price, r.error)
        if ok:
            size_map = {r.ticker: r.actual_size for r in ok if r.actual_size is not None}
            filled_tickers = {r.ticker for r in ok}
            filled_signals = [
                {**s, "size": size_map.get(s["ticker"], s["size"])}
                for s in signals if s["ticker"] in filled_tickers
            ]
            fresh_positions = load_positions()
            add_signal_positions(fresh_positions, filled_signals)
            await self._store_order_ids(entry_results)
            if self._stream:
                all_pos = load_positions().get("positions", [])
                self._stream.update_subscriptions({p["ticker"] for p in all_pos})
        return entry_results

    async def _execute_queued_entries(self, equity: float) -> list:
        """Re-attempt queued signals freed by today's closes."""
        queued = await asyncio.to_thread(process_signal_queue)
        if not queued:
            return []
        q_results = await execute_entries(queued, equity=equity)
        q_ok = [r for r in q_results if r.success]
        if q_ok:
            q_size_map = {r.ticker: r.actual_size for r in q_ok if r.actual_size is not None}
            filled_queued = [
                {**s, "size": q_size_map.get(s["ticker"], s["size"])}
                for s in queued if s["ticker"] in {r.ticker for r in q_ok}
            ]
            add_signal_positions(load_positions(), filled_queued)
            await self._store_order_ids(q_results)
        return q_results
