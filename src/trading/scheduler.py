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

from config import PARAMS
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
    record_closed_position as _rec_closed,
)
from db.positions import (
    save_latest_scan as _save_latest_scan,
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
    execute_entries,
    execute_exits,
    get_account_equity,
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
    get_total_exposure,
    process_signal_queue,
    sync_positions_from_alpaca,
    update_existing_positions,
    update_skipped_outcomes,
)
from trading.regime_monitor import check_regime_shifts, get_regime_states, is_cascade
from trading.risk_manager import evaluate, execute_actions
from trading.scheduler_loops import (  # noqa: F401 — re-export
    housekeeping_loop,
    maybe_run_weekend_crypto_scan,
    polling_loop,
    reconcile_loop,
    regime_loop,
    retrain_loop,
    scan_loop,
)
from trading.scheduler_stream import (  # noqa: F401 — re-export
    execute_breach_guarded,
    handle_stream_alert,
    has_open_crypto_positions,
    on_stream_trade,
    save_watchlist,
    should_use_fast_alerts,
    store_order_ids,
)

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
        # Weekend crypto scan dedup: "{date}_{6h_slot}"
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

        # Snapshot fetch guard
        self._snapshot_lock: asyncio.Lock  # assigned in start()

        # Housekeeping cycle counter for IEX coverage fallback
        self._housekeeping_cycles: int = 0

        # Stream health monitoring
        self._stream_degraded_cycles: int = 0
        self._stream_alert_sent: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start stream (if enabled) and all housekeeping loops."""
        self._loop = asyncio.get_event_loop()
        self._exit_lock = asyncio.Lock()
        self._snapshot_lock = asyncio.Lock()

        # Start stream if Alpaca + streaming both enabled
        if alpaca_enabled() and is_stream_enabled():
            try:
                positions = load_positions()
                tickers = {p["ticker"] for p in positions.get("positions", [])}
                self._stream = AlpacaStreamManager(
                    on_trade=lambda ticker, price: on_stream_trade(self, ticker, price),
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
            asyncio.create_task(housekeeping_loop(self), name="housekeeping"),
            asyncio.create_task(polling_loop(self), name="polling"),
            asyncio.create_task(reconcile_loop(self), name="reconcile"),
            asyncio.create_task(regime_loop(self), name="regime"),
            asyncio.create_task(scan_loop(self), name="scan"),
            asyncio.create_task(retrain_loop(self), name="retrain"),
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

    async def _prepare_scan_results(
        self, results: list[dict],
    ) -> tuple[list[dict], dict, list[dict], list[dict]]:
        """Shared scan post-processing: sync, detect exits, filter signals.

        Returns (signals, positions, closed, positions_for_embed).
        """
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
        save_watchlist(results, positions)
        await asyncio.to_thread(_save_latest_scan, results, positions)

        # Refresh watchlist for mid-day scans
        manifest = load_manifest()
        if manifest:
            self._watchlist_tickers = sorted(manifest.keys())

        open_tickers = {p["ticker"] for p in positions.get("positions", [])}
        cooldown_tickers: set[str] = set()
        if PARAMS.cooldown_days > 0:
            cooldown_tickers = get_recently_closed_tickers(PARAMS.cooldown_days)
            cooled = [
                s["ticker"] for s in all_signals
                if s["ticker"] not in open_tickers
                and s["ticker"] in cooldown_tickers
            ]
            if cooled:
                logger.info(
                    "Cooldown filter (%dd): skipping %s",
                    PARAMS.cooldown_days, cooled,
                )
        signals = [
            s for s in all_signals
            if s["ticker"] not in open_tickers
            and s["ticker"] not in cooldown_tickers
        ]

        positions_for_embed = list(positions.get("positions", []))
        return signals, positions, closed, positions_for_embed

    async def run_daily_scan(
        self, scan_type: str = "daily", tickers: list[str] | None = None,
    ) -> None:
        """Run the full scan pipeline and emit a ScanResult event."""
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

        signals, positions, closed, positions_for_embed = (
            await self._prepare_scan_results(results)
        )

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
                    entry_results = await execute_entries(signals, equity=equity)
                    ok = [r for r in entry_results if r.success]
                    for r in entry_results:
                        log_order(r.ticker, "buy", "market", r.qty, "entry",
                                  r.order_id, "filled" if r.success else "failed",
                                  r.fill_price, r.error)
                    if ok:
                        size_map = {
                            r.ticker: r.actual_size for r in ok if r.actual_size is not None
                        }
                        filled_tickers = {r.ticker for r in ok}
                        filled_signals = [
                            {**s, "size": size_map.get(s["ticker"], s["size"])}
                            for s in signals if s["ticker"] in filled_tickers
                        ]

                        fresh_positions = load_positions()
                        add_signal_positions(fresh_positions, filled_signals)

                        await store_order_ids(entry_results)
                        # Update stream subscriptions after new entries
                        if self._stream:
                            all_pos = load_positions().get("positions", [])
                            self._stream.update_subscriptions({p["ticker"] for p in all_pos})
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
                queued = await asyncio.to_thread(process_signal_queue)
                if queued:
                    q_results = await execute_entries(queued, equity=equity)
                    q_ok = [r for r in q_results if r.success]
                    if q_ok:
                        q_size_map = {
                            r.ticker: r.actual_size for r in q_ok if r.actual_size is not None
                        }
                        filled_queued = [
                            {**s, "size": q_size_map.get(s["ticker"], s["size"])}
                            for s in queued if s["ticker"] in {r.ticker for r in q_ok}
                        ]

                        add_signal_positions(load_positions(), filled_queued)

                        await store_order_ids(q_results)
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

        signals, positions, closed, positions_for_embed = (
            await self._prepare_scan_results(results)
        )

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
                    entry_results = await execute_entries(signals, equity=equity)
                    ok = [r for r in entry_results if r.success]
                    for r in entry_results:
                        log_order(r.ticker, "buy", "market", r.qty, "entry",
                                  r.order_id, "filled" if r.success else "failed",
                                  r.fill_price, r.error)
                    if ok:
                        size_map = {
                            r.ticker: r.actual_size for r in ok if r.actual_size is not None
                        }
                        filled_tickers = {r.ticker for r in ok}
                        filled_signals = [
                            {**s, "size": size_map.get(s["ticker"], s["size"])}
                            for s in signals if s["ticker"] in filled_tickers
                        ]

                        fresh_positions = load_positions()
                        add_signal_positions(fresh_positions, filled_signals)

                        await store_order_ids(entry_results)
                        if self._stream:
                            all_pos = load_positions().get("positions", [])
                            self._stream.update_subscriptions({p["ticker"] for p in all_pos})
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
                queued = await asyncio.to_thread(process_signal_queue)
                if queued:
                    q_results = await execute_entries(queued, equity=equity)
                    q_ok = [r for r in q_results if r.success]
                    if q_ok:
                        q_size_map = {
                            r.ticker: r.actual_size for r in q_ok if r.actual_size is not None
                        }
                        filled_queued = [
                            {**s, "size": q_size_map.get(s["ticker"], s["size"])}
                            for s in queued if s["ticker"] in {r.ticker for r in q_ok}
                        ]

                        add_signal_positions(load_positions(), filled_queued)

                        await store_order_ids(q_results)
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
    # Helpers
    # ------------------------------------------------------------------

    def _task_sentinel(self, task: asyncio.Task) -> None:
        """done_callback: log critical if a background task dies unexpectedly."""
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

    # Backward-compat instance method delegates
    async def _store_order_ids(self, results: list) -> None:
        await store_order_ids(results)

    def _has_open_crypto_positions(self) -> bool:
        return has_open_crypto_positions()

    def _save_watchlist(self, results: list, positions: dict) -> None:
        save_watchlist(results, positions)

    async def _should_use_fast_alerts(
        self, positions: list[dict], quotes: dict[str, dict],
    ) -> bool:
        return await should_use_fast_alerts(positions, quotes)

    async def _execute_breach_guarded(
        self, ticker: str, pos: dict, price: float, alert_type: str,
    ) -> None:
        await execute_breach_guarded(self, ticker, pos, price, alert_type)

    def _on_stream_trade(self, ticker: str, price: float) -> None:
        on_stream_trade(self, ticker, price)

    async def _handle_stream_alert(
        self, event: AlertEvent, pos: dict, price: float,
    ) -> None:
        await handle_stream_alert(self, event, pos, price)

    async def _housekeeping_loop(self) -> None:
        await housekeeping_loop(self)

    async def _polling_loop(self) -> None:
        await polling_loop(self)

    async def _reconcile_loop(self) -> None:
        await reconcile_loop(self)

    async def _regime_loop(self) -> None:
        await regime_loop(self)

    async def _scan_loop(self) -> None:
        await scan_loop(self)

    async def _retrain_loop(self) -> None:
        await retrain_loop(self)

    async def _maybe_run_weekend_crypto_scan(self, now, today: str) -> None:
        await maybe_run_weekend_crypto_scan(self, now, today)
