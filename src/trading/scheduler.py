"""
TradingScheduler — decoupled orchestration of all trading tasks.

Owns AlertEngine + AlpacaStreamManager. Emits typed events via callback.
Zero Discord imports.

Scan operations live in scheduler_ops, background loops in scheduler_loops,
and execution helpers in scheduler_exec.
"""

import asyncio
import logging
import zoneinfo
from collections.abc import Awaitable, Callable
from datetime import datetime

import sentry_sdk
from rally_ml.config import ASSETS, PARAMS
from rally_ml.core.persistence import load_manifest

from db.positions import (
    load_position_meta as _load_position_meta,
)
from db.positions import load_positions
from db.positions import (
    save_position_meta as _save_position_meta,
)
from integrations.alpaca.executor import (
    is_enabled as alpaca_enabled,
)
from integrations.alpaca.stream import AlpacaStreamManager, is_stream_enabled
from trading.engine import AlertEngine
from trading.events import (
    AlertEvent,
    TradingEvent,
)
from trading.positions import update_position_for_price
from trading.scheduler_loops import (
    housekeeping_loop,
    polling_loop,
    reconcile_loop,
    regime_loop,
    retrain_loop,
    scan_loop,
)
from trading.scheduler_ops import (
    run_daily_scan as _run_daily_scan,
)
from trading.scheduler_ops import (
    run_market_open_execute as _run_market_open_execute,
)
from trading.scheduler_ops import (
    run_midday_scan as _run_midday_scan,
)
from trading.scheduler_ops import (
    run_premarket_scan as _run_premarket_scan,
)
from trading.scheduler_ops import (
    run_regime_check as _run_regime_check,
)
from trading.scheduler_ops import (
    run_retrain as _run_retrain,
)
from trading.scheduler_ops import (
    run_risk_evaluation as _run_risk_evaluation,
)

_ET = zoneinfo.ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


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
        self._exit_lock = asyncio.Lock()
        self._snapshot_lock = asyncio.Lock()

        # Start stream if Alpaca + streaming both enabled
        if alpaca_enabled() and is_stream_enabled():
            try:
                positions = load_positions()
                tickers = {p["ticker"] for p in positions.get("positions", [])}
                self._stream = AlpacaStreamManager(
                    on_trade=self._on_stream_trade,
                    excluded=set(),
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
    # Public actions (delegate to scheduler_ops)
    # ------------------------------------------------------------------

    async def run_daily_scan(
        self, scan_type: str = "daily", tickers: list[str] | None = None,
    ) -> None:
        await _run_daily_scan(self, scan_type, tickers)

    async def run_premarket_scan(self) -> None:
        await _run_premarket_scan(self)

    async def run_market_open_execute(self) -> None:
        await _run_market_open_execute(self)

    async def run_midday_scan(self) -> None:
        await _run_midday_scan(self)

    async def run_regime_check(self) -> None:
        await _run_regime_check(self)

    async def run_retrain(self) -> None:
        await _run_retrain(self)

    async def run_risk_evaluation(self) -> None:
        await _run_risk_evaluation(self)

    # ------------------------------------------------------------------
    # Stream callback (sync — called from stream daemon thread)
    # ------------------------------------------------------------------

    def _on_stream_trade(self, ticker: str, price: float) -> None:
        """Called from stream thread per throttled trade event."""
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
            fresh = _load_position_meta(event.ticker)
            if fresh is not None:
                pos = fresh
            await self._execute_breach_guarded(event.ticker, pos, price, event.alert_type)

    # ------------------------------------------------------------------
    # Helpers (kept here — used by loops and stream callback)
    # ------------------------------------------------------------------

    def _task_sentinel(self, task: asyncio.Task) -> None:
        """Log critical if a background task dies unexpectedly."""
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
        """Execute a breach exit with concurrent-exit guard."""
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
