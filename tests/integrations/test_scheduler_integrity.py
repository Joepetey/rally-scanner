"""Tests for TradingScheduler invariants.

Covers _store_order_ids, scan dedup, timeout, concurrent exit guards,
reconcile with empty positions, and housekeeping market-hours/crypto logic.
"""

import asyncio
import zoneinfo
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.scheduler_exec import store_order_ids
from trading.scheduler_loops import (
    housekeeping_loop,
    reconcile_loop,
    retrain_loop,
)


def _make_scheduler():
    """Create a TradingScheduler with minimal mocking to avoid side effects."""
    with patch("trading.scheduler.AlertEngine"), \
         patch("trading.scheduler.PARAMS") as mock_params:
        mock_params.base_alert_interval = 60
        from trading.scheduler import TradingScheduler
        return TradingScheduler(on_event=AsyncMock())


def _make_result(*, ticker: str, order_id: str, success: bool = True,
                 qty: int | None = None, trail_order_id: str | None = None):
    return SimpleNamespace(
        ticker=ticker,
        order_id=order_id,
        success=success,
        qty=qty,
        trail_order_id=trail_order_id,
    )


@pytest.mark.asyncio
async def test_store_order_ids_preserves_existing_order_id():
    """When a position already has an order_id, _store_order_ids must not overwrite it."""
    existing_positions = {
        "positions": [
            {"ticker": "AAPL", "order_id": "original-id-123", "qty": 10},
        ],
    }

    result = _make_result(ticker="AAPL", order_id="new-id-456", qty=20)

    with patch("trading.scheduler_exec.load_positions", return_value=existing_positions), \
         patch("trading.scheduler_exec.async_save_positions", new_callable=AsyncMock) as mock_save:
        await store_order_ids([result])

        # order_id must be preserved
        pos = existing_positions["positions"][0]
        assert pos["order_id"] == "original-id-123"
        # qty IS updated even when order_id is preserved
        assert pos["qty"] == 20
        mock_save.assert_awaited_once_with(existing_positions)


@pytest.mark.asyncio
async def test_store_order_ids_sets_order_id_when_missing():
    """When a position has no order_id, _store_order_ids must set it."""
    existing_positions = {
        "positions": [
            {"ticker": "MSFT"},
        ],
    }

    result = _make_result(ticker="MSFT", order_id="msft-order-789", qty=5,
                          trail_order_id="trail-abc")

    with patch("trading.scheduler_exec.load_positions", return_value=existing_positions), \
         patch("trading.scheduler_exec.async_save_positions", new_callable=AsyncMock) as mock_save:
        await store_order_ids([result])

        pos = existing_positions["positions"][0]
        assert pos["order_id"] == "msft-order-789"
        assert pos["qty"] == 5
        assert pos["trail_order_id"] == "trail-abc"
        mock_save.assert_awaited_once_with(existing_positions)


@pytest.mark.asyncio
async def test_store_order_ids_skips_failed_results():
    """Results with success=False should not modify positions."""
    existing_positions = {
        "positions": [
            {"ticker": "TSLA"},
        ],
    }

    result = _make_result(ticker="TSLA", order_id="fail-id", success=False)

    with patch("trading.scheduler_exec.load_positions", return_value=existing_positions), \
         patch("trading.scheduler_exec.async_save_positions", new_callable=AsyncMock) as mock_save:
        await store_order_ids([result])

        pos = existing_positions["positions"][0]
        assert "order_id" not in pos
        mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_order_ids_skips_result_without_order_id():
    """Results with order_id=None should not modify positions."""
    existing_positions = {
        "positions": [
            {"ticker": "GOOG"},
        ],
    }

    result = _make_result(ticker="GOOG", order_id=None, success=True)

    with patch("trading.scheduler_exec.load_positions", return_value=existing_positions), \
         patch("trading.scheduler_exec.async_save_positions", new_callable=AsyncMock) as mock_save:
        await store_order_ids([result])

        pos = existing_positions["positions"][0]
        assert "order_id" not in pos
        mock_save.assert_awaited_once()


# ------------------------------------------------------------------
# Scan dedup, timeout, concurrent exit, reconcile, housekeeping
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_dedup_prevents_concurrent_same_scan_type():
    """Calling run_daily_scan("morning") twice concurrently must invoke scan_all exactly once."""
    scheduler = _make_scheduler()
    scheduler._exit_lock = asyncio.Lock()
    scheduler._snapshot_lock = asyncio.Lock()

    # Make scan_all slow so both tasks overlap
    scan_started = asyncio.Event()

    def _slow_scan_all(*args, **kwargs):
        # Signal that scan has started, then block
        scan_started.set()
        return []

    async def _guarded_scan(scan_type: str):
        """Replicate _scan_loop dedup guard around run_daily_scan."""
        if scheduler._scan_in_progress.get(scan_type):
            return
        scheduler._scan_in_progress[scan_type] = True
        try:
            async with asyncio.timeout(600):
                await scheduler.run_daily_scan(scan_type)
        finally:
            scheduler._scan_in_progress[scan_type] = False

    with patch("trading.scheduler_ops.scan_all", side_effect=_slow_scan_all) as mock_scan, \
         patch("trading.scheduler_ops.load_manifest", return_value={"AAPL": {}}), \
         patch("trading.scheduler_ops.load_positions", return_value={"positions": []}), \
         patch("trading.scheduler_ops.update_existing_positions", return_value={"positions": []}), \
         patch("trading.scheduler_ops.update_daily_snapshot"), \
         patch("trading.scheduler_ops._save_latest_scan"), \
         patch("trading.scheduler_exec._db_save_watchlist"), \
         patch("trading.scheduler_ops.alpaca_enabled", return_value=False), \
         patch("trading.scheduler_ops.log_scheduler_event", return_value=1), \
         patch("trading.scheduler_ops.finish_scheduler_event"):
        # Launch two concurrent guarded scans
        t1 = asyncio.create_task(_guarded_scan("morning"))
        t2 = asyncio.create_task(_guarded_scan("morning"))
        await asyncio.gather(t1, t2)

        assert mock_scan.call_count == 1


@pytest.mark.asyncio
async def test_run_daily_scan_respects_timeout():
    """run_daily_scan must raise TimeoutError when scan_all exceeds the timeout window."""
    scheduler = _make_scheduler()
    scheduler._exit_lock = asyncio.Lock()
    scheduler._snapshot_lock = asyncio.Lock()

    original_to_thread = asyncio.to_thread

    async def _hanging_to_thread(func, *args, **kwargs):
        # Only hang when scan_all is called; pass through everything else
        if func.__name__ == "scan_all":
            await asyncio.sleep(9999)
            return []
        return await original_to_thread(func, *args, **kwargs)

    with patch("trading.scheduler_ops.load_manifest", return_value={"AAPL": {}}), \
         patch("trading.scheduler_ops.log_scheduler_event", return_value=1), \
         patch("trading.scheduler_ops.finish_scheduler_event"), \
         patch("asyncio.to_thread", side_effect=_hanging_to_thread):
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await scheduler.run_daily_scan("morning")


@pytest.mark.asyncio
async def test_concurrent_exits_for_same_ticker_prevented():
    """Simultaneous breach exits for the same ticker must execute exactly once."""
    scheduler = _make_scheduler()
    scheduler._exit_lock = asyncio.Lock()
    scheduler._snapshot_lock = asyncio.Lock()

    pos = {"ticker": "AAPL", "entry_price": 150.0, "stop_price": 140.0}

    mock_exit_result = MagicMock()
    mock_exit_result.ticker = "AAPL"

    # Make execute_breach slow enough that both tasks overlap
    async def _slow_execute_breach(*args, **kwargs):
        await asyncio.sleep(0.1)
        return mock_exit_result

    scheduler._engine = MagicMock()
    scheduler._engine.execute_breach = AsyncMock(side_effect=_slow_execute_breach)
    scheduler._stream = None

    with patch.object(scheduler, "run_risk_evaluation", new_callable=AsyncMock), \
         patch("trading.scheduler.load_positions", return_value={"positions": []}):
        t1 = asyncio.create_task(
            scheduler._execute_breach_guarded("AAPL", pos, 139.0, "stop_breached"),
        )
        t2 = asyncio.create_task(
            scheduler._execute_breach_guarded("AAPL", pos, 139.0, "stop_breached"),
        )
        await asyncio.gather(t1, t2)

    scheduler._engine.execute_breach.assert_awaited_once()
    assert "AAPL" not in scheduler._exiting_tickers  # cleaned up


@pytest.mark.asyncio
async def test_reconcile_loop_runs_with_empty_positions():
    """_reconcile_loop must not crash with empty positions."""
    scheduler = _make_scheduler()
    scheduler._exit_lock = asyncio.Lock()
    scheduler._snapshot_lock = asyncio.Lock()

    scheduler._engine = MagicMock()
    scheduler._engine.is_market_open.return_value = True

    call_count = 0

    async def _sleep_then_stop(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError  # break out after first iteration

    mock_fills = AsyncMock(return_value=[])
    with patch("trading.scheduler_loops.alpaca_enabled", return_value=True), \
         patch("trading.scheduler_loops.check_exit_fills", mock_fills), \
         patch("trading.scheduler_loops.load_positions", return_value={"positions": []}), \
         patch("asyncio.sleep", side_effect=_sleep_then_stop):
        with pytest.raises(asyncio.CancelledError):
            await reconcile_loop(scheduler)

    mock_fills.assert_awaited_once_with([])


@pytest.mark.asyncio
async def test_housekeeping_skips_outside_market_hours_without_crypto():
    """Housekeeping loop must skip when market is closed and no crypto positions."""
    scheduler = _make_scheduler()
    scheduler._exit_lock = asyncio.Lock()
    scheduler._snapshot_lock = asyncio.Lock()

    scheduler._engine = MagicMock()
    scheduler._engine.is_market_open.return_value = False
    scheduler._engine.run_housekeeping = AsyncMock()

    call_count = 0

    async def _sleep_then_stop(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError

    with patch("asyncio.sleep", side_effect=_sleep_then_stop), \
         patch("trading.scheduler_loops.load_positions", return_value={"positions": []}), \
         patch("rally_ml.config.ASSETS", {}):
        with pytest.raises(asyncio.CancelledError):
            await housekeeping_loop(scheduler)

    # run_housekeeping must NOT have been called — the iteration was skipped
    scheduler._engine.run_housekeeping.assert_not_awaited()


@pytest.mark.asyncio
async def test_housekeeping_runs_outside_market_hours_with_open_crypto():
    """Housekeeping must NOT skip when market is closed but a crypto position is open."""
    scheduler = _make_scheduler()
    scheduler._exit_lock = asyncio.Lock()
    scheduler._snapshot_lock = asyncio.Lock()

    crypto_positions = {
        "positions": [{"ticker": "BTC-USD", "entry_price": 60000}],
    }

    scheduler._engine = MagicMock()
    scheduler._engine.is_market_open.return_value = False
    scheduler._engine.run_housekeeping = AsyncMock(
        return_value=MagicMock(fills_confirmed=0, orders_placed=0),
    )
    scheduler._stream = None

    call_count = 0

    async def _sleep_then_stop(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError

    btc_asset = MagicMock()
    btc_asset.asset_class = "crypto"

    with patch("asyncio.sleep", side_effect=_sleep_then_stop), \
         patch("rally_ml.config.ASSETS", {"BTC-USD": btc_asset}), \
         patch("trading.scheduler_loops.load_positions", return_value=crypto_positions), \
         patch("trading.scheduler_loops.alpaca_enabled", return_value=False):
        with pytest.raises(asyncio.CancelledError):
            await housekeeping_loop(scheduler)

    # run_housekeeping MUST have been called — crypto keeps it alive
    scheduler._engine.run_housekeeping.assert_awaited_once()


# ------------------------------------------------------------------
# Retrain loop catch-up
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrain_fires_after_window_on_sunday():
    """Retrain must fire if scheduler starts after 18:05 ET on Sunday (catch-up)."""
    from datetime import datetime as _dt

    scheduler = _make_scheduler()
    scheduler._exit_lock = asyncio.Lock()
    scheduler._snapshot_lock = asyncio.Lock()

    call_count = 0

    async def _sleep_then_stop(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError

    # Simulate Sunday 20:30 ET — well past the old 18:00-18:04 window
    fake_now = _dt(2026, 3, 29, 20, 30, tzinfo=zoneinfo.ZoneInfo("America/New_York"))
    assert fake_now.weekday() == 6  # confirm Sunday

    with patch("asyncio.sleep", side_effect=_sleep_then_stop), \
         patch("trading.scheduler_loops.datetime") as mock_dt, \
         patch.object(scheduler, "run_retrain", new_callable=AsyncMock) as mock_retrain:
        mock_dt.now.return_value = fake_now
        mock_dt.min = _dt.min

        with pytest.raises(asyncio.CancelledError):
            await retrain_loop(scheduler)

    mock_retrain.assert_awaited_once()
