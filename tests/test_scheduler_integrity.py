"""Tests for TradingScheduler invariants — focused on _store_order_ids."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


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
    scheduler = _make_scheduler()

    existing_positions = {
        "positions": [
            {"ticker": "AAPL", "order_id": "original-id-123", "qty": 10},
        ],
    }
    scheduler._positions_cache = existing_positions

    result = _make_result(ticker="AAPL", order_id="new-id-456", qty=20)

    with patch("trading.scheduler.async_save_positions", new_callable=AsyncMock) as mock_save:
        await scheduler._store_order_ids([result])

        # order_id must be preserved
        pos = existing_positions["positions"][0]
        assert pos["order_id"] == "original-id-123"
        # qty IS updated even when order_id is preserved
        assert pos["qty"] == 20
        mock_save.assert_awaited_once_with(existing_positions)


@pytest.mark.asyncio
async def test_store_order_ids_sets_order_id_when_missing():
    """When a position has no order_id, _store_order_ids must set it."""
    scheduler = _make_scheduler()

    existing_positions = {
        "positions": [
            {"ticker": "MSFT"},
        ],
    }
    scheduler._positions_cache = existing_positions

    result = _make_result(ticker="MSFT", order_id="msft-order-789", qty=5,
                          trail_order_id="trail-abc")

    with patch("trading.scheduler.async_save_positions", new_callable=AsyncMock) as mock_save:
        await scheduler._store_order_ids([result])

        pos = existing_positions["positions"][0]
        assert pos["order_id"] == "msft-order-789"
        assert pos["qty"] == 5
        assert pos["trail_order_id"] == "trail-abc"
        mock_save.assert_awaited_once_with(existing_positions)


@pytest.mark.asyncio
async def test_store_order_ids_skips_failed_results():
    """Results with success=False should not modify positions."""
    scheduler = _make_scheduler()

    existing_positions = {
        "positions": [
            {"ticker": "TSLA"},
        ],
    }
    scheduler._positions_cache = existing_positions

    result = _make_result(ticker="TSLA", order_id="fail-id", success=False)

    with patch("trading.scheduler.async_save_positions", new_callable=AsyncMock) as mock_save:
        await scheduler._store_order_ids([result])

        pos = existing_positions["positions"][0]
        assert "order_id" not in pos
        mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_order_ids_skips_result_without_order_id():
    """Results with order_id=None should not modify positions."""
    scheduler = _make_scheduler()

    existing_positions = {
        "positions": [
            {"ticker": "GOOG"},
        ],
    }
    scheduler._positions_cache = existing_positions

    result = _make_result(ticker="GOOG", order_id=None, success=True)

    with patch("trading.scheduler.async_save_positions", new_callable=AsyncMock) as mock_save:
        await scheduler._store_order_ids([result])

        pos = existing_positions["positions"][0]
        assert "order_id" not in pos
        mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_order_ids_invalidates_cache():
    """After storing order IDs, the positions cache must be invalidated."""
    scheduler = _make_scheduler()

    scheduler._positions_cache = {"positions": []}

    with patch("trading.scheduler.async_save_positions", new_callable=AsyncMock):
        await scheduler._store_order_ids([])

    assert scheduler._positions_cache is None
