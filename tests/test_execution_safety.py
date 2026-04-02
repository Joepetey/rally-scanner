"""Execution safety tests — duplicate orders, stale fills, ghost closes."""

from datetime import UTC, datetime, timedelta

import pytest
from alpaca.trading.enums import OrderSide

from integrations.alpaca.fills import check_pending_fills, check_trail_stop_fills

# ---------------------------------------------------------------------------
# Stale fill rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_pending_fills_ignores_fills_from_previous_days(alpaca_mock):
    """Fills from prior days must not leak into today's results."""
    order = alpaca_mock.add_filled_order("stale-order", fill_price=150.0)
    order.filled_at = datetime.now(tz=UTC) - timedelta(days=3)

    result = await check_pending_fills(["stale-order"])
    assert result == {}


@pytest.mark.asyncio
async def test_check_pending_fills_accepts_todays_fill(alpaca_mock):
    """A fill from today should be returned with its price."""
    alpaca_mock.add_filled_order("fresh-order", fill_price=142.50)

    result = await check_pending_fills(["fresh-order"])
    assert result == {"fresh-order": 142.50}


@pytest.mark.asyncio
async def test_check_pending_fills_empty_input(alpaca_mock):
    """Empty order list should short-circuit to empty dict."""
    result = await check_pending_fills([])
    assert result == {}


@pytest.mark.asyncio
async def test_check_pending_fills_mixed_stale_and_fresh(alpaca_mock):
    """Only today's fills are returned when mixed with stale ones."""
    stale = alpaca_mock.add_filled_order("old-1", fill_price=100.0)
    stale.filled_at = datetime.now(tz=UTC) - timedelta(days=1)

    alpaca_mock.add_filled_order("new-1", fill_price=200.0)

    result = await check_pending_fills(["old-1", "new-1"])
    assert result == {"new-1": 200.0}


@pytest.mark.asyncio
async def test_check_pending_fills_ignores_unknown_order_ids(alpaca_mock):
    """Order IDs not in the request list are excluded even if filled today."""
    alpaca_mock.add_filled_order("known", fill_price=100.0)
    alpaca_mock.add_filled_order("unknown", fill_price=200.0)

    result = await check_pending_fills(["known"])
    assert result == {"known": 100.0}


# ---------------------------------------------------------------------------
# Trail stop fill detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_trail_stop_fills_detects_filled_trail(alpaca_mock):
    """Filled trailing stop returns {ticker: fill_price}."""
    alpaca_mock.add_filled_order("trail-001", fill_price=95.0, symbol="MSFT",
                                 side=OrderSide.SELL)

    result = await check_trail_stop_fills({"MSFT": "trail-001"})
    assert result == {"MSFT": 95.0}


@pytest.mark.asyncio
async def test_check_trail_stop_fills_empty_input(alpaca_mock):
    """Empty trail map short-circuits."""
    result = await check_trail_stop_fills({})
    assert result == {}
