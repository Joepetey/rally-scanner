"""Tests for position tracking — DB-backed CRUD, update, exit logic."""

from unittest.mock import AsyncMock, patch

import pytest

from config import PARAMS
from db.positions import load_positions, save_positions, tighten_trailing_stop
from trading.positions import (
    add_signal_positions,
    get_merged_positions,
    sync_positions_from_alpaca,
    update_positions,
)


def test_load_empty(tmp_models_dir):
    state = load_positions()
    assert state["positions"] == []
    assert state["closed_today"] == []


def test_save_and_load_roundtrip(tmp_models_dir):
    state = {
        "positions": [
            {
                "ticker": "AAPL",
                "entry_price": 150.0,
                "current_price": 155.0,
                "stop_price": 145.0,
                "target_price": 160.0,
                "trailing_stop": 148.0,
                "highest_close": 155.0,
                "bars_held": 3,
                "size": 0.10,
                "unrealized_pnl_pct": 3.33,
                "status": "open",
                "entry_date": "2024-01-10",
            },
        ],
        "closed_today": [],
    }
    save_positions(state)
    loaded = load_positions()
    assert len(loaded["positions"]) == 1
    assert loaded["positions"][0]["ticker"] == "AAPL"
    assert loaded["last_updated"] is not None


def test_update_adds_new_signal(tmp_models_dir):
    state = {"positions": [], "closed_today": []}
    signals = [{
        "ticker": "MSFT",
        "date": "2024-01-15",
        "close": 400.0,
        "size": 0.08,
        "range_low": 390.0,
        "atr_pct": 0.02,
    }]
    results = [{"ticker": "MSFT", "status": "ok", "close": 400.0, "date": "2024-01-15"}]
    updated = update_positions(state, signals, results)
    assert len(updated["positions"]) == 1
    assert updated["positions"][0]["ticker"] == "MSFT"
    assert updated["positions"][0]["entry_price"] == 400.0


def test_update_exits_on_stop(tmp_models_dir):
    state = {
        "positions": [{
            "ticker": "AAPL",
            "entry_price": 150.0,
            "stop_price": 145.0,
            "target_price": 160.0,
            "trailing_stop": 148.0,
            "highest_close": 155.0,
            "bars_held": 2,
            "size": 0.10,
            "status": "open",
            "entry_date": "2024-01-10",
        }],
        "closed_today": [],
    }
    # Price drops to stop
    results = [{
        "ticker": "AAPL",
        "status": "ok",
        "close": 144.0,
        "date": "2024-01-15",
        "atr": 3.0,
    }]
    updated = update_positions(state, [], results)
    assert len(updated["positions"]) == 0
    assert len(updated["closed_today"]) == 1
    assert updated["closed_today"][0]["exit_reason"] == "stop"


def test_update_exits_on_profit_target(tmp_models_dir):
    state = {
        "positions": [{
            "ticker": "NVDA",
            "entry_price": 500.0,
            "stop_price": 480.0,
            "target_price": 530.0,
            "trailing_stop": 490.0,
            "highest_close": 510.0,
            "bars_held": 3,
            "size": 0.05,
            "status": "open",
            "entry_date": "2024-01-10",
        }],
        "closed_today": [],
    }
    results = [{
        "ticker": "NVDA",
        "status": "ok",
        "close": 535.0,
        "date": "2024-01-15",
        "atr": 10.0,
    }]
    updated = update_positions(state, [], results)
    assert len(updated["closed_today"]) == 1
    assert updated["closed_today"][0]["exit_reason"] == "profit_target"


def test_update_exits_on_time_stop(tmp_models_dir):
    state = {
        "positions": [{
            "ticker": "SPY",
            "entry_price": 450.0,
            "stop_price": 440.0,
            "target_price": 470.0,
            "trailing_stop": 445.0,
            "highest_close": 452.0,
            "bars_held": PARAMS.time_stop_bars - 1,  # will be incremented to threshold
            "size": 0.15,
            "status": "open",
            "entry_date": "2024-01-10",
        }],
        "closed_today": [],
    }
    # Price is between stop and target, no trailing stop hit
    results = [{
        "ticker": "SPY",
        "status": "ok",
        "close": 451.0,
        "date": "2024-01-25",
        "atr": 5.0,
    }]
    updated = update_positions(state, [], results)
    assert len(updated["closed_today"]) == 1
    assert updated["closed_today"][0]["exit_reason"] == "time_stop"


def test_no_duplicate_positions(tmp_models_dir):
    state = {
        "positions": [{
            "ticker": "AAPL",
            "entry_price": 150.0,
            "stop_price": 145.0,
            "target_price": 160.0,
            "trailing_stop": 148.0,
            "highest_close": 155.0,
            "bars_held": 1,
            "size": 0.10,
            "status": "open",
            "entry_date": "2024-01-10",
        }],
        "closed_today": [],
    }
    # New signal for same ticker — should not add duplicate
    signals = [{
        "ticker": "AAPL",
        "date": "2024-01-15",
        "close": 155.0,
        "size": 0.12,
        "range_low": 148.0,
        "atr_pct": 0.02,
    }]
    results = [{"ticker": "AAPL", "status": "ok", "close": 155.0, "date": "2024-01-15"}]
    updated = update_positions(state, signals, results)
    assert len(updated["positions"]) == 1


def test_tighten_trailing_stop(tmp_models_dir):
    """tighten_trailing_stop only tightens (never loosens)."""
    state = {
        "positions": [{
            "ticker": "AAPL",
            "entry_price": 150.0,
            "stop_price": 145.0,
            "target_price": 160.0,
            "trailing_stop": 148.0,
            "highest_close": 155.0,
            "bars_held": 3,
            "size": 0.10,
            "status": "open",
            "entry_date": "2024-01-10",
        }],
        "closed_today": [],
    }
    save_positions(state)

    # Tighten: 148 → 150 (higher = tighter)
    result = tighten_trailing_stop("AAPL", 150.0)
    assert result is not None
    assert result["trailing_stop"] == 150.0

    # Verify persisted
    loaded = load_positions()
    assert loaded["positions"][0]["trailing_stop"] == 150.0


# ---------------------------------------------------------------------------
# update_existing_positions / add_signal_positions (split functions)
# ---------------------------------------------------------------------------


def test_add_signal_positions_respects_exposure_cap(tmp_models_dir):
    """add_signal_positions skips signals that would exceed exposure cap."""
    state = {
        "positions": [{
            "ticker": "AAPL",
            "entry_price": 150.0,
            "size": PARAMS.max_portfolio_exposure - 0.01,
            "status": "open",
            "entry_date": "2024-01-10",
        }],
        "closed_today": [],
    }
    signals = [{
        "ticker": "MSFT",
        "date": "2024-01-15",
        "close": 400.0,
        "size": 0.10,
        "range_low": 390.0,
        "atr_pct": 0.02,
    }]
    updated = add_signal_positions(state, signals)
    # MSFT should not be added — would exceed max exposure
    assert len(updated["positions"]) == 1
    assert updated["positions"][0]["ticker"] == "AAPL"




# ---------------------------------------------------------------------------
# get_merged_positions (Alpaca + DB metadata)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merged_positions_alpaca_source(tmp_models_dir):
    """get_merged_positions merges Alpaca holdings with DB metadata."""
    # Seed DB with metadata
    save_positions({
        "positions": [{
            "ticker": "AAPL",
            "entry_price": 150.0,
            "entry_date": "2024-01-10",
            "stop_price": 145.0,
            "target_price": 160.0,
            "trailing_stop": 148.0,
            "highest_close": 155.0,
            "atr": 3.0,
            "bars_held": 5,
            "size": 0.10,
        }],
        "closed_today": [],
    })

    # Mock Alpaca returning AAPL
    mock_broker = AsyncMock(return_value=[{
        "ticker": "AAPL",
        "qty": 10,
        "avg_entry_price": 150.0,
        "market_value": 1550.0,
        "unrealized_pl": 50.0,
    }])

    with patch("trading.alpaca_executor.get_all_positions", mock_broker):
        state = await get_merged_positions()

    assert len(state["positions"]) == 1
    pos = state["positions"][0]
    assert pos["ticker"] == "AAPL"
    assert pos["qty"] == 10
    # Metadata from DB
    assert pos["stop_price"] == 145.0
    assert pos["target_price"] == 160.0
    assert pos["trailing_stop"] == 148.0
    assert pos["bars_held"] == 5


# ---------------------------------------------------------------------------
# sync_positions_from_alpaca — three reconcile cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_updates_qty_and_entry_keeps_metadata(tmp_models_dir):
    """Case 1: position in both Alpaca and DB — broker qty/entry wins, metadata preserved."""
    save_positions({"positions": [{
        "ticker": "AAPL", "entry_price": 150.0, "entry_date": "2024-01-10",
        "stop_price": 145.0, "target_price": 160.0, "trailing_stop": 148.0,
        "highest_close": 155.0, "atr": 3.0, "bars_held": 5, "size": 0.10, "qty": 10,
    }], "closed_today": []})

    mock_broker = AsyncMock(return_value=[{
        "ticker": "AAPL", "qty": 12, "avg_entry_price": 152.0,
        "market_value": 1824.0, "unrealized_pl": 24.0,
    }])
    with patch("trading.positions.get_all_positions", mock_broker), \
         patch("trading.positions.get_recent_sell_fills", AsyncMock(return_value={})):
        result = await sync_positions_from_alpaca()

    assert result == {"synced": 1, "closed": 0, "inserted": 0}
    from db.positions import load_position_meta
    pos = load_position_meta("AAPL")
    assert pos["qty"] == 12
    assert pos["entry_price"] == 152.0
    assert pos["stop_price"] == 145.0   # DB metadata unchanged
    assert pos["bars_held"] == 5        # DB metadata unchanged


@pytest.mark.asyncio
async def test_sync_inserts_untracked_alpaca_position(tmp_models_dir):
    """Case 2: position in Alpaca but not DB — insert with broker values + derived stops."""
    mock_broker = AsyncMock(return_value=[{
        "ticker": "TSLA", "qty": 5, "avg_entry_price": 200.0,
        "market_value": 1050.0, "unrealized_pl": 50.0,
    }])
    with patch("trading.positions.get_all_positions", mock_broker), \
         patch("trading.positions.get_recent_sell_fills", AsyncMock(return_value={})):
        result = await sync_positions_from_alpaca()

    assert result == {"synced": 0, "closed": 0, "inserted": 1}
    from db.positions import load_position_meta
    pos = load_position_meta("TSLA")
    assert pos["qty"] == 5
    assert pos["entry_price"] == 200.0
    assert pos["stop_price"] == round(200.0 * (1 - PARAMS.fallback_stop_pct), 4)
    assert pos["target_price"] > 200.0
    assert pos["trailing_stop"] < 200.0
    assert pos["bars_held"] == 0


@pytest.mark.asyncio
async def test_sync_closes_position_missing_from_broker(tmp_models_dir):
    """Case 3: position in DB but not Alpaca — broker closed it, recover fill and record."""
    save_positions({"positions": [{
        "ticker": "MSFT", "entry_price": 400.0, "entry_date": "2024-01-10",
        "stop_price": 390.0, "target_price": 420.0, "trailing_stop": 395.0,
        "highest_close": 410.0, "atr": 4.0, "bars_held": 3, "size": 0.08, "qty": 8,
    }], "closed_today": []})

    with patch("trading.positions.get_all_positions", AsyncMock(return_value=[])), \
         patch("trading.positions.get_recent_sell_fills", AsyncMock(return_value={"MSFT": 405.0})):
        result = await sync_positions_from_alpaca()

    assert result == {"synced": 0, "closed": 1, "inserted": 0}
    from db.positions import get_closed_today, load_position_meta
    assert load_position_meta("MSFT") is None
    closed = get_closed_today()
    assert len(closed) == 1
    assert closed[0]["ticker"] == "MSFT"
    assert closed[0]["exit_reason"] == "broker_closed"
    assert closed[0]["exit_price"] == 405.0


def test_get_recently_closed_tickers(tmp_models_dir):
    """get_recently_closed_tickers returns tickers closed within N days."""
    from datetime import date, timedelta

    from db.positions import get_recently_closed_tickers, record_closed_position

    # Insert a position closed today
    record_closed_position({
        "ticker": "AAPL", "entry_price": 150.0, "entry_date": str(date.today() - timedelta(days=5)),
        "exit_price": 160.0, "exit_date": str(date.today()),
        "exit_reason": "profit_target", "realized_pnl_pct": 0.067, "bars_held": 5, "size": 0.10,
    })
    # Insert a position closed 20 days ago
    record_closed_position({
        "ticker": "OLD", "entry_price": 100.0, "entry_date": str(date.today() - timedelta(days=30)),
        "exit_price": 95.0, "exit_date": str(date.today() - timedelta(days=20)),
        "exit_reason": "stop", "realized_pnl_pct": -0.05, "bars_held": 10, "size": 0.08,
    })

    recent_10 = get_recently_closed_tickers(10)
    assert "AAPL" in recent_10
    assert "OLD" not in recent_10

    recent_30 = get_recently_closed_tickers(30)
    assert "AAPL" in recent_30
    assert "OLD" in recent_30

    recent_0 = get_recently_closed_tickers(0)
    assert "AAPL" in recent_0  # closed today
    assert "OLD" not in recent_0
