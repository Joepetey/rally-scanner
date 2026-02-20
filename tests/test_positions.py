"""Tests for position tracking — DB-backed CRUD, update, exit logic."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rally.config import PARAMS
from rally.trading.positions import (
    add_signal_positions,
    get_merged_positions,
    load_positions,
    save_positions,
    tighten_trailing_stop,
    update_existing_positions,
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


def test_tighten_trailing_stop_no_loosen(tmp_models_dir):
    """tighten_trailing_stop refuses to loosen (lower stop)."""
    state = {
        "positions": [{
            "ticker": "AAPL",
            "entry_price": 150.0,
            "trailing_stop": 148.0,
            "stop_price": 145.0,
            "target_price": 160.0,
            "highest_close": 155.0,
            "bars_held": 3,
            "size": 0.10,
            "status": "open",
            "entry_date": "2024-01-10",
        }],
        "closed_today": [],
    }
    save_positions(state)

    # Try to loosen: 148 → 145 (lower = looser) — should return None
    result = tighten_trailing_stop("AAPL", 145.0)
    assert result is None

    # Stop unchanged
    loaded = load_positions()
    assert loaded["positions"][0]["trailing_stop"] == 148.0


def test_tighten_trailing_stop_not_found(tmp_models_dir):
    """tighten_trailing_stop returns None for unknown ticker."""
    result = tighten_trailing_stop("UNKNOWN", 150.0)
    assert result is None


# ---------------------------------------------------------------------------
# update_existing_positions / add_signal_positions (split functions)
# ---------------------------------------------------------------------------


def test_update_existing_no_new_positions(tmp_models_dir):
    """update_existing_positions never adds new positions from signals."""
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
    results = [
        {"ticker": "AAPL", "status": "ok", "close": 152.0, "date": "2024-01-16", "atr": 3.0},
        {"ticker": "MSFT", "status": "ok", "close": 400.0, "date": "2024-01-16"},
    ]
    updated = update_existing_positions(state, results)
    # Only AAPL remains — MSFT never added even though it's in results
    assert len(updated["positions"]) == 1
    assert updated["positions"][0]["ticker"] == "AAPL"
    assert updated["positions"][0]["current_price"] == 152.0


def test_update_existing_triggers_exit(tmp_models_dir):
    """update_existing_positions closes a position on stop hit."""
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
    results = [{"ticker": "AAPL", "status": "ok", "close": 144.0, "date": "2024-01-16", "atr": 3.0}]
    updated = update_existing_positions(state, results)
    assert len(updated["positions"]) == 0
    assert len(updated["closed_today"]) == 1
    assert updated["closed_today"][0]["exit_reason"] == "stop"


def test_add_signal_positions_basic(tmp_models_dir):
    """add_signal_positions adds new positions from signals."""
    state = {"positions": [], "closed_today": []}
    signals = [{
        "ticker": "MSFT",
        "date": "2024-01-15",
        "close": 400.0,
        "size": 0.08,
        "range_low": 390.0,
        "atr_pct": 0.02,
    }]
    updated = add_signal_positions(state, signals)
    assert len(updated["positions"]) == 1
    assert updated["positions"][0]["ticker"] == "MSFT"
    assert updated["positions"][0]["entry_price"] == 400.0


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


def test_add_signal_positions_no_duplicates(tmp_models_dir):
    """add_signal_positions skips tickers already in open positions."""
    state = {
        "positions": [{
            "ticker": "AAPL",
            "entry_price": 150.0,
            "size": 0.10,
            "status": "open",
            "entry_date": "2024-01-10",
        }],
        "closed_today": [],
    }
    signals = [{
        "ticker": "AAPL",
        "date": "2024-01-15",
        "close": 155.0,
        "size": 0.12,
        "range_low": 148.0,
        "atr_pct": 0.02,
    }]
    updated = add_signal_positions(state, signals)
    assert len(updated["positions"]) == 1


def test_update_positions_wrapper(tmp_models_dir):
    """update_positions wrapper calls both split functions."""
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
    signals = [{
        "ticker": "MSFT",
        "date": "2024-01-15",
        "close": 400.0,
        "size": 0.08,
        "range_low": 390.0,
        "atr_pct": 0.02,
    }]
    results = [
        {"ticker": "AAPL", "status": "ok", "close": 152.0, "date": "2024-01-15", "atr": 3.0},
        {"ticker": "MSFT", "status": "ok", "close": 400.0, "date": "2024-01-15"},
    ]
    updated = update_positions(state, signals, results)
    # Both AAPL (updated) and MSFT (new) should be present
    tickers = {p["ticker"] for p in updated["positions"]}
    assert tickers == {"AAPL", "MSFT"}


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

    with patch("rally.bot.alpaca_executor.get_all_positions", mock_broker):
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


@pytest.mark.asyncio
async def test_merged_positions_no_metadata(tmp_models_dir):
    """Positions at Alpaca without DB metadata get default values."""
    mock_broker = AsyncMock(return_value=[{
        "ticker": "TSLA",
        "qty": 5,
        "avg_entry_price": 200.0,
        "market_value": 1050.0,
        "unrealized_pl": 50.0,
    }])

    with patch("rally.bot.alpaca_executor.get_all_positions", mock_broker):
        state = await get_merged_positions()

    assert len(state["positions"]) == 1
    pos = state["positions"][0]
    assert pos["ticker"] == "TSLA"
    assert pos["stop_price"] == 0  # default — no metadata
    assert pos["target_price"] == 0
    assert pos["entry_price"] == 200.0  # from Alpaca


def test_db_persistence_roundtrip(tmp_models_dir):
    """Positions written to DB can be read back with correct values."""
    from rally.trading.positions import load_position_meta, save_position_meta

    save_position_meta({
        "ticker": "GOOG",
        "entry_price": 180.0,
        "entry_date": "2024-02-01",
        "stop_price": 175.0,
        "target_price": 195.0,
        "trailing_stop": 177.0,
        "highest_close": 182.0,
        "atr": 2.5,
        "bars_held": 2,
        "size": 0.07,
        "qty": 15,
        "order_id": "ord_123",
        "trail_order_id": "trail_456",
    })

    loaded = load_position_meta("GOOG")
    assert loaded is not None
    assert loaded["ticker"] == "GOOG"
    assert loaded["entry_price"] == 180.0
    assert loaded["stop_price"] == 175.0
    assert loaded["order_id"] == "ord_123"
    assert loaded["trail_order_id"] == "trail_456"
    assert loaded["status"] == "open"


# ---------------------------------------------------------------------------
# Remote fetch (RALLY_API_URL)
# ---------------------------------------------------------------------------

def test_remote_fetch_when_api_url_set(tmp_models_dir, monkeypatch):
    """get_merged_positions_sync uses Railway API when RALLY_API_URL is set."""
    from rally.trading.positions import get_merged_positions_sync

    monkeypatch.setenv("RALLY_API_URL", "https://my-railway.app")
    monkeypatch.setenv("RALLY_API_KEY", "secret123")

    remote_data = {"positions": [{"ticker": "AAPL", "qty": 10}], "closed_today": []}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(remote_data).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        result = get_merged_positions_sync()

    assert result == remote_data
    # Verify auth header was set
    req = mock_open.call_args[0][0]
    assert req.get_header("Authorization") == "Bearer secret123"
    assert req.full_url == "https://my-railway.app/api/positions"


def test_remote_fetch_fallback_on_error(tmp_models_dir, monkeypatch):
    """Falls back to local DB when Railway API is unreachable."""
    from rally.trading.positions import get_merged_positions_sync

    monkeypatch.setenv("RALLY_API_URL", "https://unreachable.app")

    with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
        result = get_merged_positions_sync()

    assert "positions" in result
    assert isinstance(result["positions"], list)
