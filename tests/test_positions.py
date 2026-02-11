"""Tests for position tracking — load, save, update, exit logic."""

import json

from rally.positions import load_positions, save_positions, update_positions
from rally.config import PARAMS


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
