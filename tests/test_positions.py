"""Tests for position tracking — load, save, update, exit logic."""


from rally.config import PARAMS
from rally.trading.positions import (
    add_signal_positions,
    load_positions,
    reconcile_with_broker,
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
    state = {"positions": [], "closed_today": []}
    save_positions(state)
    result = tighten_trailing_stop("UNKNOWN", 150.0)
    assert result is None


# ---------------------------------------------------------------------------
# reconcile_with_broker
# ---------------------------------------------------------------------------


def test_reconcile_removes_ghost_positions(tmp_models_dir):
    """Ghost positions (local but not at broker) should be auto-removed."""
    state = {
        "positions": [
            {"ticker": "AAPL", "entry_price": 150.0, "status": "open"},
            {"ticker": "GHOST1", "entry_price": 50.0, "status": "open"},
            {"ticker": "GHOST2", "entry_price": 25.0, "status": "open"},
        ],
        "closed_today": [],
    }
    save_positions(state)

    broker = [{"ticker": "AAPL", "qty": 10}]
    warnings = reconcile_with_broker(broker, trail_fills={})

    assert len(warnings) == 2
    assert any("GHOST1" in w for w in warnings)
    assert any("GHOST2" in w for w in warnings)

    # Verify ghosts removed from disk
    reloaded = load_positions()
    tickers = [p["ticker"] for p in reloaded["positions"]]
    assert tickers == ["AAPL"]


def test_reconcile_detects_orphaned_positions(tmp_models_dir):
    """Orphaned positions (at broker but not local) should be warned about."""
    state = {"positions": [], "closed_today": []}
    save_positions(state)

    broker = [{"ticker": "MSFT", "qty": 5}]
    warnings = reconcile_with_broker(broker, trail_fills={})

    assert len(warnings) == 1
    assert "Orphaned" in warnings[0]
    assert "MSFT" in warnings[0]


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
