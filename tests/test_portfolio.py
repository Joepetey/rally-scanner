"""Tests for portfolio tracking — CSV round-trip."""

from rally.portfolio import (
    update_daily_snapshot,
    record_closed_trades,
    load_equity_history,
    load_trade_journal,
)


def test_equity_snapshot_creates_csv(tmp_models_dir):
    positions_state = {
        "positions": [
            {"size": 0.10, "unrealized_pnl_pct": 2.5},
            {"size": 0.05, "unrealized_pnl_pct": -1.0},
        ]
    }
    scan_results = [
        {"signal": True, "status": "ok"},
        {"signal": False, "status": "ok"},
        {"signal": False, "status": "error"},
    ]
    snapshot = update_daily_snapshot(positions_state, scan_results)
    assert snapshot["n_positions"] == 2
    assert snapshot["n_signals_today"] == 1
    assert snapshot["n_scanned"] == 2

    history = load_equity_history()
    assert len(history) == 1
    assert history[0]["n_positions"] == "2"  # CSV reads as strings


def test_equity_history_limit(tmp_models_dir):
    positions_state = {"positions": []}
    scan_results = []
    for _ in range(5):
        update_daily_snapshot(positions_state, scan_results)

    assert len(load_equity_history()) == 5
    assert len(load_equity_history(days=3)) == 3


def test_record_closed_trades(tmp_models_dir):
    trades = [
        {
            "ticker": "AAPL",
            "entry_date": "2024-01-10",
            "entry_price": 150.0,
            "exit_price": 156.0,
            "realized_pnl_pct": 4.0,
            "size": 0.10,
            "bars_held": 5,
            "exit_reason": "profit_target",
        },
        {
            "ticker": "MSFT",
            "entry_date": "2024-01-12",
            "entry_price": 400.0,
            "exit_price": 395.0,
            "realized_pnl_pct": -1.25,
            "size": 0.08,
            "bars_held": 3,
            "exit_reason": "stop",
        },
    ]
    record_closed_trades(trades)
    journal = load_trade_journal()
    assert len(journal) == 2
    assert journal[0]["ticker"] == "AAPL"
    assert journal[1]["exit_reason"] == "stop"


def test_record_empty_trades(tmp_models_dir):
    record_closed_trades([])
    journal = load_trade_journal()
    assert len(journal) == 0


def test_trade_journal_limit(tmp_models_dir):
    trades = [
        {
            "ticker": f"T{i}",
            "entry_date": f"2024-01-{10+i:02d}",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "realized_pnl_pct": 5.0,
            "size": 0.05,
            "bars_held": 3,
            "exit_reason": "profit_target",
        }
        for i in range(10)
    ]
    record_closed_trades(trades)
    assert len(load_trade_journal()) == 10
    assert len(load_trade_journal(limit=5)) == 5


def test_multiple_snapshots_append(tmp_models_dir):
    for i in range(3):
        positions_state = {
            "positions": [{"size": 0.10, "unrealized_pnl_pct": float(i)}]
        }
        update_daily_snapshot(positions_state, [])

    history = load_equity_history()
    assert len(history) == 3
