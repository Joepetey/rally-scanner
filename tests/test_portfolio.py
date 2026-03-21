"""Tests for portfolio tracking — DB round-trip."""

from datetime import date, timedelta

import pytest

from db.portfolio import save_snapshot
from trading.portfolio import (
    load_equity_history,
    load_trade_journal,
    record_closed_trades,
    update_daily_snapshot,
)


def test_equity_snapshot_persists(pg_db):
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
    assert history[0]["n_positions"] == 2  # native int (not string)


def test_equity_history_limit(pg_db):
    """Insert 5 snapshots with different dates and verify limit/count behaviour."""
    today = date.today()
    for i in range(5):
        day = today - timedelta(days=4 - i)
        save_snapshot({
            "date": day.isoformat(),
            "n_positions": i,
            "total_exposure": 0.0,
            "total_unrealized_pnl_pct": 0.0,
            "n_signals_today": 0,
            "n_scanned": 0,
        })

    assert len(load_equity_history()) == 5
    assert len(load_equity_history(days=3)) == 3


def test_record_closed_trades(pg_db):
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
    # load_trade_journal returns chronological order (oldest first)
    tickers = [r["ticker"] for r in journal]
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    stop_entry = next(r for r in journal if r["ticker"] == "MSFT")
    assert stop_entry["exit_reason"] == "stop"


def test_record_empty_trades(pg_db):
    record_closed_trades([])
    journal = load_trade_journal()
    assert len(journal) == 0


def test_trade_journal_limit(pg_db):
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


def test_multiple_snapshots_distinct_dates(pg_db):
    """Each snapshot date is unique; multiple calls on same date upsert."""
    today = date.today()
    for i in range(3):
        day = today - timedelta(days=2 - i)
        save_snapshot({
            "date": day.isoformat(),
            "n_positions": i + 1,
            "total_exposure": float(i) * 0.1,
            "total_unrealized_pnl_pct": 0.0,
            "n_signals_today": 0,
            "n_scanned": 0,
        })

    history = load_equity_history()
    assert len(history) == 3
