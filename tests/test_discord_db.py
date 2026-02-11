"""Tests for Discord SQLite database layer."""

import pytest

from rally.discord_db import (
    close_trade,
    ensure_user,
    get_db,
    get_open_trades,
    get_pnl_summary,
    get_trade_history,
    init_db,
    open_trade,
)


@pytest.fixture
def db(tmp_path):
    """Create an in-memory-like temp DB for each test."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return db_path


def test_init_creates_tables(db):
    conn = get_db(db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {r["name"] for r in tables}
    assert "users" in names
    assert "trades" in names


def test_ensure_user_creates(db):
    ensure_user(12345, "alice", db_path=db)
    conn = get_db(db)
    row = conn.execute("SELECT * FROM users WHERE discord_id = 12345").fetchone()
    assert row["username"] == "alice"


def test_ensure_user_updates_username(db):
    ensure_user(12345, "alice", db_path=db)
    ensure_user(12345, "alice_renamed", db_path=db)
    conn = get_db(db)
    row = conn.execute("SELECT * FROM users WHERE discord_id = 12345").fetchone()
    assert row["username"] == "alice_renamed"


def test_ensure_user_no_duplicate(db):
    ensure_user(12345, "alice", db_path=db)
    ensure_user(12345, "alice", db_path=db)
    conn = get_db(db)
    count = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
    assert count == 1


def test_open_trade_returns_id(db):
    ensure_user(1, "bob", db_path=db)
    trade_id = open_trade(1, "AAPL", 150.0, "2024-01-10", db_path=db)
    assert trade_id == 1


def test_open_trade_stored_correctly(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "AAPL", 150.0, "2024-01-10", size=0.5, notes="test", db_path=db)
    trades = get_open_trades(1, db_path=db)
    assert len(trades) == 1
    assert trades[0]["ticker"] == "AAPL"
    assert trades[0]["entry_price"] == 150.0
    assert trades[0]["size"] == 0.5
    assert trades[0]["notes"] == "test"


def test_open_trade_uppercases_ticker(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "aapl", 150.0, "2024-01-10", db_path=db)
    trades = get_open_trades(1, db_path=db)
    assert trades[0]["ticker"] == "AAPL"


def test_close_trade_fifo(db):
    """Closing should close the oldest open trade for the ticker."""
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "AAPL", 150.0, "2024-01-10", db_path=db)
    open_trade(1, "AAPL", 160.0, "2024-01-15", db_path=db)

    result = close_trade(1, "AAPL", 155.0, "2024-01-20", db_path=db)

    assert result is not None
    assert result["entry_price"] == 150.0  # first one closed
    assert result["pnl_pct"] == pytest.approx(3.33, abs=0.01)

    # Second one still open
    remaining = get_open_trades(1, db_path=db)
    assert len(remaining) == 1
    assert remaining[0]["entry_price"] == 160.0


def test_close_trade_no_open(db):
    ensure_user(1, "bob", db_path=db)
    result = close_trade(1, "MSFT", 200.0, db_path=db)
    assert result is None


def test_close_trade_pnl_calculation(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "NVDA", 500.0, "2024-01-10", db_path=db)
    result = close_trade(1, "NVDA", 550.0, "2024-01-20", db_path=db)

    assert result["pnl_pct"] == 10.0  # (550/500 - 1) * 100


def test_close_trade_negative_pnl(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "TSLA", 200.0, "2024-01-10", db_path=db)
    result = close_trade(1, "TSLA", 180.0, "2024-01-20", db_path=db)

    assert result["pnl_pct"] == -10.0


def test_get_trade_history_all(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "AAPL", 150.0, "2024-01-10", db_path=db)
    open_trade(1, "MSFT", 400.0, "2024-01-12", db_path=db)
    close_trade(1, "AAPL", 155.0, "2024-01-20", db_path=db)

    history = get_trade_history(1, db_path=db)
    assert len(history) == 2


def test_get_trade_history_filtered(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "AAPL", 150.0, "2024-01-10", db_path=db)
    open_trade(1, "MSFT", 400.0, "2024-01-12", db_path=db)

    history = get_trade_history(1, ticker="AAPL", db_path=db)
    assert len(history) == 1
    assert history[0]["ticker"] == "AAPL"


def test_get_trade_history_limit(db):
    ensure_user(1, "bob", db_path=db)
    for i in range(10):
        open_trade(1, "AAPL", 150.0 + i, f"2024-01-{10+i:02d}", db_path=db)

    history = get_trade_history(1, limit=3, db_path=db)
    assert len(history) == 3


def test_pnl_summary_empty(db):
    ensure_user(1, "bob", db_path=db)
    summary = get_pnl_summary(1, db_path=db)
    assert summary["n_trades"] == 0
    assert summary["total_pnl"] == 0.0
    assert summary["win_rate"] == 0.0


def test_pnl_summary_mixed(db):
    ensure_user(1, "bob", db_path=db)
    # Win: +10%
    open_trade(1, "AAPL", 100.0, "2024-01-10", db_path=db)
    close_trade(1, "AAPL", 110.0, "2024-01-20", db_path=db)
    # Loss: -5%
    open_trade(1, "MSFT", 200.0, "2024-01-10", db_path=db)
    close_trade(1, "MSFT", 190.0, "2024-01-20", db_path=db)
    # Win: +20%
    open_trade(1, "NVDA", 500.0, "2024-01-10", db_path=db)
    close_trade(1, "NVDA", 600.0, "2024-01-20", db_path=db)

    summary = get_pnl_summary(1, db_path=db)
    assert summary["n_trades"] == 3
    assert summary["total_pnl"] == pytest.approx(25.0, abs=0.01)
    assert summary["win_rate"] == pytest.approx(66.7, abs=0.1)
    assert summary["best_trade"] == pytest.approx(20.0, abs=0.01)
    assert summary["worst_trade"] == pytest.approx(-5.0, abs=0.01)


def test_pnl_summary_ignores_open(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "AAPL", 100.0, "2024-01-10", db_path=db)  # open, not closed
    summary = get_pnl_summary(1, db_path=db)
    assert summary["n_trades"] == 0


def test_multiple_users_isolated(db):
    ensure_user(1, "alice", db_path=db)
    ensure_user(2, "bob", db_path=db)

    open_trade(1, "AAPL", 150.0, "2024-01-10", db_path=db)
    open_trade(2, "MSFT", 400.0, "2024-01-12", db_path=db)

    alice_trades = get_open_trades(1, db_path=db)
    bob_trades = get_open_trades(2, db_path=db)

    assert len(alice_trades) == 1
    assert alice_trades[0]["ticker"] == "AAPL"
    assert len(bob_trades) == 1
    assert bob_trades[0]["ticker"] == "MSFT"
