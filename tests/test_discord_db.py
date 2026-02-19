"""Tests for Discord SQLite database layer."""

import pytest

from rally.bot.discord_db import (
    clear_conversation_history,
    close_trade,
    ensure_user,
    get_capital,
    get_conversation_history,
    get_db,
    get_open_trades,
    get_pnl_summary,
    get_trade_history,
    init_db,
    open_trade,
    save_conversation_history,
    set_capital,
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


# ---------------------------------------------------------------------------
# Capital management
# ---------------------------------------------------------------------------


def test_capital_default_zero(db):
    ensure_user(1, "alice", db_path=db)
    assert get_capital(1, db_path=db) == 0.0


def test_set_and_get_capital(db):
    ensure_user(1, "alice", db_path=db)
    set_capital(1, 100_000.0, db_path=db)
    assert get_capital(1, db_path=db) == 100_000.0


def test_update_capital(db):
    ensure_user(1, "alice", db_path=db)
    set_capital(1, 50_000.0, db_path=db)
    set_capital(1, 75_000.0, db_path=db)
    assert get_capital(1, db_path=db) == 75_000.0


# ---------------------------------------------------------------------------
# Stop/target on trades
# ---------------------------------------------------------------------------


def test_open_trade_with_stop_target(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(
        1, "AAPL", 150.0, "2024-01-10",
        stop_price=145.0, target_price=160.0, db_path=db,
    )
    trades = get_open_trades(1, db_path=db)
    assert trades[0]["stop_price"] == 145.0
    assert trades[0]["target_price"] == 160.0


def test_open_trade_stop_target_default_none(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "AAPL", 150.0, "2024-01-10", db_path=db)
    trades = get_open_trades(1, db_path=db)
    assert trades[0]["stop_price"] is None
    assert trades[0]["target_price"] is None


# ---------------------------------------------------------------------------
# Dollar PnL
# ---------------------------------------------------------------------------


def test_close_trade_dollar_pnl_with_capital(db):
    ensure_user(1, "bob", db_path=db)
    set_capital(1, 100_000.0, db_path=db)
    open_trade(1, "AAPL", 100.0, "2024-01-10", size=0.15, db_path=db)
    result = close_trade(1, "AAPL", 110.0, "2024-01-20", db_path=db)

    assert result["pnl_pct"] == 10.0
    # Dollar PnL: 100k * 0.15 * 0.10 = $1500
    assert result["pnl_dollar"] == pytest.approx(1500.0, abs=0.01)


def test_close_trade_dollar_pnl_without_capital(db):
    ensure_user(1, "bob", db_path=db)
    # No capital set
    open_trade(1, "AAPL", 100.0, "2024-01-10", size=0.15, db_path=db)
    result = close_trade(1, "AAPL", 110.0, "2024-01-20", db_path=db)

    assert result["pnl_pct"] == 10.0
    assert result["pnl_dollar"] is None


def test_pnl_summary_includes_dollar_total(db):
    ensure_user(1, "bob", db_path=db)
    set_capital(1, 100_000.0, db_path=db)

    open_trade(1, "AAPL", 100.0, "2024-01-10", size=0.15, db_path=db)
    close_trade(1, "AAPL", 110.0, "2024-01-20", db_path=db)

    open_trade(1, "MSFT", 200.0, "2024-01-10", size=0.10, db_path=db)
    close_trade(1, "MSFT", 190.0, "2024-01-20", db_path=db)

    summary = get_pnl_summary(1, db_path=db)
    # AAPL: 100k * 0.15 * 10% = +$1500, MSFT: 100k * 0.10 * -5% = -$500
    assert summary["total_pnl_dollar"] == pytest.approx(1000.0, abs=0.01)


def test_pnl_summary_dollar_zero_when_no_capital(db):
    ensure_user(1, "bob", db_path=db)
    open_trade(1, "AAPL", 100.0, "2024-01-10", db_path=db)
    close_trade(1, "AAPL", 110.0, "2024-01-20", db_path=db)

    summary = get_pnl_summary(1, db_path=db)
    assert summary["total_pnl_dollar"] == 0.0


# ---------------------------------------------------------------------------
# Migration (existing DB without new columns)
# ---------------------------------------------------------------------------


def test_migration_adds_columns(db):
    """Simulate a pre-migration DB and verify columns are added."""
    conn = get_db(db)
    # Columns should exist after init_db (which calls _migrate)
    user_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    trade_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()
    }
    assert "capital" in user_cols
    assert "stop_price" in trade_cols
    assert "target_price" in trade_cols
    assert "pnl_dollar" in trade_cols


# ---------------------------------------------------------------------------
# Conversation history — truncation safety
# ---------------------------------------------------------------------------


def _tool_use_msg():
    """Assistant message containing a tool_use block."""
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_abc123", "name": "get_signals",
             "input": {}},
        ],
    }


def _tool_result_msg():
    """User message containing a tool_result block."""
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_abc123",
             "content": '{"signals": []}'},
        ],
    }


def _user_msg(text="hello"):
    return {"role": "user", "content": text}


def _asst_msg(text="Hi there"):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def test_conversation_history_round_trip(db):
    """Save and retrieve basic conversation history."""
    ensure_user(1, "alice", db_path=db)
    history = [_user_msg(), _asst_msg()]
    save_conversation_history(1, history, db_path=db)
    got = get_conversation_history(1, db_path=db)
    assert got == history


def test_conversation_truncation_skips_orphan_tool_result(db):
    """Truncation must not start on an orphaned tool_result message."""
    ensure_user(1, "alice", db_path=db)
    # Build history: plain user, assistant tool_use, tool_result, assistant text, plain user, assistant text
    history = [
        _user_msg("first"),           # 0
        _tool_use_msg(),              # 1 — assistant with tool_use
        _tool_result_msg(),           # 2 — user with tool_result
        _asst_msg("tool done"),       # 3
        _user_msg("second"),          # 4
        _asst_msg("reply"),           # 5
    ]
    save_conversation_history(1, history, db_path=db)

    # Limit=4 would naively slice to [2,3,4,5] starting on tool_result
    got = get_conversation_history(1, limit_messages=4, db_path=db)

    # Should skip the orphaned tool_result and start at msg 4 (plain user)
    assert got[0]["role"] == "user"
    assert got[0]["content"] == "second"


def test_conversation_truncation_skips_assistant_start(db):
    """History must not start with an assistant message."""
    ensure_user(1, "alice", db_path=db)
    history = [
        _user_msg("old"),
        _asst_msg("old reply"),
        _user_msg("new"),
        _asst_msg("new reply"),
    ]
    save_conversation_history(1, history, db_path=db)

    # limit=3 slices to [assistant "old reply", user "new", assistant "new reply"]
    got = get_conversation_history(1, limit_messages=3, db_path=db)

    # Should start at the first plain user message in the slice
    assert got[0]["role"] == "user"
    assert got[0]["content"] == "new"


def test_conversation_no_truncation_when_small(db):
    """Short history is returned as-is."""
    ensure_user(1, "alice", db_path=db)
    history = [_user_msg(), _asst_msg()]
    save_conversation_history(1, history, db_path=db)
    got = get_conversation_history(1, limit_messages=20, db_path=db)
    assert got == history


def test_conversation_truncation_no_safe_start_returns_empty(db):
    """If no plain user message exists in the slice, return empty."""
    ensure_user(1, "alice", db_path=db)
    # All tool interactions — no plain user message
    history = [
        _tool_use_msg(),
        _tool_result_msg(),
        _asst_msg("done"),
    ]
    save_conversation_history(1, history, db_path=db)
    got = get_conversation_history(1, limit_messages=3, db_path=db)
    assert got == []


def test_clear_conversation_history(db):
    ensure_user(1, "alice", db_path=db)
    save_conversation_history(1, [_user_msg()], db_path=db)
    clear_conversation_history(1, db_path=db)
    got = get_conversation_history(1, db_path=db)
    assert got == []
