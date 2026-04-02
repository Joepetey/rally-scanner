"""Tests for the database persistence layer (PostgreSQL)."""

import pytest

from db.ops.conversations import (
    clear_conversation_history,
    get_conversation_history,
    save_conversation_history,
)
from db.ops.users import ensure_user, get_capital, set_capital
from db.trading.trades import (
    close_trade,
    get_open_trades,
    get_pnl_summary,
    get_trade_history,
    open_trade,
)

# pg_db fixture from conftest.py handles init + truncate


def test_init_creates_tables(pg_db):
    from db.core.pool import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        names = {r["tablename"] for r in cur.fetchall()}
    assert "users" in names
    assert "trades" in names


def test_ensure_user_creates(pg_db):
    ensure_user(12345, "alice")
    from db.core.pool import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE discord_id = 12345")
        row = cur.fetchone()
    assert row["username"] == "alice"


def test_ensure_user_updates_username(pg_db):
    ensure_user(12345, "alice")
    ensure_user(12345, "alice_renamed")
    from db.core.pool import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE discord_id = 12345")
        row = cur.fetchone()
    assert row["username"] == "alice_renamed"


def test_ensure_user_no_duplicate(pg_db):
    ensure_user(12345, "alice")
    ensure_user(12345, "alice")
    from db.core.pool import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM users")
        count = cur.fetchone()["c"]
    assert count == 1


def test_open_trade_returns_id(pg_db):
    trade_id = open_trade("AAPL", 150.0, "2024-01-10")
    assert isinstance(trade_id, int)
    assert trade_id > 0


def test_open_trade_stored_correctly(pg_db):
    open_trade("AAPL", 150.0, "2024-01-10", size=0.5, notes="test")
    trades = get_open_trades()
    assert len(trades) == 1
    assert trades[0]["ticker"] == "AAPL"
    assert trades[0]["entry_price"] == 150.0
    assert trades[0]["size"] == 0.5
    assert trades[0]["notes"] == "test"


def test_close_trade_fifo(pg_db):
    """Closing should close the oldest open trade for the ticker."""
    open_trade("AAPL", 150.0, "2024-01-10")
    open_trade("AAPL", 160.0, "2024-01-15")

    result = close_trade("AAPL", 155.0, "2024-01-20")

    assert result is not None
    assert result["entry_price"] == 150.0  # first one closed
    assert result["pnl_pct"] == pytest.approx(3.33, abs=0.01)

    remaining = get_open_trades()
    assert len(remaining) == 1
    assert remaining[0]["entry_price"] == 160.0


def test_close_trade_no_open(pg_db):
    result = close_trade("MSFT", 200.0)
    assert result is None


def test_close_trade_pnl_calculation(pg_db):
    open_trade("NVDA", 500.0, "2024-01-10")
    result = close_trade("NVDA", 550.0, "2024-01-20")
    assert result["pnl_pct"] == 10.0


def test_get_trade_history_all(pg_db):
    open_trade("AAPL", 150.0, "2024-01-10")
    open_trade("MSFT", 400.0, "2024-01-12")
    close_trade("AAPL", 155.0, "2024-01-20")
    history = get_trade_history()
    assert len(history) == 2


def test_get_trade_history_filtered(pg_db):
    open_trade("AAPL", 150.0, "2024-01-10")
    open_trade("MSFT", 400.0, "2024-01-12")
    history = get_trade_history(ticker="AAPL")
    assert len(history) == 1
    assert history[0]["ticker"] == "AAPL"


def test_pnl_summary_mixed(pg_db):
    open_trade("AAPL", 100.0, "2024-01-10")
    close_trade("AAPL", 110.0, "2024-01-20")
    open_trade("MSFT", 200.0, "2024-01-10")
    close_trade("MSFT", 190.0, "2024-01-20")
    open_trade("NVDA", 500.0, "2024-01-10")
    close_trade("NVDA", 600.0, "2024-01-20")

    summary = get_pnl_summary()
    assert summary["n_trades"] == 3
    assert summary["total_pnl"] == pytest.approx(25.0, abs=0.01)
    assert summary["win_rate"] == pytest.approx(66.7, abs=0.1)
    assert summary["best_trade"] == pytest.approx(20.0, abs=0.01)
    assert summary["worst_trade"] == pytest.approx(-5.0, abs=0.01)


# ---------------------------------------------------------------------------
# Capital management
# ---------------------------------------------------------------------------


def test_set_and_get_capital(pg_db):
    ensure_user(1, "alice")
    set_capital(1, 100_000.0)
    assert get_capital(1) == 100_000.0


def test_open_trade_with_stop_target(pg_db):
    open_trade("AAPL", 150.0, "2024-01-10", stop_price=145.0, target_price=160.0)
    trades = get_open_trades()
    assert trades[0]["stop_price"] == 145.0
    assert trades[0]["target_price"] == 160.0


# ---------------------------------------------------------------------------
# Dollar PnL
# ---------------------------------------------------------------------------


def test_close_trade_dollar_pnl_with_capital(pg_db):
    open_trade("AAPL", 100.0, "2024-01-10", size=0.15)
    result = close_trade("AAPL", 110.0, "2024-01-20", capital=100_000.0)
    assert result["pnl_pct"] == 10.0
    # Dollar PnL: 100k * 0.15 * 0.10 = $1500
    assert result["pnl_dollar"] == pytest.approx(1500.0, abs=0.01)


def test_pnl_summary_includes_dollar_total(pg_db):
    open_trade("AAPL", 100.0, "2024-01-10", size=0.15)
    close_trade("AAPL", 110.0, "2024-01-20", capital=100_000.0)
    open_trade("MSFT", 200.0, "2024-01-10", size=0.10)
    close_trade("MSFT", 190.0, "2024-01-20", capital=100_000.0)
    summary = get_pnl_summary()
    # AAPL: 100k * 0.15 * 10% = +$1500, MSFT: 100k * 0.10 * -5% = -$500
    assert summary["total_pnl_dollar"] == pytest.approx(1000.0, abs=0.01)


# ---------------------------------------------------------------------------
# Conversation history — truncation safety
# ---------------------------------------------------------------------------


def _tool_use_msg():
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_abc123", "name": "get_signals",
             "input": {}},
        ],
    }


def _tool_result_msg():
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


def test_conversation_history_round_trip(pg_db):
    ensure_user(1, "alice")
    history = [_user_msg(), _asst_msg()]
    save_conversation_history(1, history)
    got = get_conversation_history(1)
    assert got == history


def test_conversation_truncation_skips_orphan_tool_result(pg_db):
    ensure_user(1, "alice")
    history = [
        _user_msg("first"),
        _tool_use_msg(),
        _tool_result_msg(),
        _asst_msg("tool done"),
        _user_msg("second"),
        _asst_msg("reply"),
    ]
    save_conversation_history(1, history)
    got = get_conversation_history(1, limit_messages=4)
    assert got[0]["role"] == "user"
    assert got[0]["content"] == "second"


def test_conversation_truncation_skips_assistant_start(pg_db):
    ensure_user(1, "alice")
    history = [
        _user_msg("old"),
        _asst_msg("old reply"),
        _user_msg("new"),
        _asst_msg("new reply"),
    ]
    save_conversation_history(1, history)
    got = get_conversation_history(1, limit_messages=3)
    assert got[0]["role"] == "user"
    assert got[0]["content"] == "new"


def test_conversation_no_truncation_when_small(pg_db):
    ensure_user(1, "alice")
    history = [_user_msg(), _asst_msg()]
    save_conversation_history(1, history)
    got = get_conversation_history(1, limit_messages=20)
    assert got == history


def test_conversation_truncation_no_safe_start_returns_empty(pg_db):
    ensure_user(1, "alice")
    history = [
        _tool_use_msg(),
        _tool_result_msg(),
        _asst_msg("done"),
    ]
    save_conversation_history(1, history)
    got = get_conversation_history(1, limit_messages=3)
    assert got == []


def test_clear_conversation_history(pg_db):
    ensure_user(1, "alice")
    save_conversation_history(1, [_user_msg()])
    clear_conversation_history(1)
    got = get_conversation_history(1)
    assert got == []
