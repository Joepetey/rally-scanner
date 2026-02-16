"""Tests for Alpaca MCP executor and order embeds."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rally.alpaca_executor import (
    OrderResult,
    check_pending_fills,
    execute_entries,
    execute_exit,
    execute_exits,
    get_account_equity,
    get_snapshots,
    is_enabled,
)
from rally.notify import _order_embed, _order_failure_embed
from rally.positions import close_position_intraday, update_fill_prices


# ---------------------------------------------------------------------------
# is_enabled()
# ---------------------------------------------------------------------------


def test_is_enabled_true(monkeypatch):
    monkeypatch.setenv("ALPACA_AUTO_EXECUTE", "1")
    assert is_enabled() is True


def test_is_enabled_false(monkeypatch):
    monkeypatch.delenv("ALPACA_AUTO_EXECUTE", raising=False)
    assert is_enabled() is False


def test_is_enabled_wrong_value(monkeypatch):
    monkeypatch.setenv("ALPACA_AUTO_EXECUTE", "0")
    assert is_enabled() is False


# ---------------------------------------------------------------------------
# OrderResult model
# ---------------------------------------------------------------------------


def test_order_result_success():
    r = OrderResult(
        ticker="AAPL", side="buy", qty=10, success=True,
        order_id="abc123", fill_price=150.25,
    )
    assert r.ticker == "AAPL"
    assert r.fill_price == 150.25


def test_order_result_failure():
    r = OrderResult(
        ticker="AAPL", side="buy", qty=10, success=False,
        error="Insufficient buying power",
    )
    assert r.success is False
    assert r.order_id is None
    assert r.fill_price is None


# ---------------------------------------------------------------------------
# _order_embed / _order_failure_embed
# ---------------------------------------------------------------------------


def test_order_embed():
    results = [
        OrderResult(ticker="AAPL", side="buy", qty=10, success=True,
                    order_id="abc12345", fill_price=150.25),
        OrderResult(ticker="MSFT", side="sell", qty=5, success=True,
                    order_id="def67890"),
    ]
    embed = _order_embed(results, equity=100_000.0)
    assert embed["title"] == "Alpaca Orders (2)"
    assert embed["color"] == 0x87CEEB
    assert "$150.25" in embed["fields"][0]["value"]
    assert "abc12345" in embed["fields"][0]["value"]
    assert "$100,000" in embed["footer"]["text"]


def test_order_failure_embed():
    results = [
        OrderResult(ticker="TSLA", side="buy", qty=5, success=False,
                    error="Insufficient buying power"),
    ]
    embed = _order_failure_embed(results)
    assert embed["title"] == "Alpaca Order Failures (1)"
    assert embed["color"] == 0xFF4500
    assert "Insufficient buying power" in embed["fields"][0]["value"]
    assert "unaffected" in embed["footer"]["text"].lower()


# ---------------------------------------------------------------------------
# close_position_intraday
# ---------------------------------------------------------------------------


def test_close_position_intraday(tmp_path, monkeypatch):
    positions_file = tmp_path / "models" / "positions.json"
    positions_file.parent.mkdir(parents=True)
    state = {
        "positions": [
            {"ticker": "AAPL", "entry_price": 150.0, "current_price": 145.0,
             "stop_price": 145.0, "target_price": 165.0, "status": "open"},
            {"ticker": "MSFT", "entry_price": 400.0, "current_price": 410.0,
             "stop_price": 390.0, "target_price": 420.0, "status": "open"},
        ],
        "closed_today": [],
        "last_updated": None,
    }
    positions_file.write_text(json.dumps(state))

    monkeypatch.setattr("rally.positions.POSITIONS_FILE", positions_file)

    closed = close_position_intraday("AAPL", 144.50, "stop")
    assert closed is not None
    assert closed["ticker"] == "AAPL"
    assert closed["exit_price"] == 144.50
    assert closed["exit_reason"] == "stop"
    assert closed["realized_pnl_pct"] == pytest.approx(-3.67, abs=0.01)
    assert closed["status"] == "closed"

    # Verify positions.json updated
    reloaded = json.loads(positions_file.read_text())
    assert len(reloaded["positions"]) == 1
    assert reloaded["positions"][0]["ticker"] == "MSFT"
    assert len(reloaded["closed_today"]) == 1


def test_close_position_intraday_not_found(tmp_path, monkeypatch):
    positions_file = tmp_path / "models" / "positions.json"
    positions_file.parent.mkdir(parents=True)
    state = {"positions": [], "closed_today": [], "last_updated": None}
    positions_file.write_text(json.dumps(state))

    monkeypatch.setattr("rally.positions.POSITIONS_FILE", positions_file)

    result = close_position_intraday("AAPL", 144.50, "stop")
    assert result is None


# ---------------------------------------------------------------------------
# update_fill_prices
# ---------------------------------------------------------------------------


def test_update_fill_prices(tmp_path, monkeypatch):
    positions_file = tmp_path / "models" / "positions.json"
    positions_file.parent.mkdir(parents=True)
    state = {
        "positions": [
            {"ticker": "AAPL", "entry_price": 150.0, "current_price": 152.0,
             "order_id": "order-123", "unrealized_pnl_pct": 1.33, "status": "open"},
            {"ticker": "MSFT", "entry_price": 400.0, "current_price": 405.0,
             "status": "open"},
        ],
        "closed_today": [],
        "last_updated": None,
    }
    positions_file.write_text(json.dumps(state))
    monkeypatch.setattr("rally.positions.POSITIONS_FILE", positions_file)

    count = update_fill_prices({"order-123": 149.50})
    assert count == 1

    reloaded = json.loads(positions_file.read_text())
    aapl = reloaded["positions"][0]
    assert aapl["entry_price"] == 149.50
    assert "order_id" not in aapl
    # PnL recalculated: (152 / 149.50 - 1) * 100
    assert aapl["unrealized_pnl_pct"] == pytest.approx(1.67, abs=0.01)


def test_update_fill_prices_no_matches(tmp_path, monkeypatch):
    positions_file = tmp_path / "models" / "positions.json"
    positions_file.parent.mkdir(parents=True)
    state = {
        "positions": [
            {"ticker": "AAPL", "entry_price": 150.0, "current_price": 152.0,
             "status": "open"},
        ],
        "closed_today": [],
        "last_updated": None,
    }
    positions_file.write_text(json.dumps(state))
    monkeypatch.setattr("rally.positions.POSITIONS_FILE", positions_file)

    count = update_fill_prices({"order-999": 149.50})
    assert count == 0


# ---------------------------------------------------------------------------
# execute_entries (mocked MCP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_entries_skips_crypto():
    """Crypto tickers should be skipped."""
    signals = [
        {"ticker": "BTC", "entry_price": 50000.0, "size": 0.10},
    ]

    mock_session = AsyncMock()
    with patch("rally.alpaca_executor._mcp_session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 0
    mock_session.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_execute_entries_qty_calculation():
    """Verify qty = int(equity * size / price)."""
    signals = [
        {"ticker": "AAPL", "entry_price": 150.0, "size": 0.10},
    ]

    mock_session = AsyncMock()
    # Mock _call_tool to return order data
    order_response = MagicMock()
    order_response.content = [MagicMock(text=json.dumps({
        "id": "order-abc",
        "status": "accepted",
        "filled_avg_price": None,
    }))]
    mock_session.call_tool.return_value = order_response

    with patch("rally.alpaca_executor._mcp_session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    r = results[0]
    assert r.ticker == "AAPL"
    assert r.side == "buy"
    # 100000 * 0.10 / 150 = 66
    assert r.qty == 66
    assert r.success is True
    assert r.order_id == "order-abc"


# ---------------------------------------------------------------------------
# execute_exit (mocked MCP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_exit():
    """Verify close_position call and fill price extraction."""
    close_response = MagicMock()
    close_response.content = [MagicMock(text=json.dumps({
        "id": "order-exit-1",
        "qty": "10",
        "filled_avg_price": "148.50",
    }))]

    mock_session = AsyncMock()
    mock_session.call_tool.return_value = close_response

    with patch("rally.alpaca_executor._mcp_session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await execute_exit("AAPL")

    assert result.ticker == "AAPL"
    assert result.side == "sell"
    assert result.qty == 10
    assert result.success is True
    assert result.fill_price == 148.50
    assert result.order_id == "order-exit-1"


# ---------------------------------------------------------------------------
# check_pending_fills (mocked MCP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_pending_fills():
    orders_response = MagicMock()
    orders_response.content = [MagicMock(text=json.dumps([
        {"id": "order-1", "filled_avg_price": "150.25", "status": "filled"},
        {"id": "order-2", "filled_avg_price": "400.50", "status": "filled"},
        {"id": "order-other", "filled_avg_price": "99.99", "status": "filled"},
    ]))]

    mock_session = AsyncMock()
    mock_session.call_tool.return_value = orders_response

    with patch("rally.alpaca_executor._mcp_session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

        fills = await check_pending_fills(["order-1", "order-2"])

    assert fills == {"order-1": 150.25, "order-2": 400.50}
    # order-other not in our list, should be excluded
    assert "order-other" not in fills


@pytest.mark.asyncio
async def test_check_pending_fills_empty():
    fills = await check_pending_fills([])
    assert fills == {}
