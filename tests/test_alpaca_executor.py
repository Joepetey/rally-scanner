"""Tests for Alpaca executor (alpaca-py SDK) and order embeds."""

from unittest.mock import MagicMock, patch

import pytest

from integrations.alpaca.executor import (
    OrderResult,
    cancel_order,
    check_pending_fills,
    check_trail_stop_fills,
    execute_entries,
    execute_exit,
    execute_exits,
    get_account_equity,
    get_all_positions,
    get_snapshots,
    is_enabled,
    place_trailing_stop,
)
from integrations.discord.notify import _order_embed, _order_failure_embed
from db.positions import load_positions, save_positions
from trading.positions import (
    close_position_intraday,
    get_trail_order_ids,
    update_fill_prices,
)

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
    assert r.trail_order_id is None


def test_order_result_with_trail():
    r = OrderResult(
        ticker="AAPL", side="buy", qty=10, success=True,
        order_id="abc123", trail_order_id="trail-456",
    )
    assert r.trail_order_id == "trail-456"


# ---------------------------------------------------------------------------
# _order_embed / _order_failure_embed
# ---------------------------------------------------------------------------


def test_order_embed():
    # Orders arrive as model_dump() dicts (matching scheduler's orders.extend([r.model_dump() ...]))
    results = [
        OrderResult(ticker="AAPL", side="buy", qty=10, success=True,
                    order_id="abc12345", fill_price=150.25).model_dump(),
        OrderResult(ticker="MSFT", side="sell", qty=5, success=True,
                    order_id="def67890").model_dump(),
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
                    error="Insufficient buying power").model_dump(),
    ]
    embed = _order_failure_embed(results)
    assert embed["title"] == "Alpaca Order Failures (1)"
    assert embed["color"] == 0xFF4500
    assert "Insufficient buying power" in embed["fields"][0]["value"]
    assert "unaffected" in embed["footer"]["text"].lower()


# ---------------------------------------------------------------------------
# close_position_intraday (DB-backed)
# ---------------------------------------------------------------------------


def test_close_position_intraday(tmp_models_dir):
    save_positions({
        "positions": [
            {"ticker": "AAPL", "entry_price": 150.0, "entry_date": "2024-01-10",
             "stop_price": 145.0, "target_price": 165.0, "status": "open"},
            {"ticker": "MSFT", "entry_price": 400.0, "entry_date": "2024-01-10",
             "stop_price": 390.0, "target_price": 420.0, "status": "open"},
        ],
        "closed_today": [],
    })

    closed = close_position_intraday("AAPL", 144.50, "stop")
    assert closed is not None
    assert closed["ticker"] == "AAPL"
    assert closed["exit_price"] == 144.50
    assert closed["exit_reason"] == "stop"
    assert closed["realized_pnl_pct"] == pytest.approx(-3.67, abs=0.01)
    assert closed["status"] == "closed"

    # Verify DB updated
    reloaded = load_positions()
    assert len(reloaded["positions"]) == 1
    assert reloaded["positions"][0]["ticker"] == "MSFT"


def test_close_position_intraday_not_found(tmp_models_dir):
    result = close_position_intraday("AAPL", 144.50, "stop")
    assert result is None


# ---------------------------------------------------------------------------
# update_fill_prices (DB-backed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_fill_prices(tmp_models_dir):
    save_positions({
        "positions": [
            {"ticker": "AAPL", "entry_price": 150.0, "entry_date": "2024-01-10",
             "current_price": 152.0, "order_id": "order-123",
             "stop_price": 100.0, "trailing_stop": 145.0, "highest_close": 150.0,
             "atr": 3.0, "unrealized_pnl_pct": 1.33, "status": "open"},
            {"ticker": "MSFT", "entry_price": 400.0, "entry_date": "2024-01-10",
             "current_price": 405.0, "status": "open"},
        ],
        "closed_today": [],
    })

    count = await update_fill_prices({"order-123": 149.50})
    assert count == 1

    reloaded = load_positions()
    aapl = [p for p in reloaded["positions"] if p["ticker"] == "AAPL"][0]
    assert aapl["entry_price"] == 149.50
    assert aapl["order_id"] is None
    # Stops recalibrated to fill price (signal stop was far off)
    from config import PARAMS
    assert aapl["stop_price"] == round(149.50 * (1 - PARAMS.fallback_stop_pct), 4)
    assert aapl["trailing_stop"] == round(149.50 - PARAMS.trailing_stop_atr_mult * 3.0, 4)
    assert aapl["highest_close"] == 149.50


@pytest.mark.asyncio
async def test_update_fill_prices_stop_already_correct(tmp_models_dir):
    """Stop within 0.1% of fill — must not be clobbered."""
    from config import PARAMS
    fill_price = 149.50
    correct_stop = round(fill_price * (1 - PARAMS.fallback_stop_pct), 4)
    save_positions({
        "positions": [
            {"ticker": "AAPL", "entry_price": 150.0, "entry_date": "2024-01-10",
             "current_price": 152.0, "order_id": "order-456",
             "stop_price": correct_stop, "trailing_stop": 147.0, "highest_close": 150.0,
             "atr": 3.0, "unrealized_pnl_pct": 1.33, "status": "open"},
        ],
        "closed_today": [],
    })

    await update_fill_prices({"order-456": fill_price})

    reloaded = load_positions()
    aapl = [p for p in reloaded["positions"] if p["ticker"] == "AAPL"][0]
    # stop_price should be unchanged (was already correct)
    assert aapl["stop_price"] == correct_stop


@pytest.mark.asyncio
async def test_update_fill_prices_no_matches(tmp_models_dir):
    save_positions({
        "positions": [
            {"ticker": "AAPL", "entry_price": 150.0, "entry_date": "2024-01-10",
             "current_price": 152.0, "status": "open"},
        ],
        "closed_today": [],
    })

    count = await update_fill_prices({"order-999": 149.50})
    assert count == 0


# ---------------------------------------------------------------------------
# Helper: mock Alpaca order object
# ---------------------------------------------------------------------------


def _mock_order(order_id="order-abc", qty="10", filled_avg_price=None, filled_qty=None,
                status="filled"):
    from alpaca.trading.enums import OrderStatus
    order = MagicMock()
    order.id = order_id
    order.qty = qty
    order.filled_qty = filled_qty
    order.filled_avg_price = filled_avg_price
    order.status = OrderStatus(status)
    return order


def _mock_position(symbol="AAPL", qty="10", avg_entry_price="150.25",
                   market_value="1520.00", unrealized_pl="17.50"):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.avg_entry_price = avg_entry_price
    pos.market_value = market_value
    pos.unrealized_pl = unrealized_pl
    return pos


# ---------------------------------------------------------------------------
# execute_entries (mocked alpaca-py)
# ---------------------------------------------------------------------------

# Common no-op patches needed for all execute_entries tests (new DB helpers).
_ENTRY_NOOP_PATCHES = [
    patch("integrations.alpaca.executor.load_all_position_meta", return_value=[]),
    patch("integrations.alpaca.executor.enqueue_signal"),
    patch("integrations.alpaca.executor.log_skipped_signal"),
    patch("integrations.alpaca.executor.remove_from_queue"),
]


def _apply_noop_patches(patches):
    """Context manager that enters a list of patch objects."""
    from contextlib import ExitStack
    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


@pytest.mark.asyncio
async def test_execute_entries_crypto_uses_slash_symbol():
    """Crypto entries use BTC/USD Alpaca symbol format and fractional qty."""
    signals = [
        {"ticker": "BTC", "entry_price": 50000.0, "size": 0.10},
    ]

    mock_order = MagicMock()
    mock_order.id = "crypto-order-id"
    mock_order.filled_avg_price = 50000.0
    mock_client = MagicMock()
    mock_client.submit_order.return_value = mock_order
    _empty = {"positions": [], "closed_today": [], "last_updated": ""}
    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client), \
         patch("integrations.alpaca.executor.get_total_exposure", return_value=0.0), \
         patch("integrations.alpaca.executor.get_group_exposure", return_value=(0, 0.0)), \
         patch("integrations.alpaca.executor.load_positions", return_value=_empty), \
         patch("integrations.alpaca.executor.is_circuit_breaker_active", return_value=False), \
         patch("integrations.alpaca.executor.load_all_position_meta", return_value=[]), \
         patch("integrations.alpaca.executor.enqueue_signal"), \
         patch("integrations.alpaca.executor.log_skipped_signal"), \
         patch("integrations.alpaca.executor.remove_from_queue"):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    result = results[0]
    assert result.success
    # qty should be fractional: 100_000 * 0.10 / 50_000 = 0.2
    import pytest as _pytest
    assert result.qty == _pytest.approx(0.2, rel=1e-6)
    # Alpaca symbol must use slash format
    submitted = mock_client.submit_order.call_args[0][0]
    assert submitted.symbol == "BTC/USD"


@pytest.mark.asyncio
async def test_execute_entries_qty():
    """Verify qty calculation (trailing stop is now deferred)."""
    signals = [
        {"ticker": "AAPL", "close": 150.0, "size": 0.10, "atr_pct": 0.02},
    ]

    mock_order = _mock_order(order_id="order-abc", filled_avg_price=None)
    mock_client = MagicMock()
    mock_client.submit_order.return_value = mock_order

    _empty = {"positions": [], "closed_today": [], "last_updated": ""}
    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client), \
         patch("integrations.alpaca.executor.get_total_exposure", return_value=0.0), \
         patch("integrations.alpaca.executor.get_group_exposure", return_value=(0, 0.0)), \
         patch("integrations.alpaca.executor.load_positions", return_value=_empty), \
         patch("integrations.alpaca.executor.is_circuit_breaker_active", return_value=False), \
         patch("integrations.alpaca.executor.load_all_position_meta", return_value=[]), \
         patch("integrations.alpaca.executor.enqueue_signal"), \
         patch("integrations.alpaca.executor.log_skipped_signal"), \
         patch("integrations.alpaca.executor.remove_from_queue"):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    r = results[0]
    assert r.ticker == "AAPL"
    assert r.side == "buy"
    # 100000 * 0.10 / 150 = 66
    assert r.qty == 66
    assert r.success is True
    assert r.order_id == "order-abc"
    # Trailing stop is deferred — no trail_order_id yet
    assert r.trail_order_id is None

    # Only market buy call
    assert mock_client.submit_order.call_count == 1


@pytest.mark.asyncio
async def test_execute_entries_exposure_cap():
    """Signal is queued when no capital available even after partial sizing."""
    # Exposure at 99.5% leaves only 0.5% — below min_position_size (1%)
    # so partial sizing can't help and the signal is queued.
    signals = [
        {"ticker": "AAPL", "close": 150.0, "size": 0.10, "atr_pct": 0.02, "p_rally": 0.60},
    ]

    mock_client = MagicMock()
    _empty = {"positions": [], "closed_today": [], "last_updated": ""}
    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client), \
         patch("integrations.alpaca.executor.get_total_exposure", return_value=0.995), \
         patch("integrations.alpaca.executor.get_group_exposure", return_value=(0, 0.0)), \
         patch("integrations.alpaca.executor.load_positions", return_value=_empty), \
         patch("integrations.alpaca.executor.is_circuit_breaker_active", return_value=False), \
         patch("integrations.alpaca.executor.load_all_position_meta", return_value=[]), \
         patch("integrations.alpaca.executor.enqueue_signal") as mock_enqueue, \
         patch("integrations.alpaca.executor.log_skipped_signal"), \
         patch("integrations.alpaca.executor.remove_from_queue"):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].skipped is True
    assert "exposure cap" in results[0].error.lower()
    mock_enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_execute_entries_partial_sizing():
    """Signal is scaled down when available capital is below requested size."""
    # 95% used, 5% available — signal wants 10% but gets scaled to 5%
    signals = [
        {"ticker": "AAPL", "close": 150.0, "size": 0.10, "atr_pct": 0.02, "p_rally": 0.60},
    ]
    mock_order = _mock_order(order_id="partial-1", filled_avg_price=None)
    mock_client = MagicMock()
    mock_client.submit_order.return_value = mock_order

    _empty = {"positions": [], "closed_today": [], "last_updated": ""}
    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client), \
         patch("integrations.alpaca.executor.get_total_exposure", return_value=0.95), \
         patch("integrations.alpaca.executor.get_group_exposure", return_value=(0, 0.0)), \
         patch("integrations.alpaca.executor.load_positions", return_value=_empty), \
         patch("integrations.alpaca.executor.is_circuit_breaker_active", return_value=False), \
         patch("integrations.alpaca.executor.load_all_position_meta", return_value=[]), \
         patch("integrations.alpaca.executor.enqueue_signal"), \
         patch("integrations.alpaca.executor.log_skipped_signal"), \
         patch("integrations.alpaca.executor.remove_from_queue"):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    assert results[0].success is True
    # qty = 100000 * 0.05 / 150 = 33 (scaled to 5% available)
    assert results[0].qty == 33
    # actual_size must reflect the reduced allocation, not the original 10%
    assert results[0].actual_size == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_execute_entries_circuit_breaker():
    """Entries should be blocked when circuit breaker is active."""
    signals = [
        {"ticker": "AAPL", "close": 150.0, "size": 0.10, "atr_pct": 0.02},
    ]

    with patch("integrations.alpaca.executor.is_circuit_breaker_active", return_value=True), \
         patch("integrations.alpaca.executor.get_total_exposure", return_value=0.0), \
         patch("integrations.alpaca.executor.get_group_exposure", return_value=(0, 0.0)):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    assert results[0].success is False
    assert "circuit breaker" in results[0].error.lower()


@pytest.mark.asyncio
async def test_execute_entries_group_limit():
    """Entries should be skipped when group position limit is reached."""
    signals = [
        {"ticker": "AAPL", "close": 150.0, "size": 0.10, "atr_pct": 0.02},
    ]

    mock_client = MagicMock()
    _empty = {"positions": [], "closed_today": [], "last_updated": ""}
    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client), \
         patch("integrations.alpaca.executor.get_total_exposure", return_value=0.0), \
         patch("integrations.alpaca.executor.get_group_exposure", return_value=(3, 0.3)), \
         patch("integrations.alpaca.executor.load_positions", return_value=_empty), \
         patch("integrations.alpaca.executor.is_circuit_breaker_active", return_value=False), \
         patch("integrations.alpaca.executor.load_all_position_meta", return_value=[]), \
         patch("integrations.alpaca.executor.enqueue_signal"), \
         patch("integrations.alpaca.executor.log_skipped_signal"), \
         patch("integrations.alpaca.executor.remove_from_queue"):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].skipped is True
    assert "max positions" in results[0].error.lower()


@pytest.mark.asyncio
async def test_execute_entries_not_tradable():
    """42210000 (asset not tradable) should be a skipped result, not a hard failure."""
    signals = [
        {"ticker": "RGC", "close": 10.0, "size": 0.05, "atr_pct": 0.02},
    ]
    mock_client = MagicMock()
    mock_client.submit_order.side_effect = Exception(
        '{"code":42210000,"message":"asset \\"RGC\\" is not tradable"}'
    )
    _empty = {"positions": [], "closed_today": [], "last_updated": ""}
    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client), \
         patch("integrations.alpaca.executor.get_total_exposure", return_value=0.0), \
         patch("integrations.alpaca.executor.get_group_exposure", return_value=(0, 0.0)), \
         patch("integrations.alpaca.executor.load_positions", return_value=_empty), \
         patch("integrations.alpaca.executor.is_circuit_breaker_active", return_value=False), \
         patch("integrations.alpaca.executor.load_all_position_meta", return_value=[]), \
         patch("integrations.alpaca.executor.enqueue_signal"), \
         patch("integrations.alpaca.executor.log_skipped_signal"), \
         patch("integrations.alpaca.executor.remove_from_queue"):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].skipped is True
    assert results[0].ticker == "RGC"


# ---------------------------------------------------------------------------
# execute_exit (mocked alpaca-py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_exit():
    """Verify close_position call and fill price extraction."""
    mock_order = _mock_order(
        order_id="order-exit-1", qty="10", filled_avg_price="148.50",
    )
    mock_client = MagicMock()
    mock_client.close_position.return_value = mock_order

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        result = await execute_exit("AAPL")

    assert result.ticker == "AAPL"
    assert result.side == "sell"
    assert result.qty == 10
    assert result.success is True
    assert result.fill_price == 148.50
    assert result.order_id == "order-exit-1"


# ---------------------------------------------------------------------------
# check_pending_fills (mocked alpaca-py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_pending_fills():
    mock_orders = [
        _mock_order(order_id="order-1", filled_avg_price="150.25"),
        _mock_order(order_id="order-2", filled_avg_price="400.50"),
        _mock_order(order_id="order-other", filled_avg_price="99.99"),
    ]

    mock_client = MagicMock()
    mock_client.get_orders.return_value = mock_orders

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        fills = await check_pending_fills(["order-1", "order-2"])

    assert fills == {"order-1": 150.25, "order-2": 400.50}
    # order-other not in our list, should be excluded
    assert "order-other" not in fills


@pytest.mark.asyncio
async def test_check_pending_fills_empty():
    fills = await check_pending_fills([])
    assert fills == {}


# ---------------------------------------------------------------------------
# cancel_order (mocked alpaca-py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_order_success():
    mock_client = MagicMock()
    mock_client.cancel_order_by_id.return_value = None  # cancel returns None

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        result = await cancel_order("trail-123")

    assert result is True


@pytest.mark.asyncio
async def test_cancel_order_failure():
    mock_client = MagicMock()
    mock_client.cancel_order_by_id.side_effect = Exception("connection failed")

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        result = await cancel_order("trail-123")

    assert result is False


# ---------------------------------------------------------------------------
# execute_exit with trailing stop cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_exit_cancels_trailing_stop():
    """When trail_order_id is given, it should be cancelled before closing."""
    mock_order = _mock_order(
        order_id="order-exit-1", qty="10", filled_avg_price="148.50",
    )
    mock_client = MagicMock()
    mock_client.close_position.return_value = mock_order

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        result = await execute_exit("AAPL", trail_order_id="trail-999")

    assert result.success is True
    assert result.fill_price == 148.50
    # cancel_order_by_id + close_position both called
    mock_client.cancel_order_by_id.assert_called_once_with("trail-999")
    mock_client.close_position.assert_called_once()


# ---------------------------------------------------------------------------
# execute_exits passes trail_order_id from position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_exits_passes_trail_order_id():
    """Closed positions with trail_order_id should pass it to exit."""
    closed = [
        {"ticker": "AAPL", "trail_order_id": "trail-111"},
    ]

    mock_order = _mock_order(order_id="order-exit-1", qty="10")
    mock_client = MagicMock()
    mock_client.close_position.return_value = mock_order

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        results = await execute_exits(closed)

    assert len(results) == 1
    assert results[0].success is True
    # Trailing stop should have been cancelled
    mock_client.cancel_order_by_id.assert_called_once_with("trail-111")


# ---------------------------------------------------------------------------
# position not found (40410000) — trailing stop already closed it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_exit_position_not_found():
    """40410000 means trailing stop already closed the position — treat as success."""
    mock_client = MagicMock()
    mock_client.close_position.side_effect = Exception(
        '{"code":40410000,"message":"position not found: WULF"}'
    )

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        result = await execute_exit("WULF")

    assert result.success is True
    assert result.already_closed is True
    assert result.ticker == "WULF"
    assert result.qty == 0


@pytest.mark.asyncio
async def test_execute_exits_position_not_found_does_not_fail_batch():
    """A 40410000 on one ticker should not prevent other tickers from being processed."""
    mock_order = _mock_order(order_id="order-exit-2", qty="5", filled_avg_price="50.00")
    mock_client = MagicMock()
    mock_client.close_position.side_effect = [
        Exception('{"code":40410000,"message":"position not found: ABVX"}'),
        mock_order,
    ]

    closed = [
        {"ticker": "ABVX"},
        {"ticker": "MSFT"},
    ]

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        results = await execute_exits(closed)

    assert len(results) == 2
    abvx = next(r for r in results if r.ticker == "ABVX")
    msft = next(r for r in results if r.ticker == "MSFT")
    assert abvx.success is True
    assert abvx.already_closed is True
    assert msft.success is True
    assert msft.fill_price == 50.0


# ---------------------------------------------------------------------------
# check_trail_stop_fills
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_trail_stop_fills():
    mock_orders = [
        _mock_order(order_id="trail-1", filled_avg_price="145.00"),
        _mock_order(order_id="trail-other", filled_avg_price="99.00"),
    ]

    mock_client = MagicMock()
    mock_client.get_orders.return_value = mock_orders

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        fills = await check_trail_stop_fills({"AAPL": "trail-1"})

    assert fills == {"AAPL": 145.00}


@pytest.mark.asyncio
async def test_check_trail_stop_fills_empty():
    fills = await check_trail_stop_fills({})
    assert fills == {}


# ---------------------------------------------------------------------------
# get_trail_order_ids (DB-backed positions helper)
# ---------------------------------------------------------------------------


def test_get_trail_order_ids(tmp_models_dir):
    save_positions({
        "positions": [
            {"ticker": "AAPL", "entry_price": 150.0, "entry_date": "2024-01-10",
             "trail_order_id": "trail-1", "status": "open"},
            {"ticker": "MSFT", "entry_price": 400.0, "entry_date": "2024-01-10",
             "status": "open"},  # no trail order
            {"ticker": "NVDA", "entry_price": 500.0, "entry_date": "2024-01-10",
             "trail_order_id": "trail-3", "status": "open"},
        ],
        "closed_today": [],
    })

    result = get_trail_order_ids()
    assert result == {"AAPL": "trail-1", "NVDA": "trail-3"}


def test_get_trail_order_ids_empty(tmp_models_dir):
    result = get_trail_order_ids()
    assert result == {}


# ---------------------------------------------------------------------------
# get_account_equity (mocked alpaca-py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_account_equity():
    mock_account = MagicMock()
    mock_account.equity = "125000.50"

    mock_client = MagicMock()
    mock_client.get_account.return_value = mock_account

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        equity = await get_account_equity()

    assert equity == 125000.50
    mock_client.get_account.assert_called_once()


# ---------------------------------------------------------------------------
# get_all_positions (mocked alpaca-py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_all_positions():
    mock_positions = [
        _mock_position("AAPL", "10", "150.25", "1520.00", "17.50"),
        _mock_position("MSFT", "5", "400.00", "2050.00", "50.00"),
    ]

    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = mock_positions

    with patch("integrations.alpaca.broker._trading_client", return_value=mock_client):
        positions = await get_all_positions()

    assert len(positions) == 2
    assert positions[0]["ticker"] == "AAPL"
    assert positions[0]["qty"] == 10
    assert positions[0]["avg_entry_price"] == 150.25
    assert positions[0]["market_value"] == 1520.00
    assert positions[0]["unrealized_pl"] == 17.50
    assert positions[1]["ticker"] == "MSFT"


@pytest.mark.asyncio
async def test_get_all_positions_empty():
    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = []

    with patch("integrations.alpaca.broker._trading_client", return_value=mock_client):
        positions = await get_all_positions()

    assert positions == []


@pytest.mark.asyncio
async def test_get_all_positions_api_error():
    """If API raises, exception propagates."""
    mock_client = MagicMock()
    mock_client.get_all_positions.side_effect = Exception("unauthorized")

    with patch("integrations.alpaca.broker._trading_client", return_value=mock_client):
        with pytest.raises(Exception, match="unauthorized"):
            await get_all_positions()


# ---------------------------------------------------------------------------
# get_snapshots (mocked alpaca-py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_snapshots():
    mock_snap_aapl = MagicMock()
    mock_snap_aapl.latest_trade.price = 152.50
    mock_snap_aapl.latest_quote.bid_price = 152.40
    mock_snap_aapl.latest_quote.ask_price = 152.60
    mock_snap_aapl.latest_quote.bid_size = 100
    mock_snap_aapl.latest_quote.ask_size = 200

    mock_snap_msft = MagicMock()
    mock_snap_msft.latest_trade.price = 410.00
    mock_snap_msft.latest_quote.bid_price = 409.90
    mock_snap_msft.latest_quote.ask_price = 410.10
    mock_snap_msft.latest_quote.bid_size = 50
    mock_snap_msft.latest_quote.ask_size = 75

    mock_data = MagicMock()
    mock_data.get_stock_snapshot.return_value = {
        "AAPL": mock_snap_aapl,
        "MSFT": mock_snap_msft,
    }

    with patch("integrations.alpaca.executor._data_client", return_value=mock_data):
        result = await get_snapshots(["AAPL", "MSFT"])

    assert result["AAPL"]["price"] == 152.50
    assert result["AAPL"]["bid"] == 152.40
    assert result["AAPL"]["ask"] == 152.60
    assert result["MSFT"]["price"] == 410.00
    assert result["MSFT"]["bid"] == 409.90


@pytest.mark.asyncio
async def test_get_snapshots_returns_crypto_via_yfinance():
    """Crypto tickers are fetched via yfinance and returned alongside equity."""
    from unittest.mock import patch as _patch

    mock_yf_data = {
        "BTC-USD": {"price": 95000.0, "bid": 0, "ask": 0, "error": None},
    }
    # fetch_quotes is called with the yfinance-format ticker ("BTC-USD")
    # and keyed by uppercase symbol
    mock_yf_quotes = {
        "BTC-USD": {"price": 95000.0, "prev_close": 94000.0,
                    "change": 1000.0, "change_pct": 1.06,
                    "open": 94500.0, "day_high": 96000.0, "day_low": 94000.0,
                    "volume": 10000, "market_cap": None, "currency": "USD"},
    }
    with _patch("integrations.alpaca.executor.fetch_quotes", return_value=mock_yf_quotes):
        result = await get_snapshots(["BTC"])

    assert "BTC" in result
    assert result["BTC"]["price"] == 95000.0
    assert result["BTC"]["bid"] == 0


# ---------------------------------------------------------------------------
# place_trailing_stop (mocked alpaca-py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_trailing_stop():
    mock_order = _mock_order(order_id="trail-order-789")
    mock_client = MagicMock()
    mock_client.submit_order.return_value = mock_order

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        trail_id = await place_trailing_stop("AAPL", qty=10, trail_pct=3.0)

    assert trail_id == "trail-order-789"
    mock_client.submit_order.assert_called_once()
    # Verify the order request was a trailing stop sell
    call_args = mock_client.submit_order.call_args
    request = call_args[0][0]
    assert request.symbol == "AAPL"
    assert request.qty == 10
    assert request.trail_percent == 3.0


@pytest.mark.asyncio
async def test_place_trailing_stop_failure():
    """On failure, returns None instead of raising."""
    mock_client = MagicMock()
    mock_client.submit_order.side_effect = Exception("order rejected")

    with patch("integrations.alpaca.executor._trading_client", return_value=mock_client):
        trail_id = await place_trailing_stop("AAPL", qty=10, trail_pct=3.0)

    assert trail_id is None
