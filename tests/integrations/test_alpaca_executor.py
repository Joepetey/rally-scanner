"""Tests for Alpaca executor (alpaca-py SDK) and order embeds."""

from unittest.mock import MagicMock, patch

import pytest
from alpaca.trading.enums import OrderSide, OrderStatus

from db.trading.positions import load_positions, save_positions
from integrations.alpaca.broker import is_enabled
from integrations.alpaca.entries import execute_entries
from integrations.alpaca.exits import cancel_order, execute_exit, execute_exits
from integrations.alpaca.fills import check_pending_fills
from integrations.alpaca.models import OrderResult
from integrations.discord.notify import _order_embed, _order_failure_embed
from tests.helpers.alpaca_mock import MockAlpacaOrder
from trading.positions import (
    close_position_intraday,
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
    # trailing_stop and highest_close recalibrated to fill price
    from rally_ml.config import PARAMS
    assert aapl["trailing_stop"] == round(149.50 - PARAMS.trailing_stop_atr_mult * 3.0, 4)
    assert aapl["highest_close"] == 149.50
    # stop_price (range_low) left untouched — it's a strategy-derived level, not entry-relative
    assert aapl["stop_price"] == 100.0


# ---------------------------------------------------------------------------
# execute_entries (shared alpaca_mock)
# ---------------------------------------------------------------------------

# Common no-op patches needed for all execute_entries tests (new DB helpers).
_ENTRY_PATCHES = [
    patch("integrations.alpaca.entries.get_total_exposure", return_value=0.0),
    patch("integrations.alpaca.entries.get_group_exposure", return_value=(0, 0.0)),
    patch("integrations.alpaca.entries.load_positions",
          return_value={"positions": [], "closed_today": [], "last_updated": ""}),
    patch("integrations.alpaca.entries.is_circuit_breaker_active", return_value=False),
    patch("integrations.alpaca.entries.load_all_position_meta", return_value=[]),
    patch("integrations.alpaca.entries.get_recently_closed_tickers", return_value=set()),
    patch("integrations.alpaca.sizing.enqueue_signal"),
    patch("integrations.alpaca.sizing.log_skipped_signal"),
    patch("integrations.alpaca.entries.remove_from_queue"),
    # Avoid real Alpaca data API calls in tests — return fallback-based limit price
    patch("integrations.alpaca.entries._compute_limit_price",
          side_effect=lambda price, ticker: round(price * 1.002, 2)),
]


def _apply_entry_patches():
    from contextlib import ExitStack
    stack = ExitStack()
    for p in _ENTRY_PATCHES:
        stack.enter_context(p)
    return stack


@pytest.mark.asyncio
async def test_execute_entries_crypto_uses_slash_symbol(alpaca_mock):
    """Crypto entries use BTC/USD Alpaca symbol format and fractional qty."""
    alpaca_mock.set_fill_behavior("immediate", fill_price=50000.0)
    signals = [
        {"ticker": "BTC", "entry_price": 50000.0, "size": 0.10},
    ]

    with _apply_entry_patches():
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    result = results[0]
    assert result.success
    # qty should be fractional: 100_000 * 0.10 / 50_000 = 0.2
    assert result.qty == pytest.approx(0.2, rel=1e-6)
    # Alpaca symbol must use slash format
    submitted = alpaca_mock.submit_order.call_args[0][0]
    assert submitted.symbol == "BTC/USD"
    # Must be a limit order, not market (MIC-107)
    from alpaca.trading.requests import LimitOrderRequest
    assert isinstance(submitted, LimitOrderRequest)


@pytest.mark.asyncio
async def test_execute_entries_qty(alpaca_mock):
    """Verify qty calculation (trailing stop is now deferred)."""
    alpaca_mock.set_fill_behavior("pending")
    signals = [
        {"ticker": "AAPL", "close": 150.0, "size": 0.10, "atr_pct": 0.02},
    ]

    with _apply_entry_patches():
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    r = results[0]
    assert r.ticker == "AAPL"
    assert r.side == "buy"
    # 100000 * 0.10 / 150 = 66
    assert r.qty == 66
    assert r.success is True
    assert r.order_id is not None
    # Trailing stop is deferred — no trail_order_id yet
    assert r.trail_order_id is None

    # Only one limit buy call (MIC-107: no more market orders)
    alpaca_mock.submit_order.assert_called_once()
    from alpaca.trading.requests import LimitOrderRequest
    submitted = alpaca_mock.submit_order.call_args[0][0]
    assert isinstance(submitted, LimitOrderRequest)


@pytest.mark.asyncio
async def test_execute_entries_exposure_cap(alpaca_mock):
    """Signal is queued when no capital available even after partial sizing."""
    # Exposure at 99.5% leaves only 0.5% — below min_position_size (1%)
    # so partial sizing can't help and the signal is queued.
    signals = [
        {"ticker": "AAPL", "close": 150.0, "size": 0.10, "atr_pct": 0.02, "p_rally": 0.60},
    ]

    with patch("integrations.alpaca.entries.get_total_exposure", return_value=0.995), \
         patch("integrations.alpaca.entries.get_group_exposure", return_value=(0, 0.0)), \
         patch("integrations.alpaca.entries.load_positions",
               return_value={"positions": [], "closed_today": [], "last_updated": ""}), \
         patch("integrations.alpaca.entries.is_circuit_breaker_active", return_value=False), \
         patch("integrations.alpaca.entries.load_all_position_meta", return_value=[]), \
         patch("integrations.alpaca.entries.get_recently_closed_tickers", return_value=set()), \
         patch("integrations.alpaca.sizing.enqueue_signal") as mock_enqueue, \
         patch("integrations.alpaca.sizing.log_skipped_signal"), \
         patch("integrations.alpaca.entries.remove_from_queue"):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].skipped is True
    assert "exposure cap" in results[0].error.lower()
    mock_enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_execute_entries_partial_sizing(alpaca_mock):
    """Signal is scaled down when available capital is below requested size."""
    # 95% used, 5% available — signal wants 10% but gets scaled to 5%
    alpaca_mock.set_fill_behavior("pending")
    signals = [
        {"ticker": "AAPL", "close": 150.0, "size": 0.10, "atr_pct": 0.02, "p_rally": 0.60},
    ]

    with patch("integrations.alpaca.entries.get_total_exposure", return_value=0.95), \
         patch("integrations.alpaca.entries.get_group_exposure", return_value=(0, 0.0)), \
         patch("integrations.alpaca.entries.load_positions",
               return_value={"positions": [], "closed_today": [], "last_updated": ""}), \
         patch("integrations.alpaca.entries.is_circuit_breaker_active", return_value=False), \
         patch("integrations.alpaca.entries.load_all_position_meta", return_value=[]), \
         patch("integrations.alpaca.entries.get_recently_closed_tickers", return_value=set()), \
         patch("integrations.alpaca.sizing.enqueue_signal"), \
         patch("integrations.alpaca.sizing.log_skipped_signal"), \
         patch("integrations.alpaca.entries.remove_from_queue"):
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

    with patch("integrations.alpaca.entries.is_circuit_breaker_active", return_value=True), \
         patch("integrations.alpaca.entries.get_total_exposure", return_value=0.0), \
         patch("integrations.alpaca.entries.get_group_exposure", return_value=(0, 0.0)):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    assert results[0].success is False
    assert "circuit breaker" in results[0].error.lower()


@pytest.mark.asyncio
async def test_execute_entries_group_limit(alpaca_mock):
    """Entries should be skipped when group position limit is reached."""
    signals = [
        {"ticker": "AAPL", "close": 150.0, "size": 0.10, "atr_pct": 0.02},
    ]

    with patch("integrations.alpaca.entries.get_total_exposure", return_value=0.0), \
         patch("integrations.alpaca.entries.get_group_exposure", return_value=(3, 0.3)), \
         patch("integrations.alpaca.entries.load_positions",
               return_value={"positions": [], "closed_today": [], "last_updated": ""}), \
         patch("integrations.alpaca.entries.is_circuit_breaker_active", return_value=False), \
         patch("integrations.alpaca.entries.load_all_position_meta", return_value=[]), \
         patch("integrations.alpaca.entries.get_recently_closed_tickers", return_value=set()), \
         patch("integrations.alpaca.sizing.enqueue_signal"), \
         patch("integrations.alpaca.sizing.log_skipped_signal"), \
         patch("integrations.alpaca.entries.log_skipped_signal"), \
         patch("integrations.alpaca.entries.remove_from_queue"):
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].skipped is True
    assert "max positions" in results[0].error.lower()


@pytest.mark.asyncio
async def test_execute_entries_not_tradable(alpaca_mock):
    """42210000 (asset not tradable) should be a skipped result, not a hard failure."""
    alpaca_mock.set_submit_error('{"code":42210000,"message":"asset \\"RGC\\" is not tradable"}')
    signals = [
        {"ticker": "RGC", "close": 10.0, "size": 0.05, "atr_pct": 0.02},
    ]

    with _apply_entry_patches():
        results = await execute_entries(signals, equity=100_000.0)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].skipped is True
    assert results[0].ticker == "RGC"


@pytest.mark.asyncio
async def test_execute_entries_cooldown_skips_ticker(alpaca_mock):
    """Ticker in cooldown period should be skipped by prepare_entry_plan."""
    signals = [
        {"ticker": "CGON", "close": 68.0, "size": 0.10, "atr_pct": 0.02},
    ]

    with _apply_entry_patches(), \
         patch("integrations.alpaca.entries.get_recently_closed_tickers",
               return_value={"CGON"}):
        results = await execute_entries(signals, equity=100_000.0)

    # CGON should be skipped with an informative result (no order attempted)
    assert len(results) == 1
    assert results[0].skipped is True
    assert results[0].success is False
    assert "cooldown" in results[0].error
    alpaca_mock.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# execute_exit (shared alpaca_mock)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_exit(alpaca_mock):
    """Verify close_position call and fill price extraction."""
    alpaca_mock.set_fill_behavior("immediate", fill_price=148.50)

    result = await execute_exit("AAPL")

    assert result.ticker == "AAPL"
    assert result.side == "sell"
    assert result.qty == 10
    assert result.success is True
    assert result.fill_price == 148.50
    assert result.order_id is not None


# ---------------------------------------------------------------------------
# check_pending_fills (shared alpaca_mock)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_pending_fills(alpaca_mock):
    alpaca_mock.add_filled_order("order-1", fill_price=150.25)
    alpaca_mock.add_filled_order("order-2", fill_price=400.50)
    alpaca_mock.add_filled_order("order-other", fill_price=99.99)

    fills = await check_pending_fills(["order-1", "order-2"])

    assert fills == {"order-1": 150.25, "order-2": 400.50}
    assert "order-other" not in fills


@pytest.mark.asyncio
async def test_check_pending_fills_empty():
    fills = await check_pending_fills([])
    assert fills == {}


# ---------------------------------------------------------------------------
# cancel_order (shared alpaca_mock)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_order_success(alpaca_mock):
    result = await cancel_order("trail-123")
    assert result is True


@pytest.mark.asyncio
async def test_cancel_order_failure(alpaca_mock):
    alpaca_mock.set_cancel_error("connection failed")
    result = await cancel_order("trail-123")
    assert result is False


# ---------------------------------------------------------------------------
# position not found (40410000) — trailing stop already closed it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_exit_position_not_found(alpaca_mock):
    """40410000 means trailing stop already closed the position — treat as success."""
    alpaca_mock.set_close_behavior("already_closed")

    result = await execute_exit("WULF")

    assert result.success is True
    assert result.already_closed is True
    assert result.ticker == "WULF"
    assert result.qty == 0


@pytest.mark.asyncio
async def test_execute_exits_position_not_found_does_not_fail_batch(alpaca_mock):
    """A 40410000 on one ticker should not prevent other tickers from being processed."""
    # First call raises already-closed, second returns a filled order
    filled_order = MockAlpacaOrder(
        id="order-exit-2",
        symbol="MSFT",
        qty="5",
        side=OrderSide.SELL,
        status=OrderStatus.FILLED,
        filled_avg_price="50.00",
        filled_qty="5",
    )
    alpaca_mock.close_position.side_effect = [
        Exception('{"code":40410000,"message":"position not found: ABVX"}'),
        filled_order,
    ]

    closed = [
        {"ticker": "ABVX"},
        {"ticker": "MSFT"},
    ]

    results = await execute_exits(closed)

    assert len(results) == 2
    abvx = next(r for r in results if r.ticker == "ABVX")
    msft = next(r for r in results if r.ticker == "MSFT")
    assert abvx.success is True
    assert abvx.already_closed is True
    assert msft.success is True
    assert msft.fill_price == 50.0


# ---------------------------------------------------------------------------
# _compute_limit_price (MIC-107)
# ---------------------------------------------------------------------------


def test_compute_limit_price_with_snapshot():
    """Uses bid/ask midpoint + buffer when snapshot is available."""
    from dataclasses import dataclass

    from integrations.alpaca.entries import _compute_limit_price

    @dataclass
    class FakeQuote:
        bid_price: float
        ask_price: float

    @dataclass
    class FakeSnap:
        latest_quote: FakeQuote
        latest_trade: None = None

    fake_client = MagicMock()
    fake_client.get_stock_snapshot.return_value = {
        "AAPL": FakeSnap(latest_quote=FakeQuote(bid_price=149.90, ask_price=150.10)),
    }

    with patch("integrations.alpaca.sizing._data_client", return_value=fake_client):
        price = _compute_limit_price(150.0, "AAPL")

    # midpoint = 150.0, buffer = 0.2% → 150.0 * 1.002 = 150.30
    assert price == 150.30


def test_compute_limit_price_fallback():
    """Falls back to signal price + buffer when snapshot fails."""
    from integrations.alpaca.entries import _compute_limit_price

    with patch("integrations.alpaca.sizing._data_client", side_effect=Exception("no keys")):
        price = _compute_limit_price(100.0, "AAPL")

    assert price == 100.20  # 100 * 1.002
