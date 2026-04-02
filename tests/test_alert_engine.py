"""Tests for alert engine breach detection — trading/engine.py (MIC-96).

Tests AlertEngine.evaluate_single_ticker, check_prices, and execute_breach.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.engine import AlertEngine, AlertEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pos(
    ticker: str = "AAPL",
    entry: float = 100.0,
    stop: float = 95.0,
    target: float = 110.0,
    trailing: float = 97.0,
    **extra,
) -> dict:
    return {
        "ticker": ticker,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "trailing_stop": trailing,
        "highest_close": entry,
        "atr": 2.0,
        "bars_held": 1,
        "size": 0.10,
        "status": "open",
        **extra,
    }


@pytest.fixture
def engine():
    return AlertEngine()


# ---------------------------------------------------------------------------
# evaluate_single_ticker — stop loss breach
# ---------------------------------------------------------------------------


class TestStopLossBreach:

    @patch("trading.engine.log_price_alert", return_value=True)
    def test_stop_breached(self, mock_log, engine):
        pos = _make_pos(stop=95.0, trailing=0.0)
        event = engine.evaluate_single_ticker("AAPL", 94.0, pos)
        assert event is not None
        assert event.alert_type == "stop_breached"
        assert event.current_price == 94.0

    @patch("trading.engine.log_price_alert", return_value=True)
    def test_trailing_stop_breached(self, mock_log, engine):
        pos = _make_pos(stop=90.0, trailing=97.0)
        # trailing > stop, price <= trailing
        event = engine.evaluate_single_ticker("AAPL", 96.0, pos)
        assert event is not None
        assert event.alert_type == "stop_breached"
        assert event.level_name == "Trailing Stop"

    @patch("trading.engine.log_price_alert")
    def test_no_breach_above_stop(self, mock_log, engine):
        pos = _make_pos(stop=95.0, trailing=97.0)
        event = engine.evaluate_single_ticker("AAPL", 98.0, pos)
        # Price 98 > trailing 97 → no stop breach
        # Could be near_stop or near_target depending on proximity
        if event:
            assert event.alert_type != "stop_breached"


# ---------------------------------------------------------------------------
# evaluate_single_ticker — profit target breach
# ---------------------------------------------------------------------------


class TestProfitTargetBreach:

    @patch("trading.engine.log_price_alert", return_value=True)
    def test_target_breached(self, mock_log, engine):
        pos = _make_pos(stop=95.0, trailing=0.0, target=110.0)
        event = engine.evaluate_single_ticker("AAPL", 111.0, pos)
        assert event is not None
        assert event.alert_type == "target_breached"
        assert event.level_name == "Target"


# ---------------------------------------------------------------------------
# evaluate_single_ticker — proximity alerts
# ---------------------------------------------------------------------------


class TestProximityAlerts:

    @patch("trading.engine.log_price_alert", return_value=True)
    def test_near_stop_alert(self, mock_log, engine):
        engine._proximity_pct = 2.0
        pos = _make_pos(stop=100.0, trailing=0.0, target=120.0)
        # Price is 1% above stop → within proximity
        event = engine.evaluate_single_ticker("AAPL", 101.0, pos)
        assert event is not None
        assert event.alert_type == "near_stop"
        assert event.distance_pct > 0

    @patch("trading.engine.log_price_alert", return_value=True)
    def test_near_target_alert(self, mock_log, engine):
        engine._proximity_pct = 2.0
        pos = _make_pos(stop=80.0, trailing=0.0, target=110.0)
        # Price is 1% below target → within proximity
        event = engine.evaluate_single_ticker("AAPL", 109.0, pos)
        assert event is not None
        assert event.alert_type == "near_target"


# ---------------------------------------------------------------------------
# evaluate_single_ticker — PnL calculation
# ---------------------------------------------------------------------------


class TestPnlCalculation:

    @patch("trading.engine.log_price_alert", return_value=True)
    def test_pnl_calculated_correctly(self, mock_log, engine):
        pos = _make_pos(entry=100.0, stop=95.0, trailing=0.0)
        event = engine.evaluate_single_ticker("AAPL", 94.0, pos)
        assert event is not None
        assert event.pnl_pct == -6.0  # (94/100 - 1) * 100

    @patch("trading.engine.log_price_alert", return_value=True)
    def test_pnl_positive_on_target(self, mock_log, engine):
        pos = _make_pos(entry=100.0, stop=95.0, trailing=0.0, target=110.0)
        event = engine.evaluate_single_ticker("AAPL", 111.0, pos)
        assert event is not None
        assert event.pnl_pct == 11.0


# ---------------------------------------------------------------------------
# Breach dedup via log_price_alert
# ---------------------------------------------------------------------------


class TestBreachDedup:

    @patch("trading.engine.log_price_alert", return_value=False)
    def test_dedup_blocks_duplicate(self, mock_log, engine):
        """When log_price_alert returns False, no event is emitted (dedup)."""
        pos = _make_pos(stop=95.0, trailing=0.0)
        event = engine.evaluate_single_ticker("AAPL", 94.0, pos)
        assert event is None  # deduplicated


# ---------------------------------------------------------------------------
# check_prices — batch evaluation
# ---------------------------------------------------------------------------


class TestCheckPrices:

    @pytest.mark.asyncio
    @patch("trading.engine.save_position_meta")
    @patch("trading.engine.log_price_alert", return_value=True)
    async def test_batch_evaluation(self, mock_log, mock_save, engine):
        positions = [
            _make_pos("AAPL", entry=100.0, stop=95.0, trailing=0.0, target=110.0),
            _make_pos("MSFT", entry=200.0, stop=190.0, trailing=0.0, target=220.0),
        ]
        quotes = {
            "AAPL": {"price": 94.0},   # breaches stop
            "MSFT": {"price": 205.0},   # no breach
        }

        events = await engine.check_prices(positions, quotes)
        breached_tickers = [e.ticker for e in events]
        assert "AAPL" in breached_tickers

    @pytest.mark.asyncio
    async def test_skips_quotes_with_errors(self, engine):
        positions = [_make_pos("AAPL")]
        quotes = {"AAPL": {"error": "network down"}}

        events = await engine.check_prices(positions, quotes)
        assert events == []

    @pytest.mark.asyncio
    async def test_skips_missing_quotes(self, engine):
        positions = [_make_pos("AAPL")]
        quotes = {}  # no quote for AAPL

        events = await engine.check_prices(positions, quotes)
        assert events == []


# ---------------------------------------------------------------------------
# execute_breach
# ---------------------------------------------------------------------------


class TestExecuteBreach:

    def _make_broker(self, **overrides):
        broker = MagicMock()
        broker.cancel_exit_orders = AsyncMock()
        broker.execute_exit = AsyncMock()
        broker.get_recent_sell_fills = AsyncMock(return_value={})
        broker.close_position = AsyncMock(return_value=None)
        broker.log_order = MagicMock()
        for k, v in overrides.items():
            setattr(broker, k, v)
        return broker

    @pytest.mark.asyncio
    async def test_cancels_oco_then_exits(self):
        broker = self._make_broker()
        broker.execute_exit.return_value = MagicMock(
            already_closed=False, fill_price=94.0, qty=10, order_id="exit_789",
            success=True, error=None,
        )
        broker.close_position.return_value = {
            "realized_pnl_pct": -6.0, "bars_held": 3,
        }
        engine = AlertEngine(broker=broker)

        pos = _make_pos(target_order_id="t_123", trail_order_id="s_456")
        result = await engine.execute_breach("AAPL", pos, 94.0, "stop_loss")
        assert result is not None
        assert result.ticker == "AAPL"
        assert result.fill_price == 94.0
        broker.cancel_exit_orders.assert_called_once_with("t_123", "s_456")

    @pytest.mark.asyncio
    async def test_oco_already_filled(self):
        """When OCO already filled, reason becomes 'oco_fill' with broker fill price."""
        broker = self._make_broker()
        broker.execute_exit.return_value = MagicMock(
            already_closed=True, fill_price=None, qty=10, order_id="exit_789",
            success=True, error=None,
        )
        broker.get_recent_sell_fills.return_value = {"AAPL": 95.5}
        broker.close_position.return_value = {
            "realized_pnl_pct": -4.5, "bars_held": 2,
        }
        engine = AlertEngine(broker=broker)

        pos = _make_pos()
        result = await engine.execute_breach("AAPL", pos, 94.0, "stop_loss")
        assert result is not None
        assert result.exit_reason == "oco_fill"
        assert result.fill_price == 95.5

    @pytest.mark.asyncio
    async def test_exit_failure_returns_none(self):
        broker = self._make_broker()
        broker.execute_exit.side_effect = Exception("Alpaca API down")
        engine = AlertEngine(broker=broker)

        pos = _make_pos()
        result = await engine.execute_breach("AAPL", pos, 94.0, "stop_loss")
        assert result is None


# ---------------------------------------------------------------------------
# AlertEvent model
# ---------------------------------------------------------------------------


class TestAlertEvent:

    def test_alert_event_types(self):
        for atype in ["stop_breached", "target_breached", "near_stop", "near_target"]:
            event = AlertEvent(
                ticker="AAPL", alert_type=atype, current_price=100.0,
                level_price=95.0, level_name="Stop", entry_price=90.0, pnl_pct=5.0,
            )
            assert event.alert_type == atype
