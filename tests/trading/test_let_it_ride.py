"""Tests for let-it-ride feature — target hit converts to trailing stop instead of selling.

Exercises the real logic paths:
  1. EOD: target_price=0 must NOT trigger profit_target exit
  2. Intraday: evaluate_breach with target=0 must NOT fire target_breached
  3. convert_to_trailing_stop: cancels OCO, places trailing stop, updates position
  4. _execute_breach_guarded: branches target_breached to let-it-ride
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rally_ml.config import PARAMS

from trading.engine import AlertEngine
from trading.events import LetItRideEvent
from trading.positions import update_existing_positions

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
        "qty": 10,
        "status": "open",
        "target_order_id": "t_123",
        "trail_order_id": "s_456",
        **extra,
    }


# ---------------------------------------------------------------------------
# EOD: target_price=0 does NOT trigger profit_target exit
# ---------------------------------------------------------------------------

class TestLetItRideEODGuard:

    def test_let_it_ride_does_not_exit(self, tmp_models_dir):
        """A let-it-ride position must NOT close as profit_target at EOD."""
        state = {
            "positions": [{
                "ticker": "FDX",
                "entry_price": 270.0,
                "stop_price": 274.05,  # profit lock floor
                "target_price": 280.0, # original target preserved
                "trailing_stop": 275.0,
                "highest_close": 280.0,
                "bars_held": 2,
                "size": 0.10,
                "status": "open",
                "entry_date": "2024-01-10",
                "let_it_ride": True,
            }],
            "closed_today": [],
        }
        results = [{
            "ticker": "FDX",
            "status": "ok",
            "close": 282.0,  # above entry, would have been above old target
            "date": "2024-01-15",
            "atr": 5.0,
        }]
        updated = update_existing_positions(state, results)
        assert len(updated["positions"]) == 1, "Position should remain open"
        assert len(updated["closed_today"]) == 0, "Position should NOT be closed"

    def test_normal_target_still_exits(self, tmp_models_dir):
        """Normal positions (target_price > 0) should still close at profit_target."""
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
        results = [{
            "ticker": "AAPL",
            "status": "ok",
            "close": 162.0,
            "date": "2024-01-15",
            "atr": 3.0,
        }]
        updated = update_existing_positions(state, results)
        assert len(updated["closed_today"]) == 1
        assert updated["closed_today"][0]["exit_reason"] == "profit_target"


# ---------------------------------------------------------------------------
# Intraday: evaluate_breach with target=0 does NOT fire
# ---------------------------------------------------------------------------

class TestLetItRideIntraday:

    def test_let_it_ride_no_target_breach(self, pg_db):
        """evaluate_single_ticker must not fire target_breached for let-it-ride positions."""
        engine = AlertEngine()
        pos = _make_pos(entry=270.0, target=280.0, stop=274.0, trailing=275.0, let_it_ride=True)
        event = engine.evaluate_single_ticker("FDX", 282.0, pos)
        assert event is None

    def test_stop_still_fires_for_let_it_ride(self, pg_db):
        """A let-it-ride position should still exit on stop breach."""
        engine = AlertEngine()
        pos = _make_pos(entry=270.0, target=280.0, stop=274.0, trailing=275.0, let_it_ride=True)
        event = engine.evaluate_single_ticker("FDX", 273.0, pos)
        assert event is not None
        assert event.alert_type == "stop_breached"
        assert event.level_name == "Trailing Stop"


# ---------------------------------------------------------------------------
# convert_to_trailing_stop — the core let-it-ride function
# ---------------------------------------------------------------------------

class TestConvertToTrailingStop:

    @pytest.mark.asyncio
    @patch("trading.engine.housekeeping.log_order")
    @patch("trading.engine.housekeeping.save_position_meta")
    @patch("trading.engine.housekeeping.place_trailing_stop", new_callable=AsyncMock)
    @patch("trading.engine.housekeeping.cancel_exit_orders", new_callable=AsyncMock)
    @patch("trading.engine.housekeeping._check_broker_position", new_callable=AsyncMock)
    async def test_converts_position(
        self, mock_broker_pos, mock_cancel, mock_place_trail, mock_save, mock_log,
    ):
        """Full convert flow: cancel OCO → place trailing stop → update position."""
        from trading.engine.housekeeping import convert_to_trailing_stop

        mock_broker_pos.return_value = 10.0  # Broker still holds shares
        mock_place_trail.return_value = "trail_order_999"
        pos = _make_pos(entry=270.0, target=280.0, stop=262.0, trailing=265.0)

        event = await convert_to_trailing_stop("FDX", pos, 281.0)

        # OCO was cancelled
        mock_cancel.assert_called_once_with("t_123", "s_456")

        # Trailing stop was placed on Alpaca
        mock_place_trail.assert_called_once_with("FDX", 10, PARAMS.let_it_ride_trail_pct)

        # Position was updated correctly
        assert pos["let_it_ride"] is True
        assert pos["target_price"] == 280.0  # original target preserved
        assert pos["target_order_id"] is None
        assert pos["trail_order_id"] == "trail_order_999"
        # Profit lock floor should be applied
        expected_floor = round(270.0 * (1 + PARAMS.profit_lock_floor_pct), 4)
        assert pos["stop_price"] >= expected_floor

        # Position was persisted
        mock_save.assert_called_once_with(pos)

        # Event was returned
        assert isinstance(event, LetItRideEvent)
        assert event.ticker == "FDX"
        assert event.trail_order_id == "trail_order_999"
        assert event.pnl_pct > 0

    @pytest.mark.asyncio
    @patch("trading.engine.housekeeping.log_order")
    @patch("trading.engine.housekeeping.save_position_meta")
    @patch("trading.engine.housekeeping.place_trailing_stop", new_callable=AsyncMock)
    @patch("trading.engine.housekeeping.cancel_exit_orders", new_callable=AsyncMock)
    @patch("trading.engine.housekeeping._check_broker_position", new_callable=AsyncMock)
    async def test_trailing_stop_failure_still_updates_position(
        self, mock_broker_pos, mock_cancel, mock_place_trail, mock_save, mock_log,
    ):
        """If Alpaca trailing stop fails, position still converts (internal stop protects)."""
        from trading.engine.housekeeping import convert_to_trailing_stop

        mock_broker_pos.return_value = 10.0  # Broker still holds shares
        mock_place_trail.return_value = None  # Alpaca call failed
        pos = _make_pos(entry=270.0, target=280.0)

        event = await convert_to_trailing_stop("FDX", pos, 281.0)

        assert pos["let_it_ride"] is True
        assert pos["trail_order_id"] is None
        assert event is not None
        assert event.trail_order_id is None

    @pytest.mark.asyncio
    @patch("trading.engine.housekeeping.async_close_position", new_callable=AsyncMock)
    @patch("trading.engine.housekeeping.cancel_exit_orders", new_callable=AsyncMock)
    @patch("trading.engine.housekeeping._check_broker_position", new_callable=AsyncMock)
    async def test_oco_already_filled_closes_position(
        self, mock_broker_pos, mock_cancel, mock_close,
    ):
        """If OCO target already filled at broker, close in DB instead of placing trailing stop."""
        from trading.engine.housekeeping import convert_to_trailing_stop
        from trading.events import ExitResult

        mock_broker_pos.return_value = None  # No position at broker — OCO already sold
        mock_close.return_value = {
            "ticker": "FDX",
            "realized_pnl_pct": 3.7,
            "bars_held": 2,
        }
        pos = _make_pos(entry=270.0, target=280.0, stop=262.0, trailing=265.0)

        with patch(
            "integrations.alpaca.fills.get_recent_sell_fills",
            new_callable=AsyncMock,
            return_value={"FDX": 280.5},
        ):
            event = await convert_to_trailing_stop("FDX", pos, 281.0)

        # Should return ExitResult, not LetItRideEvent
        assert isinstance(event, ExitResult)
        assert event.exit_reason == "oco_fill"
        assert event.fill_price == 280.5
        assert event.realized_pnl_pct == 3.7

        # Should have closed in DB with the broker fill price
        mock_close.assert_called_once_with("FDX", 280.5, "oco_fill")

    @pytest.mark.asyncio
    @patch("trading.engine.housekeeping.cancel_exit_orders", new_callable=AsyncMock)
    async def test_crypto_skips_let_it_ride(self, mock_cancel):
        """Crypto positions should return None (fall through to normal sell)."""
        from trading.engine.housekeeping import convert_to_trailing_stop

        pos = _make_pos(ticker="BTC-USD", entry=60000.0, target=62000.0)

        with patch("trading.engine.housekeeping.config") as mock_config:
            mock_config.ASSETS = {"BTC-USD": MagicMock(asset_class="crypto")}
            event = await convert_to_trailing_stop("BTC-USD", pos, 62500.0)

        assert event is None
        mock_cancel.assert_not_called()


# ---------------------------------------------------------------------------
# LetItRideEvent model
# ---------------------------------------------------------------------------

class TestLetItRideEvent:

    def test_event_fields(self):
        event = LetItRideEvent(
            ticker="FDX",
            entry_price=270.0,
            target_price=280.0,
            current_price=281.0,
            trail_pct=1.5,
            trail_order_id="trail_123",
            pnl_pct=4.07,
        )
        assert event.ticker == "FDX"
        assert event.trail_pct == 1.5
        d = event.model_dump()
        assert "trail_order_id" in d


# ---------------------------------------------------------------------------
# Discord embed
# ---------------------------------------------------------------------------

class TestLetItRideEmbed:

    def test_embed_structure(self):
        from integrations.discord.embeds import _let_it_ride_embed

        event_dict = {
            "ticker": "FDX",
            "entry_price": 270.0,
            "target_price": 280.0,
            "current_price": 281.0,
            "trail_pct": 1.5,
            "trail_order_id": "abc12345-6789",
            "pnl_pct": 4.07,
        }
        embed = _let_it_ride_embed(event_dict)
        assert "Let It Ride" in embed["title"]
        assert embed["color"] == 0x00FF00
        assert len(embed["fields"]) == 1
        assert "Target hit" in embed["fields"][0]["value"]
        assert "1.5%" in embed["fields"][0]["value"]
