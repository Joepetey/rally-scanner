"""Tests for position sync lifecycle — trading/positions.py (MIC-95).

Tests sync_positions_from_alpaca reconciliation, update_position_for_price,
close_position_intraday, update_fill_prices, exposure calculations, and
add_signal_positions group cap enforcement.

All tests mock the DB layer so no TEST_DATABASE_URL is required.
"""

from unittest.mock import AsyncMock, patch

import pytest

from config import PARAMS, TICKER_TO_GROUP
from trading.positions import (
    add_signal_positions,
    close_position_intraday,
    get_group_exposure,
    get_total_exposure,
    sync_positions_from_alpaca,
    update_fill_prices,
    update_position_for_price,
)

# ---------------------------------------------------------------------------
# update_position_for_price — profit lock (pure logic, no DB)
# ---------------------------------------------------------------------------


class TestUpdatePositionForPrice:

    def test_trailing_stop_ratchets_up(self):
        pos = {
            "ticker": "AAPL",
            "entry_price": 100.0,
            "highest_close": 100.0,
            "trailing_stop": 97.0,
            "stop_price": 95.0,
            "atr": 2.0,
        }
        changed = update_position_for_price(pos, 110.0)
        assert changed is True
        assert pos["highest_close"] == 110.0
        assert pos["trailing_stop"] > 97.0  # ratcheted up

    def test_trailing_stop_never_moves_down(self):
        pos = {
            "ticker": "AAPL",
            "entry_price": 100.0,
            "highest_close": 115.0,
            "trailing_stop": 112.0,
            "stop_price": 95.0,
            "atr": 2.0,
        }
        old_trail = pos["trailing_stop"]
        # Price drops below highest but still above trail — trail unchanged
        update_position_for_price(pos, 113.0)
        assert pos["trailing_stop"] == old_trail

    def test_profit_lock_raises_stop(self):
        entry = 100.0
        lock_price = round(entry * (1 + PARAMS.profit_lock_pct), 4)
        pos = {
            "ticker": "AAPL",
            "entry_price": entry,
            "highest_close": entry,
            "trailing_stop": 97.0,
            "stop_price": 95.0,
            "atr": 2.0,
        }
        # Price reaches profit lock level
        changed = update_position_for_price(pos, lock_price + 1)
        assert changed is True
        assert pos["stop_price"] >= lock_price

    def test_profit_lock_does_not_lower_stop(self):
        entry = 100.0
        pos = {
            "ticker": "AAPL",
            "entry_price": entry,
            "highest_close": 120.0,
            "trailing_stop": 117.0,
            "stop_price": 110.0,  # already above lock level
            "atr": 2.0,
        }
        old_stop = pos["stop_price"]
        update_position_for_price(pos, entry * (1 + PARAMS.profit_lock_pct) + 0.01)
        # Stop should not be lowered
        assert pos["stop_price"] >= old_stop

    def test_no_change_when_price_below_highest(self):
        pos = {
            "ticker": "AAPL",
            "entry_price": 100.0,
            "highest_close": 110.0,
            "trailing_stop": 107.0,
            "stop_price": 95.0,
            "atr": 2.0,
        }
        changed = update_position_for_price(pos, 105.0)
        # Price is below highest_close so no trailing ratchet
        # If below profit lock too, no change
        if 105.0 < 100.0 * (1 + PARAMS.profit_lock_pct):
            assert changed is False


# ---------------------------------------------------------------------------
# close_position_intraday (mock DB)
# ---------------------------------------------------------------------------


class TestClosePositionIntraday:

    @patch("trading.positions.record_closed_position")
    @patch("trading.positions.delete_position_meta")
    @patch("trading.positions.load_position_meta")
    def test_closes_and_records_pnl(self, mock_load, mock_delete, mock_record):
        mock_load.return_value = {
            "ticker": "AAPL", "entry_price": 100.0, "entry_date": "2024-01-10",
            "stop_price": 95.0, "target_price": 110.0, "trailing_stop": 98.0,
            "highest_close": 105.0, "atr": 2.0, "bars_held": 3,
            "size": 0.10, "status": "open",
        }

        result = close_position_intraday("AAPL", 108.0, "profit_target")
        assert result is not None
        assert result["exit_reason"] == "profit_target"
        assert result["exit_price"] == 108.0
        assert result["realized_pnl_pct"] == 8.0  # (108/100 - 1) * 100
        assert result["status"] == "closed"

        mock_delete.assert_called_once_with("AAPL")
        mock_record.assert_called_once()

    @patch("trading.positions.load_position_meta", return_value=None)
    def test_returns_none_for_unknown_ticker(self, mock_load):
        result = close_position_intraday("NONEXISTENT", 100.0, "stop")
        assert result is None


# ---------------------------------------------------------------------------
# update_fill_prices (mock DB)
# ---------------------------------------------------------------------------


class TestUpdateFillPrices:

    @pytest.mark.asyncio
    @patch("trading.positions.save_position_meta")
    @patch("trading.positions.load_all_position_meta")
    async def test_updates_entry_price_from_fill(self, mock_load_all, mock_save):
        mock_load_all.return_value = [{
            "ticker": "AAPL", "entry_price": 0.0, "entry_date": "2024-01-10",
            "stop_price": 95.0, "target_price": 110.0, "trailing_stop": 98.0,
            "highest_close": 100.0, "atr": 2.0, "bars_held": 0,
            "size": 0.10, "order_id": "ord_123", "status": "open",
        }]

        fills = {"ord_123": 101.5}
        count = await update_fill_prices(fills)
        assert count == 1

        saved_pos = mock_save.call_args[0][0]
        assert saved_pos["entry_price"] == 101.5
        assert saved_pos["order_id"] is None
        assert saved_pos["highest_close"] == 101.5

    @pytest.mark.asyncio
    @patch("trading.positions.save_position_meta")
    @patch("trading.positions.load_all_position_meta")
    async def test_no_update_when_no_matching_order(self, mock_load_all, mock_save):
        mock_load_all.return_value = [{
            "ticker": "AAPL", "entry_price": 100.0, "entry_date": "2024-01-10",
            "stop_price": 95.0, "target_price": 110.0, "trailing_stop": 98.0,
            "highest_close": 100.0, "atr": 2.0, "bars_held": 0,
            "size": 0.10, "order_id": "ord_123", "status": "open",
        }]

        fills = {"ord_999": 101.5}  # different order_id
        count = await update_fill_prices(fills)
        assert count == 0
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# get_total_exposure / get_group_exposure (mock DB)
# ---------------------------------------------------------------------------


class TestExposure:

    @patch("trading.positions.load_all_position_meta")
    def test_total_exposure(self, mock_load_all):
        mock_load_all.return_value = [
            {"ticker": "AAPL", "size": 0.10},
            {"ticker": "MSFT", "size": 0.08},
            {"ticker": "NVDA", "size": 0.12},
        ]

        total = get_total_exposure()
        assert abs(total - 0.30) < 0.001

    @patch("trading.positions.load_all_position_meta")
    def test_group_exposure(self, mock_load_all):
        mock_load_all.return_value = [
            {"ticker": "AAPL", "size": 0.10},
            {"ticker": "MSFT", "size": 0.08},
            {"ticker": "JPM", "size": 0.05},
        ]

        group = TICKER_TO_GROUP.get("AAPL")
        if group:
            count, exposure = get_group_exposure(group)
            assert count >= 1
            assert exposure > 0


# ---------------------------------------------------------------------------
# add_signal_positions — group cap enforcement (mock DB)
# ---------------------------------------------------------------------------


class TestAddSignalGroupCaps:

    @patch("trading.positions.save_position_meta")
    def test_respects_group_cap(self, mock_save):
        """Adding more than max_group_positions to same group is blocked."""
        state = {"positions": [], "closed_today": []}

        # Fill up group with max_group_positions
        for i in range(PARAMS.max_group_positions):
            ticker = f"MEGA{i}"
            TICKER_TO_GROUP[ticker] = "mega_tech"
            state["positions"].append({
                "ticker": ticker, "entry_price": 100.0, "size": 0.05,
                "status": "open", "entry_date": "2024-01-10",
            })

        # Try to add another in same group
        TICKER_TO_GROUP["MEGA_NEW"] = "mega_tech"
        signals = [{
            "ticker": "MEGA_NEW", "date": "2024-01-15", "close": 100.0,
            "size": 0.05, "range_low": 95.0, "atr_pct": 0.02,
        }]

        updated = add_signal_positions(state, signals)
        tickers = {p["ticker"] for p in updated["positions"]}
        assert "MEGA_NEW" not in tickers

        # Cleanup
        for i in range(PARAMS.max_group_positions):
            TICKER_TO_GROUP.pop(f"MEGA{i}", None)
        TICKER_TO_GROUP.pop("MEGA_NEW", None)


# ---------------------------------------------------------------------------
# sync_positions_from_alpaca — edge cases (mock DB)
# ---------------------------------------------------------------------------


class TestSyncEdgeCases:

    @pytest.mark.asyncio
    @patch("trading.positions.record_closed_position")
    @patch("trading.positions.delete_position_meta")
    @patch("trading.positions.save_position_meta")
    @patch("trading.positions.load_all_position_meta")
    async def test_fill_lookup_returns_zero(
        self, mock_load_all, mock_save, mock_delete, mock_record,
    ):
        """Case 3 with fill_price=0: position still closed deterministically."""
        mock_load_all.return_value = [{
            "ticker": "FAIL", "entry_price": 100.0, "entry_date": "2024-01-10",
            "stop_price": 95.0, "target_price": 110.0, "trailing_stop": 98.0,
            "highest_close": 105.0, "atr": 2.0, "bars_held": 3, "size": 0.08, "qty": 5,
        }]

        with patch("trading.positions.get_all_positions", AsyncMock(return_value=[])), \
             patch("integrations.alpaca.executor.get_recent_sell_fills", AsyncMock(return_value={})):
            result = await sync_positions_from_alpaca()

        assert result["closed"] == 1
        mock_delete.assert_called_once_with("FAIL")
        recorded = mock_record.call_args[0][0]
        assert recorded["exit_price"] == 0.0
        assert recorded["realized_pnl_pct"] == 0.0

    @pytest.mark.asyncio
    @patch("trading.positions.record_closed_position")
    @patch("trading.positions.delete_position_meta")
    @patch("trading.positions.save_position_meta")
    @patch("trading.positions.load_all_position_meta")
    async def test_sync_with_equity_computes_size(
        self, mock_load_all, mock_save, mock_delete, mock_record,
    ):
        """Case 2: untracked position with equity > 0 computes size correctly."""
        mock_load_all.return_value = []  # no DB positions

        mock_broker = AsyncMock(return_value=[{
            "ticker": "NEW", "qty": 10, "avg_entry_price": 50.0,
            "market_value": 500.0, "unrealized_pl": 0.0,
        }])
        with patch("trading.positions.get_all_positions", mock_broker), \
             patch("integrations.alpaca.executor.get_recent_sell_fills", AsyncMock(return_value={})):
            await sync_positions_from_alpaca(equity=10000.0)

        saved_pos = mock_save.call_args[0][0]
        assert saved_pos["ticker"] == "NEW"
        assert saved_pos["size"] == round(500.0 / 10000.0, 4)
