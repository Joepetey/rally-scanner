"""Tests for position sync lifecycle — trading/positions.py (MIC-95).

Tests sync_positions_from_alpaca reconciliation, update_position_for_price,
close_position_intraday, update_fill_prices, exposure calculations, and
add_signal_positions group cap enforcement.

All tests mock the DB layer so no TEST_DATABASE_URL is required.
"""

from unittest.mock import patch

import pytest

from config import PARAMS, TICKER_TO_GROUP
from trading.positions import (
    add_signal_positions,
    close_position_intraday,
    get_total_exposure,
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



# ---------------------------------------------------------------------------
# close_position_intraday (mock DB)
# ---------------------------------------------------------------------------


class TestClosePositionIntraday:

    @patch("trading.position_exits.record_closed_position")
    @patch("trading.position_exits.delete_position_meta")
    @patch("trading.position_exits.load_position_meta")
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



# ---------------------------------------------------------------------------
# update_fill_prices (mock DB)
# ---------------------------------------------------------------------------


class TestUpdateFillPrices:

    @pytest.mark.asyncio
    @patch("trading.position_exits.save_position_meta")
    @patch("trading.position_exits.load_all_position_meta")
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



# ---------------------------------------------------------------------------
# get_total_exposure / get_group_exposure (mock DB)
# ---------------------------------------------------------------------------


class TestExposure:

    @patch("trading.position_exits.load_all_position_meta")
    def test_total_exposure(self, mock_load_all):
        mock_load_all.return_value = [
            {"ticker": "AAPL", "size": 0.10},
            {"ticker": "MSFT", "size": 0.08},
            {"ticker": "NVDA", "size": 0.12},
        ]

        total = get_total_exposure()
        assert abs(total - 0.30) < 0.001



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


