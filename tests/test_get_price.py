"""Tests for the get_price tool wrapper — unique wrapper behavior only.

Core fetch_quotes tests live in test_data.py.
"""

from unittest.mock import patch

from services.trading_ops import get_price


@patch("services.trading_ops.fetch_quotes")
def test_get_price_caps_at_10(mock_fetch):
    mock_fetch.return_value = {}
    get_price([f"T{i}" for i in range(20)])
    assert len(mock_fetch.call_args[0][0]) == 10


@patch("services.trading_ops.fetch_quotes")
def test_get_price_network_error(mock_fetch):
    mock_fetch.side_effect = ConnectionError("network down")
    result = get_price(["HD"])
    assert "error" in result
