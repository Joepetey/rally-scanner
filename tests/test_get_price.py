"""Tests for the get_price tool and fetch_quotes helper."""

import sys
from unittest.mock import MagicMock, patch

# claude_agent imports anthropic at module level; stub it before importing.
sys.modules.setdefault("anthropic", MagicMock())

from rally.bot.claude_agent import _get_price, execute_tool  # noqa: E402, I001


# ---------------------------------------------------------------------------
# _get_price unit tests
# ---------------------------------------------------------------------------


@patch("rally.core.data.fetch_quotes")
def test_get_price_single_ticker(mock_fetch):
    mock_fetch.return_value = {
        "HD": {
            "price": 392.00,
            "prev_close": 389.51,
            "change": 2.49,
            "change_pct": 0.64,
            "open": 390.20,
            "day_high": 394.00,
            "day_low": 387.10,
            "volume": 1_483_228,
            "market_cap": 380_000_000_000,
            "currency": "USD",
        }
    }
    result = _get_price(["HD"])
    assert result["count"] == 1
    assert result["quotes"]["HD"]["price"] == 392.00


@patch("rally.core.data.fetch_quotes")
def test_get_price_multiple_tickers(mock_fetch):
    mock_fetch.return_value = {
        "HD": {"price": 392.00},
        "AAPL": {"price": 258.33},
    }
    result = _get_price(["HD", "AAPL"])
    assert result["count"] == 2
    assert "HD" in result["quotes"]
    assert "AAPL" in result["quotes"]


@patch("rally.core.data.fetch_quotes")
def test_get_price_invalid_ticker(mock_fetch):
    mock_fetch.return_value = {
        "INVALIDXYZ": {"error": "Could not fetch INVALIDXYZ: KeyError"}
    }
    result = _get_price(["INVALIDXYZ"])
    assert result["count"] == 1
    assert "error" in result["quotes"]["INVALIDXYZ"]


def test_get_price_empty_list():
    result = _get_price([])
    assert "error" in result


@patch("rally.core.data.fetch_quotes")
def test_get_price_caps_at_10(mock_fetch):
    mock_fetch.return_value = {}
    _get_price([f"T{i}" for i in range(20)])
    assert len(mock_fetch.call_args[0][0]) == 10


@patch("rally.core.data.fetch_quotes")
def test_get_price_network_error(mock_fetch):
    mock_fetch.side_effect = ConnectionError("network down")
    result = _get_price(["HD"])
    assert "error" in result


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------


@patch("rally.bot.claude_agent._get_price")
def test_execute_tool_dispatches_get_price(mock_gp):
    mock_gp.return_value = {"count": 1, "quotes": {"HD": {"price": 392}}}
    result = execute_tool("get_price", {"tickers": ["HD"]}, 12345, 10000)
    mock_gp.assert_called_once_with(["HD"])
    assert result["count"] == 1


# ---------------------------------------------------------------------------
# fetch_quotes unit tests (mocked yfinance)
# ---------------------------------------------------------------------------


@patch("rally.core.data.yf.Ticker")
def test_fetch_quotes_success(mock_ticker_cls):
    from rally.core.data import fetch_quotes

    fi_data = {
        "lastPrice": 392.0,
        "regularMarketPreviousClose": 389.51,
        "open": 390.20,
        "dayHigh": 394.0,
        "dayLow": 387.1,
        "lastVolume": 1_483_228,
        "marketCap": 380_000_000_000,
        "currency": "USD",
    }
    mock_fi = MagicMock()
    mock_fi.__getitem__ = lambda self, k: fi_data[k]
    mock_fi.get = lambda k, d=None: fi_data.get(k, d)
    mock_ticker_cls.return_value.fast_info = mock_fi

    result = fetch_quotes(["HD"])
    assert "HD" in result
    assert result["HD"]["price"] == 392.0
    assert result["HD"]["change"] == round(392.0 - 389.51, 2)
    assert result["HD"]["volume"] == 1_483_228


@patch("rally.core.data.yf.Ticker")
def test_fetch_quotes_invalid_ticker(mock_ticker_cls):
    from rally.core.data import fetch_quotes

    mock_ticker_cls.return_value.fast_info.__getitem__ = MagicMock(
        side_effect=KeyError("currentTradingPeriod")
    )
    result = fetch_quotes(["INVALIDXYZ"])
    assert "error" in result["INVALIDXYZ"]


def test_fetch_quotes_empty_list():
    from rally.core.data import fetch_quotes

    result = fetch_quotes([])
    assert result == {}
