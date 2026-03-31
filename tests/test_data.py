"""Tests for data fetching — core/data.py (MIC-93).

Mocks yfinance to verify DataFrame shapes, column schemas, and failure handling.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from config import AssetConfig
from core.data import (
    OHLCV_COLS,
    fetch_daily,
    fetch_daily_batch,
    fetch_quotes,
    fetch_vix,
    fetch_vix_safe,
    merge_vix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, start: str = "2023-01-02") -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with proper structure."""
    np.random.seed(1)
    dates = pd.bdate_range(start, periods=n)
    close = 100 + np.cumsum(np.random.normal(0, 0.5, n))
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


ASSET = AssetConfig(ticker="AAPL", asset_class="equity", r_up=0.03, d_dn=0.015)


# ---------------------------------------------------------------------------
# fetch_daily
# ---------------------------------------------------------------------------


class TestFetchDaily:

    @patch("core.data.yf.Ticker")
    def test_returns_ohlcv_columns(self, mock_ticker_cls):
        df = _make_ohlcv()
        # yfinance Ticker.history can return extra columns; fetch_daily selects OHLCV_COLS
        df["Dividends"] = 0.0
        df["Stock Splits"] = 0.0
        mock_ticker_cls.return_value.history.return_value = df.copy()

        result = fetch_daily(ASSET, start="2023-01-02")
        assert list(result.columns) == OHLCV_COLS

    @patch("core.data.yf.Ticker")
    def test_index_sorted_ascending(self, mock_ticker_cls):
        df = _make_ohlcv()
        # Shuffle the index to simulate unsorted data
        mock_ticker_cls.return_value.history.return_value = df.iloc[::-1].copy()

        result = fetch_daily(ASSET, start="2023-01-02")
        assert result.index.is_monotonic_increasing

    @patch("core.data.yf.Ticker")
    def test_index_is_tz_naive(self, mock_ticker_cls):
        df = _make_ohlcv()
        df.index = df.index.tz_localize("UTC")
        mock_ticker_cls.return_value.history.return_value = df

        result = fetch_daily(ASSET, start="2023-01-02")
        assert result.index.tz is None

    @patch("core.data.yf.Ticker")
    def test_empty_download_returns_empty_df(self, mock_ticker_cls):
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        mock_ticker_cls.return_value.history.return_value = empty

        result = fetch_daily(ASSET, start="2023-01-02")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("core.data.yf.Ticker")
    def test_drops_todays_partial_bar(self, mock_ticker_cls):
        today = pd.Timestamp.now().normalize()
        dates = pd.bdate_range(end=today, periods=10)
        close = np.arange(100, 110, dtype=float)
        df = pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": [1_000_000] * 10,
        }, index=dates)
        mock_ticker_cls.return_value.history.return_value = df

        result = fetch_daily(ASSET)
        # Today's bar should be dropped
        if today in dates:
            assert today not in result.index


# ---------------------------------------------------------------------------
# fetch_vix / fetch_vix_safe
# ---------------------------------------------------------------------------


class TestFetchVix:

    @patch("core.data.yf.Ticker")
    def test_returns_series_named_vix_close(self, mock_ticker_cls):
        from core.data import _vix_cache
        _vix_cache.clear()

        df = _make_ohlcv(50)
        mock_ticker_cls.return_value.history.return_value = df

        result = fetch_vix(start="2023-01-02")
        assert isinstance(result, pd.Series)
        assert result.name == "VIX_Close"

    @patch("core.data.yf.Ticker")
    def test_fetch_vix_safe_returns_none_on_error(self, mock_ticker_cls):
        from core.data import _vix_cache
        _vix_cache.clear()

        mock_ticker_cls.return_value.history.side_effect = Exception("network error")
        result = fetch_vix_safe(start="2023-01-02")
        assert result is None

    @patch("core.data.yf.Ticker")
    def test_fetch_vix_safe_returns_series_on_success(self, mock_ticker_cls):
        from core.data import _vix_cache
        _vix_cache.clear()

        df = _make_ohlcv(50)
        mock_ticker_cls.return_value.history.return_value = df

        result = fetch_vix_safe(start="2023-01-02")
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# merge_vix
# ---------------------------------------------------------------------------


class TestMergeVix:

    def test_adds_vix_column(self):
        df = _make_ohlcv(20)
        vix = pd.Series(np.random.uniform(15, 25, 20), index=df.index, name="VIX_Close")

        result = merge_vix(df, vix)
        assert "VIX_Close" in result.columns
        assert len(result) == len(df)

    def test_forward_fills_missing_dates(self):
        df = _make_ohlcv(10)
        # VIX has fewer dates (gaps)
        vix = pd.Series([20.0, 22.0], index=[df.index[0], df.index[5]], name="VIX_Close")

        result = merge_vix(df, vix)
        assert "VIX_Close" in result.columns
        # After ffill, all rows from index[0] onwards should have values
        assert result["VIX_Close"].iloc[3] == 20.0  # forward-filled

    def test_does_not_modify_original(self):
        df = _make_ohlcv(10)
        vix = pd.Series(np.ones(10), index=df.index, name="VIX_Close")
        original_cols = list(df.columns)

        merge_vix(df, vix)
        assert list(df.columns) == original_cols  # original unchanged


# ---------------------------------------------------------------------------
# fetch_daily_batch
# ---------------------------------------------------------------------------


class TestFetchDailyBatch:

    @patch("core.data.yf.download")
    def test_multi_ticker_returns_dict(self, mock_download):
        df_aapl = _make_ohlcv(100)
        df_msft = _make_ohlcv(100)

        # Build MultiIndex columns for multi-ticker response
        arrays = [
            ["AAPL"] * 5 + ["MSFT"] * 5,
            OHLCV_COLS * 2,
        ]
        cols = pd.MultiIndex.from_arrays(arrays)
        data = np.column_stack([df_aapl[OHLCV_COLS].values, df_msft[OHLCV_COLS].values])
        raw = pd.DataFrame(data, index=df_aapl.index, columns=cols)
        mock_download.return_value = raw

        result = fetch_daily_batch(["AAPL", "MSFT"], use_cache=False)
        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result
        assert list(result["AAPL"].columns) == OHLCV_COLS

    @patch("core.data.yf.download")
    def test_single_ticker_flat_columns(self, mock_download):
        """Single ticker: yfinance may return flat (non-MultiIndex) columns."""
        df = _make_ohlcv(100)
        mock_download.return_value = df.copy()

        result = fetch_daily_batch(["AAPL"], use_cache=False)
        assert "AAPL" in result
        assert list(result["AAPL"].columns) == OHLCV_COLS

    @patch("core.data.yf.download")
    @patch("core.data.fetch_daily")
    def test_batch_failure_falls_back_to_sequential(self, mock_fetch, mock_download):
        """When batch download raises, fallback to individual fetch_daily calls."""
        mock_download.side_effect = Exception("batch failed")
        mock_fetch.return_value = _make_ohlcv(100)

        result = fetch_daily_batch(["AAPL"], use_cache=False)
        assert "AAPL" in result
        mock_fetch.assert_called_once()

    @patch("core.data.yf.download")
    def test_empty_batch_raises_to_fallback(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        # Empty DataFrame triggers ValueError, caught internally -> fallback
        # With no fetch_daily mock, tickers just won't appear
        result = fetch_daily_batch(["AAPL"], use_cache=False)
        # Result should be empty dict since fallback also fails without mock
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# fetch_quotes
# ---------------------------------------------------------------------------


class TestFetchQuotes:

    @patch("core.data.yf.Ticker")
    def test_returns_price_data(self, mock_ticker_cls):
        fi_data = {
            "lastPrice": 150.0, "regularMarketPreviousClose": 148.0,
            "open": 149.0, "dayHigh": 152.0, "dayLow": 147.0,
            "lastVolume": 5_000_000, "marketCap": 2_000_000_000_000,
            "currency": "USD",
        }
        mock_fi = MagicMock()
        mock_fi.__getitem__ = lambda self, k: fi_data[k]
        mock_fi.get = lambda k, d=None: fi_data.get(k, d)
        mock_ticker_cls.return_value.fast_info = mock_fi

        result = fetch_quotes(["AAPL"])
        assert "AAPL" in result
        assert result["AAPL"]["price"] == 150.0
        assert result["AAPL"]["change"] == round(150.0 - 148.0, 2)

    @patch("core.data.yf.Ticker")
    def test_failed_ticker_returns_error_dict(self, mock_ticker_cls):
        mock_ticker_cls.return_value.fast_info.__getitem__ = MagicMock(
            side_effect=KeyError("no data")
        )
        result = fetch_quotes(["BAD"])
        assert "error" in result["BAD"]

    def test_empty_list(self):
        result = fetch_quotes([])
        assert result == {}

    @patch("core.data.yf.Ticker")
    def test_partial_failure(self, mock_ticker_cls):
        """One ticker fails, others succeed — result has both."""
        call_count = 0

        def ticker_factory(symbol):
            nonlocal call_count
            call_count += 1
            mock_t = MagicMock()
            if symbol == "BAD":
                mock_t.fast_info.__getitem__ = MagicMock(side_effect=KeyError("no data"))
            else:
                fi_data = {
                    "lastPrice": 100.0, "regularMarketPreviousClose": 99.0,
                    "open": 99.5, "dayHigh": 101.0, "dayLow": 98.0,
                    "lastVolume": 1_000_000,
                }
                mock_fi = MagicMock()
                mock_fi.__getitem__ = lambda self, k: fi_data[k]
                mock_fi.get = lambda k, d=None: fi_data.get(k, d)
                mock_t.fast_info = mock_fi
            return mock_t

        mock_ticker_cls.side_effect = ticker_factory

        result = fetch_quotes(["GOOD", "BAD"])
        assert "GOOD" in result
        assert result["GOOD"]["price"] == 100.0
        assert "error" in result["BAD"]
