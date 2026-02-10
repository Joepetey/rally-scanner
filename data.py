"""
Data fetching — daily OHLCV via yfinance.
"""

import pandas as pd
import yfinance as yf

from config import AssetConfig

# Session-level cache for VIX data (fetched once, reused across tickers)
_vix_cache = {}


def fetch_daily(asset: AssetConfig, start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Return a DataFrame with columns: Open, High, Low, Close, Volume
    indexed by Date (tz-naive).
    """
    ticker = yf.Ticker(asset.ticker)
    kwargs = {"start": start, "interval": "1d", "auto_adjust": True}
    if end:
        kwargs["end"] = end
    df = ticker.history(**kwargs)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df.sort_index()
    return df


def fetch_vix(start: str = "2000-01-01", end: str | None = None) -> pd.Series:
    """Fetch VIX close prices. Cached per session to avoid redundant fetches."""
    cache_key = f"{start}_{end}"
    if cache_key in _vix_cache:
        return _vix_cache[cache_key]

    ticker = yf.Ticker("^VIX")
    kwargs = {"start": start, "interval": "1d", "auto_adjust": True}
    if end:
        kwargs["end"] = end
    df = ticker.history(**kwargs)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    vix = df["Close"].rename("VIX_Close")
    _vix_cache[cache_key] = vix
    return vix


def merge_vix(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Merge VIX data into a stock DataFrame by date (forward-filled)."""
    df = df.copy()
    vix_aligned = vix.reindex(df.index, method="ffill")
    df["VIX_Close"] = vix_aligned
    return df
