"""
Data fetching — daily OHLCV via yfinance, with batch download and disk caching.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import PIPELINE, AssetConfig

# Session-level cache for VIX data (fetched once, reused across tickers)
_vix_cache: dict[str, pd.Series] = {}

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Single-ticker fetch (unchanged API — used by scanner.py)
# ---------------------------------------------------------------------------

def fetch_daily(
    asset: AssetConfig, start: str = "2000-01-01", end: str | None = None,
) -> pd.DataFrame:
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
    df = df[OHLCV_COLS].dropna()
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


# ---------------------------------------------------------------------------
# Disk cache for OHLCV data (parquet per ticker)
# ---------------------------------------------------------------------------

class DataCache:
    """Transparent parquet disk cache for OHLCV data."""

    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = Path(cache_dir or PIPELINE.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace("^", "_")
        return self.cache_dir / f"{safe}.parquet"

    def get(self, ticker: str) -> pd.DataFrame | None:
        path = self._path(ticker)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except (OSError, ValueError):
            return None

    def last_date(self, ticker: str) -> pd.Timestamp | None:
        df = self.get(ticker)
        if df is None or df.empty:
            return None
        return df.index.max()

    def put(self, ticker: str, df: pd.DataFrame) -> None:
        self._path(ticker).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._path(ticker), engine="pyarrow")

    def update(self, ticker: str, new_df: pd.DataFrame) -> pd.DataFrame:
        existing = self.get(ticker)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_df.sort_index()
        self.put(ticker, combined)
        return combined


# ---------------------------------------------------------------------------
# Batch fetch (used by retrain.py and backtest_universe.py)
# ---------------------------------------------------------------------------

def fetch_daily_batch(
    tickers: list[str],
    start: str = "2000-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Batch-fetch OHLCV data for multiple tickers via yf.download().

    With caching enabled:
      1. Load each ticker from parquet cache
      2. Determine delta start date (last cached date + 1 day)
      3. Batch-download only missing data
      4. Merge new data into cache

    Returns: {ticker: DataFrame} with columns [Open, High, Low, Close, Volume]
    """
    cache = DataCache() if (use_cache and PIPELINE.cache_enabled) else None
    result: dict[str, pd.DataFrame] = {}
    tickers_to_fetch: list[str] = []
    yesterday = (datetime.now() - timedelta(days=2)).date()

    # Phase 1: Check cache
    if cache is not None:
        for ticker in tickers:
            last = cache.last_date(ticker)
            if last is not None and last.date() >= yesterday:
                cached_df = cache.get(ticker)
                if cached_df is not None and len(cached_df) > 0:
                    result[ticker] = cached_df
                    continue
            tickers_to_fetch.append(ticker)
    else:
        tickers_to_fetch = list(tickers)

    if not tickers_to_fetch:
        return result

    # Phase 2: Batch download
    print(f"  Batch downloading {len(tickers_to_fetch)} tickers "
          f"({len(result)} from cache)...", flush=True)
    try:
        raw = yf.download(
            tickers_to_fetch,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError("yf.download returned empty DataFrame")

        # Parse results
        if len(tickers_to_fetch) == 1:
            # Single ticker: flat columns
            ticker = tickers_to_fetch[0]
            cols = [c for c in OHLCV_COLS if c in raw.columns]
            if cols:
                df_t = raw[cols].dropna(how="all")
                df_t.index = pd.to_datetime(df_t.index).tz_localize(None)
                df_t = df_t.sort_index()
                if cache is not None:
                    df_t = cache.update(ticker, df_t)
                result[ticker] = df_t
        else:
            # Multi-ticker: MultiIndex columns (ticker, field)
            for ticker in tickers_to_fetch:
                try:
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    df_t = raw[ticker][OHLCV_COLS].dropna(how="all")
                    if df_t.empty:
                        continue
                    df_t.index = pd.to_datetime(df_t.index).tz_localize(None)
                    df_t = df_t.sort_index()
                    if cache is not None:
                        df_t = cache.update(ticker, df_t)
                    result[ticker] = df_t
                except (KeyError, TypeError):
                    continue

    except Exception as e:
        print(f"  WARNING: Batch download failed ({e}), falling back to sequential")
        for ticker in tickers_to_fetch:
            try:
                tmp_asset = AssetConfig(ticker=ticker, asset_class="equity",
                                        r_up=0.03, d_dn=0.015)
                df = fetch_daily(tmp_asset, start=start, end=end)
                if cache is not None:
                    df = cache.update(ticker, df)
                result[ticker] = df
                time.sleep(0.3)
            except Exception:
                continue

    return result


# ---------------------------------------------------------------------------
# Real-time quotes (used by Claude agent get_price tool)
# ---------------------------------------------------------------------------


def fetch_quotes(tickers: list[str]) -> dict[str, dict]:
    """Fetch real-time quote data for one or more tickers via yfinance fast_info.

    Returns dict keyed by uppercase ticker.  Each value is either a quote dict
    or ``{"error": "..."}`` if the ticker could not be fetched.
    """
    results: dict[str, dict] = {}
    for symbol in tickers:
        symbol = symbol.upper().strip()
        if not symbol:
            continue
        try:
            fi = yf.Ticker(symbol).fast_info
            price = fi["lastPrice"]
            prev_close = fi["regularMarketPreviousClose"]
            change = price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0.0
            results[symbol] = {
                "price": round(price, 2),
                "prev_close": round(prev_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "open": round(fi["open"], 2),
                "day_high": round(fi["dayHigh"], 2),
                "day_low": round(fi["dayLow"], 2),
                "volume": fi["lastVolume"],
                "market_cap": fi.get("marketCap"),
                "currency": fi.get("currency", "USD"),
            }
        except Exception as e:
            results[symbol] = {"error": f"Could not fetch {symbol}: {e}"}
    return results
