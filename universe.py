"""
Universe management — S&P 500 + Nasdaq 100 tickers.

Fetches from Wikipedia and caches locally. Falls back to hardcoded S&P 100
if fetch fails.
"""

import io
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

CACHE_FILE = Path(__file__).parent / "models" / "universe_cache.json"
CACHE_MAX_AGE_DAYS = 30

# Hardcoded fallback (S&P 100)
SP100_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "CSCO",
    "INTC", "INTU", "QCOM", "TXN", "IBM",
    "GOOG", "GOOGL", "META", "NFLX", "CMCSA", "DIS", "TMUS", "CHTR", "VZ", "T",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TGT", "BKNG", "F", "GM",
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KHC",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "AXP", "USB",
    "BK", "COF", "MET", "SPG",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "AMGN",
    "GILD", "MDT", "BMY", "CVS", "DHR",
    "CAT", "HON", "UNP", "BA", "GE", "RTX", "LMT", "DE", "EMR",
    "UPS", "FDX", "GD", "MMM",
    "XOM", "CVX", "COP", "DOW",
    "NEE", "DUK", "SO", "EXC",
    "BRK-B", "LIN", "MA", "V", "PYPL", "AIG",
]


_HEADERS = {"User-Agent": "Mozilla/5.0 (market-rally-detector)"}


def _read_wiki(url: str) -> str:
    """Fetch HTML from Wikipedia with proper headers."""
    req = Request(url, headers=_HEADERS)
    with urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


def _fetch_sp500() -> list:
    """Fetch S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = _read_wiki(url)
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]
    tickers = df["Symbol"].str.strip().tolist()
    # Fix Wikipedia formatting (BRK.B -> BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def _fetch_nasdaq100() -> list:
    """Fetch Nasdaq 100 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = _read_wiki(url)
    tables = pd.read_html(io.StringIO(html))
    for table in tables:
        for col in ["Ticker", "Symbol"]:
            if col in table.columns:
                tickers = table[col].str.strip().tolist()
                tickers = [t.replace(".", "-") for t in tickers]
                return tickers
    return []


def _load_cache() -> dict | None:
    """Load cached universe if fresh enough."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        cached_date = datetime.fromisoformat(cache["fetched_at"])
        age_days = (datetime.now() - cached_date).days
        if age_days <= CACHE_MAX_AGE_DAYS:
            return cache
    except Exception:
        pass
    return None


def _save_cache(tickers: list, source: str):
    """Save universe to cache file."""
    CACHE_FILE.parent.mkdir(exist_ok=True)
    cache = {
        "tickers": tickers,
        "source": source,
        "count": len(tickers),
        "fetched_at": datetime.now().isoformat(),
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_universe(force_refresh: bool = False) -> list:
    """
    Get the combined S&P 500 + Nasdaq 100 universe.
    Caches for 30 days. Falls back to S&P 100 if fetch fails.
    """
    # Check cache first
    if not force_refresh:
        cache = _load_cache()
        if cache:
            return cache["tickers"]

    # Fetch fresh
    all_tickers = set()
    sources = []

    try:
        sp500 = _fetch_sp500()
        all_tickers.update(sp500)
        sources.append(f"S&P 500 ({len(sp500)})")
        print(f"    Fetched S&P 500: {len(sp500)} tickers")
    except Exception as e:
        print(f"    WARNING: Could not fetch S&P 500: {e}")

    try:
        ndx100 = _fetch_nasdaq100()
        all_tickers.update(ndx100)
        sources.append(f"Nasdaq 100 ({len(ndx100)})")
        print(f"    Fetched Nasdaq 100: {len(ndx100)} tickers")
    except Exception as e:
        print(f"    WARNING: Could not fetch Nasdaq 100: {e}")

    if all_tickers:
        tickers = sorted(all_tickers)
        _save_cache(tickers, " + ".join(sources))
        print(f"    Universe: {len(tickers)} unique tickers (cached)")
        return tickers

    # Fallback
    print("    WARNING: Fetch failed, using S&P 100 fallback")
    return SP100_TICKERS


def get_universe() -> list:
    """Get the current universe (from cache or fetch)."""
    return fetch_universe(force_refresh=False)
