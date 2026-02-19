"""
Universe management — S&P 500 + Nasdaq top 500 tickers.

Fetches from Wikipedia (S&P 500) and NASDAQ screener API (top 500 by market cap).
Caches locally. Falls back to hardcoded S&P 100 if all fetches fail.
"""

import io
import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_FILE = PROJECT_ROOT / "models" / "universe_cache.json"
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


def _fetch_sp500() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = _read_wiki(url)
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]
    tickers = df["Symbol"].str.strip().tolist()
    # Fix Wikipedia formatting (BRK.B -> BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def _fetch_nasdaq100() -> list[str]:
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


def _fetch_nasdaq_top500() -> list[str]:
    """Fetch top 500 Nasdaq-listed stocks by market cap from NASDAQ screener API."""
    url = ("https://api.nasdaq.com/api/screener/stocks"
           "?tableType=traded&exchange=nasdaq&limit=500"
           "&sortcolumn=marketCap&sortorder=desc")
    req = Request(url, headers={**_HEADERS, "Accept": "application/json"})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    rows = data["data"]["table"]["rows"]
    tickers = [r["symbol"].strip() for r in rows]
    # Filter out obvious non-common-stock entries (warrants, units, etc.)
    tickers = [t for t in tickers if t.isalpha() or "-" in t]
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


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
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _save_cache(tickers: list[str], source: str) -> None:
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


def fetch_universe(force_refresh: bool = False) -> list[str]:
    """
    Get the combined S&P 500 + Nasdaq top 500 universe.
    Caches for 30 days. Falls back to S&P 100 if all fetches fail.
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
        logger.info("Fetched S&P 500: %d tickers", len(sp500))
    except Exception as e:
        logger.warning("Could not fetch S&P 500: %s", e)

    try:
        ndx500 = _fetch_nasdaq_top500()
        all_tickers.update(ndx500)
        sources.append(f"Nasdaq Top 500 ({len(ndx500)})")
        logger.info("Fetched Nasdaq Top 500: %d tickers", len(ndx500))
    except Exception as e:
        logger.warning("Could not fetch Nasdaq Top 500: %s", e)
        # Fall back to Nasdaq 100 if screener API fails
        try:
            ndx100 = _fetch_nasdaq100()
            all_tickers.update(ndx100)
            sources.append(f"Nasdaq 100 ({len(ndx100)})")
            logger.info("Fetched Nasdaq 100 (fallback): %d tickers", len(ndx100))
        except Exception as e2:
            logger.warning("Could not fetch Nasdaq 100: %s", e2)

    if all_tickers:
        tickers = sorted(all_tickers)
        _save_cache(tickers, " + ".join(sources))
        logger.info("Universe: %d unique tickers (cached)", len(tickers))
        return tickers

    # Fallback
    logger.warning("Fetch failed, using S&P 100 fallback")
    return SP100_TICKERS


def get_universe() -> list[str]:
    """Get the current universe (from cache or fetch)."""
    return fetch_universe(force_refresh=False)
