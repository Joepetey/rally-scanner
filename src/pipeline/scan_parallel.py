"""Parallel scan execution — process pool orchestration and data fetching."""

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import NamedTuple

import pandas as pd
from rally_ml.config.trading import TradingConfig
from rally_ml.core.data import fetch_daily_batch, fetch_quotes, fetch_vix_safe
from rally_ml.core.persistence import load_model

from pipeline.scan_core import scan_single

logger = logging.getLogger(__name__)

_MAX_SCAN_WORKERS = 8
_MAX_WATCHLIST_WORKERS = 4


class ScanTask(NamedTuple):
    ticker: str
    vix_data: "pd.Series | None"
    ohlcv_df: "pd.DataFrame | None"
    config: "TradingConfig | None"


def _scan_one(task: ScanTask) -> dict:
    """Worker function for parallel scanning (must be top-level for pickling)."""
    ticker, vix_data, ohlcv_df, config = task
    try:
        artifacts = load_model(ticker)
        return scan_single(
            ticker, artifacts,
            vix_data=vix_data,
            ohlcv_data=ohlcv_df,
            config=config,
        )
    except Exception as e:
        return {"ticker": ticker, "status": f"error: {e}"}


def fetch_scan_data(
    scan_tickers: list[str], start: str, verbose: bool = True,
) -> tuple["pd.Series | None", dict]:
    """Fetch VIX and batch OHLCV data for all tickers."""
    logger.info("Fetching VIX data...")
    vix_data = fetch_vix_safe(start=start, verbose=verbose)

    logger.info("Batch-fetching OHLCV data...")
    try:
        ohlcv_cache = fetch_daily_batch(scan_tickers, start=start)
    except Exception as e:
        logger.warning(
            "Batch fetch failed (%s), falling back to individual fetches", e,
        )
        ohlcv_cache = {}

    return vix_data, ohlcv_cache


def run_parallel_scan(
    scan_tickers: list[str],
    vix_data: "pd.Series | None",
    ohlcv_cache: dict,
    max_workers: int,
    config: TradingConfig | None = None,
) -> list[dict]:
    """Run scan across tickers using a process pool."""
    n_workers = min(max_workers, len(scan_tickers))
    work_items = [
        ScanTask(ticker, vix_data, ohlcv_cache.get(ticker), config)
        for ticker in scan_tickers
    ]

    results = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("forkserver"),
    ) as pool:
        futures = {
            pool.submit(_scan_one, item): item[0]
            for item in work_items
        }
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            logger.debug("Scanned %s (%d/%d)", ticker, i, len(scan_tickers))
            results.append(future.result())

    return results


def update_signals_with_live_prices(signals: list[dict]) -> None:
    """Replace daily close with live quote price for signal tickers (in-place)."""
    if not signals:
        return
    live_tickers = [s["ticker"] for s in signals]
    quotes = fetch_quotes(live_tickers)
    for sig in signals:
        q = quotes.get(sig["ticker"], {})
        if "price" in q:
            daily_close = sig["close"]
            sig["close"] = q["price"]
            logger.info(
                "%s: live price $%.2f (daily close was $%.2f)",
                sig["ticker"], q["price"], daily_close,
            )
