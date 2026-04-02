"""Daily market rally scanner — public API."""

import logging
from datetime import datetime, timedelta

from rally_ml.config import CONFIGS_BY_NAME
from rally_ml.config.trading import TradingConfig
from rally_ml.core.persistence import load_manifest

from pipeline.scanner.core import LOOKBACK_DAYS, scan_single
from pipeline.scanner.parallel import (
    MAX_SCAN_WORKERS,
    MAX_WATCHLIST_WORKERS,
    fetch_scan_data,
    run_parallel_scan,
    update_signals_with_live_prices,
)

logger = logging.getLogger(__name__)

__all__ = ["resolve_config", "scan_all", "scan_single", "scan_watchlist"]


def resolve_config(config_name: str) -> TradingConfig:
    """Look up a named TradingConfig. Raises SystemExit if not found."""
    cfg = CONFIGS_BY_NAME.get(config_name)
    if cfg is None:
        logger.error(
            "Unknown config '%s'. Available: %s",
            config_name, ", ".join(CONFIGS_BY_NAME.keys()),
        )
        raise SystemExit(1)
    return cfg


def scan_all(
    tickers: list[str] | None = None,
    config_name: str = "conservative",
) -> list[dict]:
    """Scan all trained assets (or specific tickers) and return results."""
    config = resolve_config(config_name)

    manifest = load_manifest()
    if not manifest:
        logger.error("No trained models found. Run retrain.py first.")
        return []

    if tickers:
        scan_tickers = [t for t in tickers if t in manifest]
        missing = [t for t in tickers if t not in manifest]
        if missing:
            logger.warning("No models for: %s", ", ".join(missing))
    else:
        scan_tickers = sorted(manifest.keys())

    start = (
        datetime.now() - timedelta(days=LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    vix_data, ohlcv_cache = fetch_scan_data(scan_tickers, start)

    logger.info(
        "Scanning %d assets [%s] P(rally)>%.0f%% Comp>%s",
        len(scan_tickers), config_name.upper(),
        config.p_rally * 100, config.comp_score,
    )

    results = run_parallel_scan(
        scan_tickers, vix_data, ohlcv_cache, MAX_SCAN_WORKERS, config,
    )

    signals = [r for r in results if r.get("signal")]
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    err_count = sum(1 for r in results if r.get("status") != "ok")

    update_signals_with_live_prices(signals)

    logger.info(
        "Scan complete — %d/%d ok, %d signals, %d errors",
        ok_count, len(scan_tickers), len(signals), err_count,
    )

    return results


def scan_watchlist(
    tickers: list[str],
    config_name: str = "conservative",
) -> list[dict]:
    """Scan a specific set of tickers (e.g. near-threshold watchlist)."""
    config = resolve_config(config_name)

    manifest = load_manifest()
    if not manifest:
        return []

    scan_tickers = [t for t in tickers if t in manifest]
    if not scan_tickers:
        return []

    start = (
        datetime.now() - timedelta(days=LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    vix_data, ohlcv_cache = fetch_scan_data(
        scan_tickers, start, verbose=False,
    )

    results = run_parallel_scan(
        scan_tickers, vix_data, ohlcv_cache, MAX_WATCHLIST_WORKERS, config,
    )

    signals = [r for r in results if r.get("signal")]
    update_signals_with_live_prices(signals)

    return results
