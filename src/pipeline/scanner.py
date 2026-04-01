"""
Daily market rally scanner — load models, scan assets, print alerts.

Usage:
    python scanner.py                          # scan all trained assets (baseline)
    python scanner.py --config conservative    # use conservative thresholds
    python scanner.py --config aggressive      # use aggressive thresholds
    python scanner.py --tickers AAPL MSFT SPY  # scan specific tickers
    python scanner.py --positions              # also show open positions
"""

import argparse
import logging
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

from config import CONFIGS_BY_NAME, PARAMS
from core.data import fetch_daily_batch, fetch_quotes, fetch_vix_safe
from core.persistence import load_manifest
from pipeline.scan_display import (  # noqa: F401 — re-export
    _compute_breadth,
    _print_probability_table,
    _print_signal_table,
    _print_watchlist_table,
)
from pipeline.scan_worker import (  # noqa: F401 — re-export
    LOOKBACK_DAYS,
    ScanTask,
    _scan_one,
    scan_single,
)
from trading.positions import (
    get_merged_positions_sync,
    print_positions,
    update_existing_positions,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

_MAX_SCAN_WORKERS = 8
_MAX_WATCHLIST_WORKERS = 4


def apply_config(config_name: str) -> None:
    """Override PARAMS with a named configuration."""
    cfg = CONFIGS_BY_NAME.get(config_name)
    if cfg is None:
        logger.error("Unknown config '%s'. Available: %s",
                      config_name, ", ".join(CONFIGS_BY_NAME.keys()))
        raise SystemExit(1)
    PARAMS.p_rally_threshold = cfg.p_rally
    PARAMS.comp_score_threshold = cfg.comp_score
    PARAMS.vol_target_k = cfg.vol_k
    PARAMS.max_risk_frac = cfg.max_risk
    PARAMS.profit_atr_mult = cfg.profit_atr
    PARAMS.time_stop_bars = cfg.time_stop


def _fetch_scan_data(
    scan_tickers: list[str], start: str, verbose: bool = True,
) -> tuple:
    """Fetch VIX and batch OHLCV data for all tickers. Returns (vix_data, ohlcv_cache)."""
    logger.info("Fetching VIX data...")
    vix_data = fetch_vix_safe(start=start, verbose=verbose)

    logger.info("Batch-fetching OHLCV data...")
    try:
        ohlcv_cache = fetch_daily_batch(scan_tickers, start=start)
    except Exception as e:
        logger.warning("Batch fetch failed (%s), falling back to individual fetches", e)
        ohlcv_cache = {}

    return vix_data, ohlcv_cache


def scan_all(
    tickers: list[str] | None = None, show_positions: bool = False,
    config_name: str = "conservative",
) -> list[dict]:
    # Apply config
    apply_config(config_name)

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

    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    vix_data, ohlcv_cache = _fetch_scan_data(scan_tickers, start)

    print(f"\n{'='*90}")
    print(f"  RALLY DETECTOR — DAILY SCAN  [{config_name.upper()}]")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {len(scan_tickers)} assets")
    print(f"  P(rally)>{PARAMS.p_rally_threshold:.0%}  Comp>{PARAMS.comp_score_threshold}  "
          f"MaxSize={PARAMS.max_risk_frac:.0%}  ProfitATR={PARAMS.profit_atr_mult}  "
          f"TimeStop={PARAMS.time_stop_bars}")
    print(f"{'='*90}")

    # Scan all tickers in parallel using process pool
    n_workers = min(_MAX_SCAN_WORKERS, len(scan_tickers))
    work_items = [
        ScanTask(ticker, vix_data, ohlcv_cache.get(ticker))
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

    # Separate results
    signals = [r for r in results if r.get("signal")]
    watchlist = [r for r in results if r.get("status") == "ok"
                 and not r.get("signal")
                 and r.get("p_rally", 0) > PARAMS.watchlist_p_rally_min]
    errors = [r for r in results if r.get("status") != "ok"]
    ok_results = [r for r in results if r.get("status") == "ok"]

    # Update signals with live prices so entries/stops/targets use current market price
    if signals:
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

    # --- MARKET BREADTH ---
    if ok_results:
        _compute_breadth(ok_results)

    # --- NEW SIGNALS ---
    print(f"\n  {'='*86}")
    print(f"  NEW SIGNALS ({len(signals)})")
    print(f"  {'='*86}")
    if signals:
        _print_signal_table(signals)
    else:
        print("  (none)")

    # --- WATCHLIST ---
    print(f"\n  {'='*86}")
    print(f"  WATCHLIST — near-miss, P(rally) > 35% ({len(watchlist)})")
    print(f"  {'='*86}")
    if watchlist:
        _print_watchlist_table(watchlist)
    else:
        print("  (none)")

    # --- FULL PROBABILITY RANKING --- (debug only — 800+ lines floods Railway logs)
    if ok_results and logger.isEnabledFor(logging.DEBUG):
        print(f"\n  {'='*86}")
        print(f"  RALLY PROBABILITY RANKING — all {len(ok_results)} assets")
        print(f"  {'='*86}")
        _print_probability_table(ok_results)

    # --- ERRORS ---
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for r in errors:
            print(f"    {r['ticker']}: {r['status']}")

    # --- POSITIONS ---
    if show_positions:
        positions = get_merged_positions_sync()
        positions = update_existing_positions(positions, results)
        print_positions(positions)

    # Summary
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n  Scanned: {ok_count}/{len(scan_tickers)} ok, "
          f"{len(signals)} signals, {len(watchlist)} watchlist, {len(errors)} errors")
    print(f"{'='*90}\n")

    return results


def scan_watchlist(
    tickers: list[str],
    config_name: str = "conservative",
) -> list[dict]:
    """Scan a specific set of tickers (e.g. near-threshold watchlist).

    Lighter than scan_all: fewer workers, no console output.
    Returns list of result dicts (same format as scan_all).
    """
    apply_config(config_name)

    manifest = load_manifest()
    if not manifest:
        return []

    scan_tickers = [t for t in tickers if t in manifest]
    if not scan_tickers:
        return []

    # Fetch data
    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    vix_data, ohlcv_cache = _fetch_scan_data(scan_tickers, start, verbose=False)
    n_workers = min(_MAX_WATCHLIST_WORKERS, len(scan_tickers))
    work_items = [
        ScanTask(ticker, vix_data, ohlcv_cache.get(ticker))
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
        for future in as_completed(futures):
            results.append(future.result())

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily rally scanner")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers to scan (default: all trained)")
    parser.add_argument("--positions", action="store_true",
                        help="Show and update open positions")
    parser.add_argument("--config", default="conservative",
                        choices=["conservative", "baseline", "aggressive", "concentrated"],
                        help="Trading config: conservative, baseline, aggressive, concentrated")
    args = parser.parse_args()
    scan_all(tickers=args.tickers, show_positions=args.positions, config_name=args.config)


if __name__ == "__main__":
    main()
