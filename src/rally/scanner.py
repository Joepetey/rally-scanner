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
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import PARAMS, AssetConfig
from .data import fetch_daily, fetch_daily_batch, fetch_vix, merge_vix
from .features import build_features
from .hmm import predict_hmm_probs
from .persistence import load_manifest, load_model
from .positions import load_positions, print_positions, update_positions
from .trading import compute_position_size, generate_signals

warnings.filterwarnings("ignore")

LOOKBACK_DAYS = 500  # enough for 252-bar percentile window + buffer

# Named configurations (match backtest_universe.py)
CONFIGS = {
    "baseline": {
        "p_rally_threshold": 0.50, "comp_score_threshold": 0.55,
        "vol_target_k": 0.10, "max_risk_frac": 0.25,
        "profit_atr_mult": 2.0, "time_stop_bars": 10,
    },
    "conservative": {
        "p_rally_threshold": 0.55, "comp_score_threshold": 0.60,
        "vol_target_k": 0.08, "max_risk_frac": 0.15,
        "profit_atr_mult": 2.0, "time_stop_bars": 8,
    },
    "aggressive": {
        "p_rally_threshold": 0.40, "comp_score_threshold": 0.45,
        "vol_target_k": 0.12, "max_risk_frac": 0.30,
        "profit_atr_mult": 3.0, "time_stop_bars": 15,
    },
    "concentrated": {
        "p_rally_threshold": 0.55, "comp_score_threshold": 0.60,
        "vol_target_k": 0.15, "max_risk_frac": 0.40,
        "profit_atr_mult": 2.5, "time_stop_bars": 12,
    },
}


def apply_config(config_name: str) -> None:
    """Override PARAMS with a named configuration."""
    if config_name not in CONFIGS:
        print(f"  ERROR: Unknown config '{config_name}'. "
              f"Available: {', '.join(CONFIGS.keys())}")
        raise SystemExit(1)
    cfg = CONFIGS[config_name]
    for key, val in cfg.items():
        setattr(PARAMS, key, val)


def scan_single(
    ticker: str, artifacts: dict,
    vix_data: pd.Series | None = None,
    ohlcv_data: pd.DataFrame | None = None,
) -> dict:
    """Scan a single asset. Returns dict with signal info."""
    # Reconstruct AssetConfig
    ac = artifacts["asset_config"]
    asset = AssetConfig(**ac)

    # Use pre-fetched data if available, otherwise fetch individually
    if ohlcv_data is not None:
        df = ohlcv_data.copy()
    else:
        start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        df = fetch_daily(asset, start=start)

    if len(df) < 300:
        return {"ticker": ticker, "status": "insufficient_data", "bars": len(df)}

    # Merge VIX data if available
    if vix_data is not None:
        df = merge_vix(df, vix_data)

    # Build features (live mode — lagged FAIL_DN_SCORE)
    df = build_features(df, live=True)

    # HMM predictions
    hmm_model = artifacts.get("hmm_model")
    hmm_scaler = artifacts.get("hmm_scaler")
    state_order = artifacts.get("state_order")

    if hmm_model is not None:
        hmm_probs = predict_hmm_probs(hmm_model, hmm_scaler, state_order, df)
        df = df.join(hmm_probs)
    else:
        for col in ["P_compressed", "P_expanding", "HMM_transition_signal"]:
            df[col] = 0.0

    # Get the latest bar
    feature_cols = artifacts["feature_cols"]
    latest = df.iloc[[-1]]

    # Check features are available
    missing = latest[feature_cols].isna().any(axis=1).iloc[0]
    if missing:
        return {"ticker": ticker, "status": "features_incomplete"}

    # Predict rally probability
    lr = artifacts["lr_model"]
    scaler = artifacts["lr_scaler"]
    iso = artifacts["iso_calibrator"]

    X = latest[feature_cols].values
    X_s = scaler.transform(X)
    raw_prob = lr.predict_proba(X_s)[:, 1][0]
    cal_prob = float(iso.predict([raw_prob])[0])

    # Build prediction row for signal generation
    latest_pred = latest.copy()
    latest_pred["P_RALLY"] = cal_prob

    signal = generate_signals(latest_pred, require_trend=True)
    is_signal = bool(signal.iloc[0])

    size = 0.0
    if is_signal:
        size = float(compute_position_size(latest_pred).iloc[0])
        if size == 0.0:
            is_signal = False  # below minimum position size

    # Extract diagnostics
    row = df.iloc[-1]
    return {
        "ticker": ticker,
        "status": "ok",
        "date": str(row.name.date()),
        "close": round(float(row["Close"]), 2),
        "p_rally": round(cal_prob, 3),
        "p_rally_raw": round(float(raw_prob), 3),
        "comp_score": round(float(row.get("COMP_SCORE", 0)), 3),
        "fail_dn": round(float(row.get("FAIL_DN_SCORE", 0)), 3),
        "trend": int(row.get("Trend", 0)),
        "hmm_compressed": round(float(row.get("P_compressed", 0)), 3),
        "rv_pctile": round(float(row.get("p_RV", 0)), 3),
        "atr_pct": round(float(row.get("ATR_pct", 0)), 4),
        "atr": round(float(row.get("ATR", 0)), 4),
        "signal": is_signal,
        "size": round(size, 3),
        "r_up": asset.r_up,
        "d_dn": asset.d_dn,
        "range_high": round(float(row.get("RangeHigh", 0)), 2),
        "range_low": round(float(row.get("RangeLow", 0)), 2),
        "rsi": round(float(row.get("RSI", 0)), 1),
        # New features
        "golden_cross": int(row.get("GOLDEN_CROSS", 0)),
        "macd_hist": round(float(row.get("MACD_HIST", 0)), 5),
        "vol_ratio": round(float(row.get("VOL_RATIO", 1)), 2),
        "vix_pctile": round(float(row.get("VIX_PCTILE", 0.5)), 3),
        "ma50": round(float(row.get("MA50", 0)), 2),
    }


def _scan_one(args: tuple) -> dict:
    """Worker function for parallel scanning (must be top-level for pickling)."""
    ticker, vix_data, ohlcv_df = args
    try:
        artifacts = load_model(ticker)
        return scan_single(
            ticker, artifacts,
            vix_data=vix_data,
            ohlcv_data=ohlcv_df,
        )
    except Exception as e:
        return {"ticker": ticker, "status": f"error: {e}"}


def scan_all(
    tickers: list[str] | None = None, show_positions: bool = False,
    config_name: str = "baseline",
) -> list[dict]:
    # Apply config
    apply_config(config_name)

    manifest = load_manifest()
    if not manifest:
        print("  ERROR: No trained models found. Run retrain.py first.")
        return

    if tickers:
        scan_tickers = [t for t in tickers if t in manifest]
        missing = [t for t in tickers if t not in manifest]
        if missing:
            print(f"  WARNING: No models for: {', '.join(missing)}")
    else:
        scan_tickers = sorted(manifest.keys())

    # Fetch VIX data once for all tickers
    print("  Fetching VIX data...", end="\r", flush=True)
    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    try:
        vix_data = fetch_vix(start=start)
    except Exception as e:
        print(f"  WARNING: VIX data unavailable: {e}")
        vix_data = None

    # Batch-fetch OHLCV for all tickers at once (much faster than individual fetches)
    print("  Batch-fetching OHLCV data...", end="\r", flush=True)
    try:
        ohlcv_cache = fetch_daily_batch(scan_tickers, start=start)
    except Exception as e:
        print(f"  WARNING: Batch fetch failed ({e}), falling back to individual fetches")
        ohlcv_cache = {}

    print(f"\n{'='*90}")
    print(f"  RALLY DETECTOR — DAILY SCAN  [{config_name.upper()}]")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {len(scan_tickers)} assets")
    print(f"  P(rally)>{PARAMS.p_rally_threshold:.0%}  Comp>{PARAMS.comp_score_threshold}  "
          f"MaxSize={PARAMS.max_risk_frac:.0%}  ProfitATR={PARAMS.profit_atr_mult}  "
          f"TimeStop={PARAMS.time_stop_bars}")
    print(f"{'='*90}")

    # Scan all tickers in parallel using process pool
    n_workers = min(8, len(scan_tickers))
    work_items = [
        (ticker, vix_data, ohlcv_cache.get(ticker))
        for ticker in scan_tickers
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_scan_one, item): item[0]
            for item in work_items
        }
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            print(f"  Scanned {ticker:<6} ({i}/{len(scan_tickers)})...",
                  end="\r", flush=True)
            results.append(future.result())

    # Clear progress line
    print(" " * 60, end="\r")

    # Separate results
    signals = [r for r in results if r.get("signal")]
    watchlist = [r for r in results if r.get("status") == "ok"
                 and not r.get("signal")
                 and r.get("p_rally", 0) > 0.35]
    errors = [r for r in results if r.get("status") != "ok"]
    ok_results = [r for r in results if r.get("status") == "ok"]

    # --- MARKET BREADTH ---
    if ok_results:
        above_ma50 = sum(
            1 for r in ok_results
            if r["close"] > r.get("ma50", 0) and r.get("ma50", 0) > 0
        )
        above_ma200 = sum(1 for r in ok_results if r.get("trend", 0))
        golden_crosses = sum(1 for r in ok_results if r.get("golden_cross", 0))
        avg_vix_pctile = np.mean([r.get("vix_pctile", 0.5) for r in ok_results])
        breadth_50 = above_ma50 / len(ok_results)
        breadth_200 = above_ma200 / len(ok_results)

        print(f"\n  {'='*86}")
        print("  MARKET BREADTH")
        print(f"  {'='*86}")
        # Breadth bar
        b50_bar = "#" * int(breadth_50 * 40) + "." * (40 - int(breadth_50 * 40))
        b200_bar = "#" * int(breadth_200 * 40) + "." * (40 - int(breadth_200 * 40))
        n = len(ok_results)
        print(f"  Above 50-day MA:  {above_ma50:>3}/{n} ({breadth_50:.0%})  [{b50_bar}]")
        print(f"  Above 200-day MA: {above_ma200:>3}/{n} ({breadth_200:.0%})  [{b200_bar}]")
        gc_pct = golden_crosses / n
        print(f"  Golden crosses:   {golden_crosses:>3}/{n} ({gc_pct:.0%})")
        # VIX context
        if avg_vix_pctile > 0.80:
            vix_label = "EXTREME FEAR"
        elif avg_vix_pctile > 0.60:
            vix_label = "FEAR"
        elif avg_vix_pctile > 0.40:
            vix_label = "NEUTRAL"
        elif avg_vix_pctile > 0.20:
            vix_label = "GREED"
        else:
            vix_label = "EXTREME GREED"
        print(f"  VIX percentile:   {avg_vix_pctile:.0%} ({vix_label})")

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

    # --- FULL PROBABILITY RANKING ---
    if ok_results:
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
    if show_positions or signals:
        positions = load_positions()
        positions = update_positions(positions, signals, results)
        print_positions(positions)

    # Summary
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n  Scanned: {ok_count}/{len(scan_tickers)} ok, "
          f"{len(signals)} signals, {len(watchlist)} watchlist, {len(errors)} errors")
    print(f"{'='*90}\n")

    return results


def _print_signal_table(signals: list[dict]) -> None:
    header = (f"  {'Ticker':<7} {'Close':>8} {'P(rally)':>9} {'Comp':>6} "
              f"{'GC':>3} {'MACD':>7} {'Vol':>5} {'VIX':>4} "
              f"{'Size':>6} {'Stop':>8} {'Target':>8}")
    print(header)
    print(f"  {'-'*78}")
    for r in sorted(signals, key=lambda x: x["p_rally"], reverse=True):
        atr_val = r["close"] * r["atr_pct"]
        target = r["close"] + PARAMS.profit_atr_mult * atr_val
        print(f"  {r['ticker']:<7} {r['close']:>8.2f} {r['p_rally']:>8.1%} "
              f"{r['comp_score']:>6.3f} "
              f"{'Y' if r.get('golden_cross') else 'N':>3} "
              f"{r.get('macd_hist', 0):>+6.4f} "
              f"{r.get('vol_ratio', 1):>5.1f} "
              f"{r.get('vix_pctile', 0.5):>3.0%} "
              f"{r['size']:>5.1%} "
              f"{r['range_low']:>8.2f} {target:>8.2f}")


def _print_watchlist_table(watchlist: list[dict]) -> None:
    header = (f"  {'Ticker':<7} {'Close':>8} {'P(rally)':>9} {'Comp':>6} "
              f"{'Trend':>5} {'HMM_C':>6} {'RV%':>5} {'RSI':>5} {'Missing':>12}")
    print(header)
    print(f"  {'-'*67}")
    for r in sorted(watchlist, key=lambda x: x["p_rally"], reverse=True)[:20]:
        missing = []
        if r["p_rally"] <= PARAMS.p_rally_threshold:
            missing.append("P<thr")
        if r["comp_score"] <= PARAMS.comp_score_threshold:
            missing.append("Comp")
        if not r["trend"]:
            missing.append("Trend")
        missing_str = ",".join(missing) if missing else "?"

        print(f"  {r['ticker']:<7} {r['close']:>8.2f} {r['p_rally']:>8.1%} "
              f"{r['comp_score']:>6.3f} {'Y' if r['trend'] else 'N':>5} "
              f"{r['hmm_compressed']:>6.3f} {r['rv_pctile']:>5.1%} "
              f"{r['rsi']:>5.1f} {missing_str:>12}")


def _print_probability_table(results: list[dict]) -> None:
    """Print every asset ranked by P(rally) with visual probability bar."""
    p = PARAMS
    sorted_r = sorted(results, key=lambda x: x["p_rally"], reverse=True)

    header = (f"  {'#':>3} {'Ticker':<7} {'P(rally)':>9} {'':20} "
              f"{'Comp':>6} {'Trend':>5} {'HMM_C':>6} {'RV%':>5} {'RSI':>5} {'Close':>9}")
    print(header)
    print(f"  {'-'*80}")

    for i, r in enumerate(sorted_r, 1):
        prob = r["p_rally"]

        # Visual bar: 20 chars wide, filled proportionally
        bar_len = 20
        filled = int(prob * bar_len)
        # Color coding via symbols: above signal threshold vs below
        if r.get("signal"):
            bar = ">" * filled + "." * (bar_len - filled)
            tag = " <<< SIGNAL"
        elif prob > p.p_rally_threshold:
            bar = "#" * filled + "." * (bar_len - filled)
            tag = ""
        else:
            bar = "|" * filled + "." * (bar_len - filled)
            tag = ""

        print(f"  {i:>3} {r['ticker']:<7} {prob:>8.1%} [{bar}] "
              f"{r['comp_score']:>6.3f} {'Y' if r['trend'] else 'N':>5} "
              f"{r['hmm_compressed']:>6.3f} {r['rv_pctile']:>5.1%} "
              f"{r['rsi']:>5.1f} {r['close']:>9.2f}{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily rally scanner")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers to scan (default: all trained)")
    parser.add_argument("--positions", action="store_true",
                        help="Show and update open positions")
    parser.add_argument("--config", default="baseline",
                        choices=list(CONFIGS.keys()),
                        help="Trading config: baseline, conservative, aggressive, concentrated")
    args = parser.parse_args()
    scan_all(tickers=args.tickers, show_positions=args.positions, config_name=args.config)


if __name__ == "__main__":
    main()
