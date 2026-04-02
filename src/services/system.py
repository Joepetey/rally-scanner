"""System operations: model health, scanner, retrain."""

import logging
from datetime import date, datetime

from rally_ml.core.persistence import load_manifest

from db.trading.positions import save_watchlist
from pipeline.scanner import scan_all
from trading.positions import get_merged_positions_sync

logger = logging.getLogger(__name__)

STALE_MODEL_DAYS = 14
MAX_STALE_DISPLAY = 10


def get_health() -> dict:
    """Get model health status."""
    manifest = load_manifest()
    now = datetime.now()

    stale = []
    fresh = []
    for ticker, info in manifest.items():
        try:
            saved_at = datetime.fromisoformat(info["saved_at"])
            age_days = (now - saved_at).days
            if age_days > STALE_MODEL_DAYS:
                stale.append({"ticker": ticker, "age_days": age_days})
            else:
                fresh.append(ticker)
        except (KeyError, ValueError):
            stale.append({"ticker": ticker, "age_days": 999})

    state = get_merged_positions_sync()
    positions = state.get("positions", [])
    total_exposure = sum(p.get("size", 0) for p in positions) if positions else 0

    result = {
        "total_models": len(manifest),
        "fresh_models": len(fresh),
        "stale_models": len(stale),
        "open_positions": len(positions),
        "total_exposure_pct": total_exposure * 100,
    }

    if stale:
        stale_sorted = sorted(stale, key=lambda x: -x["age_days"])[:MAX_STALE_DISPLAY]
        result["stalest_models"] = stale_sorted

    return result


def run_scan(config: str = "conservative") -> dict:
    """Run the market scanner and return results."""
    try:
        logger.info("Running market scan with config: %s", config)
        results = scan_all(tickers=None, config_name=config)

        if not results:
            return {
                "success": False,
                "error": "No models found. Run retrain first.",
            }

        # Persist all scan results for later queries
        state = get_merged_positions_sync()
        positions = state.get("positions", [])
        open_tickers = {p["ticker"] for p in positions}
        save_watchlist(
            [
                {
                    "ticker": r["ticker"],
                    "p_rally": round(r.get("p_rally", 0) * 100, 1),
                    "p_rally_raw": r.get("p_rally_raw", 0),
                    "comp_score": r.get("comp_score", 0),
                    "fail_dn": r.get("fail_dn", 0),
                    "trend": r.get("trend", 0),
                    "golden_cross": r.get("golden_cross", 0),
                    "hmm_compressed": r.get("hmm_compressed", 0),
                    "rv_pctile": r.get("rv_pctile", 0),
                    "atr_pct": r.get("atr_pct", 0),
                    "macd_hist": r.get("macd_hist", 0),
                    "vol_ratio": r.get("vol_ratio", 1),
                    "vix_pctile": r.get("vix_pctile", 0),
                    "rsi": r.get("rsi", 0),
                    "close": r.get("close", 0),
                    "size": r.get("size", 0),
                    "signal": bool(r.get("signal")),
                }
                for r in sorted(results, key=lambda x: x.get("p_rally", 0), reverse=True)
                if r.get("status") == "ok" and r["ticker"] not in open_tickers
            ],
            scan_date=date.today(),
        )

        new_signals = [r for r in results if r.get("signal") and r["ticker"] not in open_tickers]

        current_state = get_merged_positions_sync()
        current_positions = current_state.get("positions", [])
        closed_today = current_state.get("closed_today", [])

        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "success": True,
            "scan_time": scan_time,
            "config": config,
            "tickers_scanned": len(results),
            "new_signals": len(new_signals),
            "existing_open_before_scan": len(positions),
            "total_open_now": len(current_positions),
            "open_positions": [p["ticker"] for p in current_positions],
            "closed_today": len(closed_today),
            "signals": [
                {
                    "ticker": r["ticker"],
                    "p_rally": round(r.get("p_rally", 0) * 100, 1),
                    "comp_score": round(r.get("comp_score", 0), 3),
                    "close": r.get("close", 0),
                    "size_pct": round(r.get("size", 0) * 100, 1),
                    "stop_price": round(r.get("range_low", 0), 2),
                }
                for r in new_signals
            ],
        }
    except Exception as e:
        logger.exception("Scan failed")
        return {
            "success": False,
            "error": str(e),
        }


def run_retrain_marker(tickers: list[str] | None = None) -> dict:
    """Return async-task marker for retrain. No side effects."""
    return {
        "_async_task": "retrain",
        "tickers": tickers,
        "message": "Starting model retraining... This will take 10-30+ minutes. You'll receive progress updates.",  # noqa: E501
    }
