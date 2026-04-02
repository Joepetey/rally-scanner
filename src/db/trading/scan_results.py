"""Scan results, current signals, and watchlist persistence — PostgreSQL."""

import logging
from datetime import date

from db.core.pool import get_conn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Watchlist CRUD
# ---------------------------------------------------------------------------

def save_watchlist(entries: list[dict], scan_date: date) -> None:
    """Upsert watchlist entries for a given scan date."""
    with get_conn() as conn:
        cur = conn.cursor()
        for e in entries:
            cur.execute(
                """INSERT INTO watchlist
                       (scan_date, ticker, p_rally, p_rally_raw, comp_score, fail_dn,
                        trend, golden_cross, hmm_compressed, rv_pctile, atr_pct,
                        macd_hist, vol_ratio, vix_pctile, rsi, close, size, is_signal)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (scan_date, ticker) DO UPDATE SET
                       p_rally=EXCLUDED.p_rally,
                       p_rally_raw=EXCLUDED.p_rally_raw,
                       comp_score=EXCLUDED.comp_score,
                       fail_dn=EXCLUDED.fail_dn,
                       trend=EXCLUDED.trend,
                       golden_cross=EXCLUDED.golden_cross,
                       hmm_compressed=EXCLUDED.hmm_compressed,
                       rv_pctile=EXCLUDED.rv_pctile,
                       atr_pct=EXCLUDED.atr_pct,
                       macd_hist=EXCLUDED.macd_hist,
                       vol_ratio=EXCLUDED.vol_ratio,
                       vix_pctile=EXCLUDED.vix_pctile,
                       rsi=EXCLUDED.rsi,
                       close=EXCLUDED.close,
                       size=EXCLUDED.size,
                       is_signal=EXCLUDED.is_signal""",
                (
                    scan_date,
                    e["ticker"],
                    e.get("p_rally", 0),
                    e.get("p_rally_raw", 0),
                    e.get("comp_score", 0),
                    e.get("fail_dn", 0),
                    e.get("trend", 0),
                    e.get("golden_cross", 0),
                    e.get("hmm_compressed", 0),
                    e.get("rv_pctile", 0),
                    e.get("atr_pct", 0),
                    e.get("macd_hist", 0),
                    e.get("vol_ratio", 1),
                    e.get("vix_pctile", 0),
                    e.get("rsi", 0),
                    e.get("close", 0),
                    e.get("size", 0),
                    bool(e.get("signal", False)),
                ),
            )


# ---------------------------------------------------------------------------
# Scan result snapshot tables
# ---------------------------------------------------------------------------

_SCAN_RESULT_COLS = (
    "ticker, p_rally, p_rally_raw, comp_score, fail_dn, "
    "trend, golden_cross, hmm_compressed, rv_pctile, atr_pct, "
    "macd_hist, vol_ratio, vix_pctile, rsi, close, size, is_signal, is_position"
)

_SIGNAL_COLS = (
    "ticker, p_rally, comp_score, fail_dn, close, size, "
    "atr_pct, range_low, trend, golden_cross, rsi, is_position"
)

_WATCHLIST_SNAPSHOT_COLS = (
    "ticker, p_rally, p_rally_raw, comp_score, fail_dn, "
    "close, size, trend, golden_cross, hmm_compressed, "
    "rv_pctile, atr_pct, macd_hist, vol_ratio, vix_pctile, rsi"
)


def _insert_scan_result_row(
    cur, r: dict, is_sig: bool, is_pos: bool, p_rally_pct: float,
) -> None:
    cur.execute(
        f"INSERT INTO latest_scan_results ({_SCAN_RESULT_COLS}) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (
            r["ticker"], p_rally_pct, r.get("p_rally_raw", 0),
            r.get("comp_score", 0), r.get("fail_dn", 0),
            r.get("trend", 0), r.get("golden_cross", 0),
            r.get("hmm_compressed", 0), r.get("rv_pctile", 0),
            r.get("atr_pct", 0), r.get("macd_hist", 0),
            r.get("vol_ratio", 1), r.get("vix_pctile", 0),
            r.get("rsi", 0), r.get("close", 0), r.get("size", 0),
            is_sig, is_pos,
        ),
    )


def _insert_current_signal_row(cur, r: dict, is_pos: bool, p_rally_pct: float) -> None:
    cur.execute(
        f"INSERT INTO current_signals ({_SIGNAL_COLS}) "  # noqa: E501
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (
            r["ticker"], p_rally_pct, r.get("comp_score", 0),
            r.get("fail_dn", 0), r.get("close", 0), r.get("size", 0),
            r.get("atr_pct", 0), r.get("range_low", 0),
            r.get("trend", 0), r.get("golden_cross", 0),
            r.get("rsi", 0), is_pos,
        ),
    )


def _insert_watchlist_snapshot_row(cur, r: dict, p_rally_pct: float) -> None:
    cur.execute(
        f"INSERT INTO current_watchlist ({_WATCHLIST_SNAPSHOT_COLS}) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (
            r["ticker"], p_rally_pct, r.get("p_rally_raw", 0),
            r.get("comp_score", 0), r.get("fail_dn", 0),
            r.get("close", 0), r.get("size", 0), r.get("trend", 0),
            r.get("golden_cross", 0), r.get("hmm_compressed", 0),
            r.get("rv_pctile", 0), r.get("atr_pct", 0),
            r.get("macd_hist", 0), r.get("vol_ratio", 1),
            r.get("vix_pctile", 0), r.get("rsi", 0),
        ),
    )


def save_latest_scan(results: list[dict], positions: dict) -> None:
    """Replace current-state scan tables atomically on each scan."""
    open_tickers = {p["ticker"] for p in positions.get("positions", [])}
    ok = [r for r in results if r.get("status") == "ok"]

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE latest_scan_results, current_signals, current_watchlist")

        for r in ok:
            is_sig = bool(r.get("signal"))
            is_pos = r["ticker"] in open_tickers
            p_rally_pct = round(r.get("p_rally", 0) * 100, 1)

            _insert_scan_result_row(cur, r, is_sig, is_pos, p_rally_pct)
            if is_sig:
                _insert_current_signal_row(cur, r, is_pos, p_rally_pct)
            if not is_sig and not is_pos:
                _insert_watchlist_snapshot_row(cur, r, p_rally_pct)


def load_current_signals() -> list[dict]:
    """Load signals from the last pre-market scan (restart fallback).

    Returns dicts compatible with execute_entries: ticker, close, size, atr_pct,
    range_low, p_rally, date.
    """
    from datetime import date as _date

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT ticker, p_rally, comp_score, close, size, atr_pct, range_low
               FROM current_signals
               WHERE NOT is_position
               ORDER BY p_rally DESC"""
        )
        rows = cur.fetchall()

    today = _date.today().isoformat()
    return [
        {
            "ticker": r["ticker"],
            "p_rally": r["p_rally"] / 100,  # stored as pct, executor expects 0-1
            "comp_score": r["comp_score"],
            "close": r["close"],
            "size": r["size"],
            "atr_pct": r["atr_pct"],
            "range_low": r["range_low"],
            "date": today,
            "signal": True,
        }
        for r in rows
    ]


def load_watchlist() -> dict:
    """Return the most recent watchlist in the same shape as the old JSON file.

    Returns: {"updated": ISO str, "count": int, "tickers": [...]}
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT w.*
               FROM watchlist w
               WHERE scan_date = (SELECT MAX(scan_date) FROM watchlist)
               ORDER BY p_rally DESC"""
        )
        rows = cur.fetchall()

    if not rows:
        return {"message": "No watchlist yet — run a scan first."}

    tickers = []
    for r in rows:
        tickers.append({
            "ticker": r["ticker"],
            "p_rally": float(r["p_rally"]),
            "comp_score": float(r["comp_score"]),
            "close": float(r["close"]),
            "signal": bool(r["is_signal"]),
        })

    updated = rows[0]["created_at"]
    if hasattr(updated, "isoformat"):
        updated = updated.isoformat()

    return {
        "updated": updated,
        "count": len(tickers),
        "tickers": tickers,
    }
