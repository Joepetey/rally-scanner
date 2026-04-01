"""System position persistence — open/closed position CRUD against PostgreSQL."""

import logging
from datetime import date, datetime

from db.pool import get_conn, row_to_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# system_positions CRUD
# ---------------------------------------------------------------------------

_UPSERT_POSITION_SQL = """INSERT INTO system_positions
       (ticker, entry_price, entry_date, stop_price, target_price,
        trailing_stop, highest_close, atr, bars_held, size, qty,
        order_id, trail_order_id, target_order_id, p_rally,
        current_price, unrealized_pnl_pct, updated_at)
   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
   ON CONFLICT (ticker) DO UPDATE SET
       entry_price=EXCLUDED.entry_price, entry_date=EXCLUDED.entry_date,
       stop_price=EXCLUDED.stop_price, target_price=EXCLUDED.target_price,
       trailing_stop=EXCLUDED.trailing_stop,
       highest_close=EXCLUDED.highest_close,
       atr=EXCLUDED.atr, bars_held=EXCLUDED.bars_held, size=EXCLUDED.size,
       qty=EXCLUDED.qty, order_id=EXCLUDED.order_id,
       trail_order_id=EXCLUDED.trail_order_id,
       target_order_id=EXCLUDED.target_order_id,
       p_rally=EXCLUDED.p_rally,
       current_price=EXCLUDED.current_price,
       unrealized_pnl_pct=EXCLUDED.unrealized_pnl_pct,
       updated_at=NOW()"""


def _position_params(pos: dict) -> tuple:
    """Extract ordered parameters for the position UPSERT."""
    return (
        pos["ticker"], pos.get("entry_price", 0), pos.get("entry_date") or None,
        pos.get("stop_price", 0), pos.get("target_price", 0),
        pos.get("trailing_stop", 0),
        pos.get("highest_close", pos.get("entry_price", 0)),
        pos.get("atr", 0), pos.get("bars_held", 0), pos.get("size", 0),
        pos.get("qty", 0), pos.get("order_id"), pos.get("trail_order_id"),
        pos.get("target_order_id"), pos.get("p_rally", 0),
        pos.get("current_price", pos.get("entry_price", 0)),
        pos.get("unrealized_pnl_pct", 0),
    )


def save_position_meta(pos: dict) -> None:
    """UPSERT a single position row into system_positions."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_UPSERT_POSITION_SQL, _position_params(pos))


def load_position_meta(ticker: str) -> dict | None:
    """Load a single position's metadata."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM system_positions WHERE ticker = %s", (ticker,)
        )
        row = cur.fetchone()
    if row is None:
        return None
    d = row_to_dict(row)
    d["status"] = "open"
    return d


def load_all_position_meta() -> list[dict]:
    """Load all open position metadata."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM system_positions")
        rows = cur.fetchall()
    result = []
    for r in rows:
        d = row_to_dict(r)
        d["status"] = "open"
        result.append(d)
    return result


def delete_position_meta(ticker: str) -> None:
    """Remove a position from the DB."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM system_positions WHERE ticker = %s", (ticker,))


# ---------------------------------------------------------------------------
# closed_positions CRUD
# ---------------------------------------------------------------------------

def record_closed_position(pos: dict) -> None:
    """Record a closed position in the closed_positions table."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO closed_positions
                   (ticker, entry_price, entry_date, exit_price, exit_date,
                    exit_reason, realized_pnl_pct, bars_held, size)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                pos["ticker"], pos.get("entry_price", 0), pos.get("entry_date") or None,
                pos.get("exit_price", 0), pos.get("exit_date") or None,
                pos.get("exit_reason", ""), pos.get("realized_pnl_pct", 0),
                pos.get("bars_held", 0), pos.get("size", 0),
            ),
        )


def get_closed_today() -> list[dict]:
    """Get positions closed today."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM closed_positions WHERE exit_date = CURRENT_DATE"
        )
        return [row_to_dict(r) for r in cur.fetchall()]


def get_recently_closed_tickers(days: int) -> set[str]:
    """Return tickers with exits in the last N calendar days."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT ticker FROM closed_positions WHERE exit_date >= CURRENT_DATE - %s",
            (days,),
        )
        return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Bulk state operations
# ---------------------------------------------------------------------------

def load_positions() -> dict:
    """Load positions from DB. Returns {"positions": [...], "closed_today": [...]}."""
    return {
        "positions": load_all_position_meta(),
        "closed_today": get_closed_today(),
        "last_updated": datetime.now().isoformat(),
    }


def save_positions(state: dict) -> None:
    """Sync DB with the given state dict. Atomic: DELETE all open + re-insert."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM system_positions")
        for pos in state.get("positions", []):
            if not pos.get("ticker"):
                continue
            cur.execute(_UPSERT_POSITION_SQL, _position_params(pos))
        # Record any closed positions from today not yet in DB
        for closed in state.get("closed_today", []):
            if closed.get("exit_date") != today:
                continue
            cur.execute(
                "SELECT 1 FROM closed_positions WHERE ticker = %s AND exit_date = %s",
                (closed["ticker"], today),
            )
            if cur.fetchone() is None:
                cur.execute(
                    """INSERT INTO closed_positions
                           (ticker, entry_price, entry_date, exit_price, exit_date,
                            exit_reason, realized_pnl_pct, bars_held, size)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        closed["ticker"], closed.get("entry_price", 0),
                        closed.get("entry_date") or None, closed.get("exit_price", 0),
                        closed.get("exit_date") or None, closed.get("exit_reason", ""),
                        closed.get("realized_pnl_pct", 0), closed.get("bars_held", 0),
                        closed.get("size", 0),
                    ),
                )


# ---------------------------------------------------------------------------
# Helper operations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Signal queue CRUD
# ---------------------------------------------------------------------------

def enqueue_signal(sig: dict, reason: str) -> None:
    """Add or replace a signal in the queue (UPSERT by ticker)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO signal_queue
                   (ticker, p_rally, comp_score, close, size, range_low, atr_pct,
                    signal_date, skip_reason, queued_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
               ON CONFLICT (ticker) DO UPDATE SET
                   p_rally=EXCLUDED.p_rally, comp_score=EXCLUDED.comp_score,
                   close=EXCLUDED.close, size=EXCLUDED.size,
                   range_low=EXCLUDED.range_low, atr_pct=EXCLUDED.atr_pct,
                   signal_date=EXCLUDED.signal_date, skip_reason=EXCLUDED.skip_reason,
                   queued_at=NOW()""",
            (
                sig["ticker"], sig.get("p_rally", 0), sig.get("comp_score", 0),
                sig["close"], sig["size"], sig.get("range_low", 0), sig.get("atr_pct", 0),
                sig.get("date", datetime.now().strftime("%Y-%m-%d")), reason,
            ),
        )


def dequeue_signals(max_age_days: int) -> list[dict]:
    """Return valid queued signals ordered by p_rally desc (best first)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM signal_queue
               WHERE signal_date >= CURRENT_DATE - (%s || ' days')::interval
               ORDER BY p_rally DESC""",
            (str(max_age_days),),
        )
        return [row_to_dict(r) for r in cur.fetchall()]


def remove_from_queue(ticker: str) -> None:
    """Remove a ticker from the signal queue."""
    with get_conn() as conn:
        conn.cursor().execute("DELETE FROM signal_queue WHERE ticker = %s", (ticker,))


def clear_expired_queue(max_age_days: int) -> int:
    """Delete expired queue entries. Returns number removed."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """DELETE FROM signal_queue
               WHERE signal_date < CURRENT_DATE - (%s || ' days')::interval""",
            (str(max_age_days),),
        )
        return cur.rowcount


# ---------------------------------------------------------------------------
# Skipped signals CRUD (opportunity cost tracking)
# ---------------------------------------------------------------------------

def log_skipped_signal(sig: dict, reason: str) -> None:
    """Record a signal that was skipped (no-op on duplicate)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO skipped_signals
                   (ticker, signal_date, p_rally, comp_score, close, size, skip_reason)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (ticker, signal_date) DO NOTHING""",
            (
                sig["ticker"], sig.get("date", datetime.now().strftime("%Y-%m-%d")),
                sig.get("p_rally", 0), sig.get("comp_score", 0),
                sig["close"], sig.get("size", 0), reason,
            ),
        )


def get_unevaluated_skipped() -> list[dict]:
    """Return skipped signals from prior days without outcome data."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM skipped_signals
               WHERE outcome_date IS NULL AND signal_date < CURRENT_DATE
               ORDER BY signal_date DESC"""
        )
        return [row_to_dict(r) for r in cur.fetchall()]


def update_skipped_outcome(
    ticker: str, signal_date: str, outcome_price: float, outcome_pct: float,
) -> None:
    """Fill in outcome fields for a previously skipped signal."""
    with get_conn() as conn:
        conn.cursor().execute(
            """UPDATE skipped_signals
               SET outcome_date = CURRENT_DATE, outcome_price = %s, outcome_pct = %s
               WHERE ticker = %s AND signal_date = %s""",
            (outcome_price, outcome_pct, ticker, signal_date),
        )


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

            cur.execute(
                """INSERT INTO latest_scan_results
                       (ticker, p_rally, p_rally_raw, comp_score, fail_dn,
                        trend, golden_cross, hmm_compressed, rv_pctile, atr_pct,
                        macd_hist, vol_ratio, vix_pctile, rsi, close, size,
                        is_signal, is_position)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
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

            if is_sig:
                cur.execute(
                    """INSERT INTO current_signals
                           (ticker, p_rally, comp_score, fail_dn, close, size,
                            atr_pct, trend, golden_cross, rsi, is_position)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        r["ticker"], p_rally_pct, r.get("comp_score", 0),
                        r.get("fail_dn", 0), r.get("close", 0), r.get("size", 0),
                        r.get("atr_pct", 0), r.get("trend", 0),
                        r.get("golden_cross", 0), r.get("rsi", 0), is_pos,
                    ),
                )

            if not is_sig and not is_pos:
                cur.execute(
                    """INSERT INTO current_watchlist
                           (ticker, p_rally, p_rally_raw, comp_score, fail_dn,
                            close, size, trend, golden_cross, hmm_compressed,
                            rv_pctile, atr_pct, macd_hist, vol_ratio, vix_pctile, rsi)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
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


def tighten_trailing_stop(ticker: str, new_stop: float) -> dict | None:
    """Tighten a position's trailing stop (only if new_stop > current).

    Returns the updated position dict, or None if not found or not tightened.
    """
    pos = load_position_meta(ticker)
    if pos is None:
        return None
    current = pos.get("trailing_stop", 0)
    if new_stop > current:
        pos["trailing_stop"] = round(new_stop, 2)
        save_position_meta(pos)
        logger.info(
            "Tightened trailing stop for %s: %.2f → %.2f",
            ticker, current, new_stop,
        )
        return pos
    return None
