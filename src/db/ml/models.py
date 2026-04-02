"""Model metadata persistence — manifest and regime states."""

from datetime import UTC, datetime

from db.core.pool import get_conn, row_to_dict

# ---------------------------------------------------------------------------
# model_manifest CRUD
# ---------------------------------------------------------------------------

def save_manifest_entry(ticker: str, meta: dict) -> None:
    """UPSERT a single ticker's model metadata into model_manifest."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO model_manifest
                   (ticker, saved_at, train_start, train_end, r_up, d_dn, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, NOW())
               ON CONFLICT (ticker) DO UPDATE SET
                   saved_at=EXCLUDED.saved_at,
                   train_start=EXCLUDED.train_start,
                   train_end=EXCLUDED.train_end,
                   r_up=EXCLUDED.r_up,
                   d_dn=EXCLUDED.d_dn,
                   updated_at=NOW()""",
            (
                ticker,
                meta["saved_at"],
                meta["train_start"],
                meta["train_end"],
                meta["r_up"],
                meta["d_dn"],
            ),
        )


def load_manifest() -> dict:
    """Return all model manifest entries as {ticker: {saved_at, train_start, train_end, r_up, d_dn}}."""  # noqa: E501
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM model_manifest")
        rows = cur.fetchall()
    result = {}
    for r in rows:
        d = row_to_dict(r)
        ticker = d.pop("ticker")
        d.pop("id", None)
        d.pop("updated_at", None)
        result[ticker] = d
    return result


# ---------------------------------------------------------------------------
# regime_states CRUD
# ---------------------------------------------------------------------------

def save_regime_states(states: dict) -> None:
    """UPSERT regime state for each ticker.

    states: {ticker: {p_compressed, p_normal, p_expanding, dominant_regime, timestamp}}
    """
    with get_conn() as conn:
        cur = conn.cursor()
        for ticker, s in states.items():
            cur.execute(
                """INSERT INTO regime_states
                       (ticker, p_compressed, p_normal, p_expanding, dominant_regime,
                        recorded_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, NOW())
                   ON CONFLICT (ticker) DO UPDATE SET
                       p_compressed=EXCLUDED.p_compressed,
                       p_normal=EXCLUDED.p_normal,
                       p_expanding=EXCLUDED.p_expanding,
                       dominant_regime=EXCLUDED.dominant_regime,
                       recorded_at=EXCLUDED.recorded_at,
                       updated_at=NOW()""",
                (
                    ticker,
                    s.get("p_compressed", 0),
                    s.get("p_normal", 0),
                    s.get("p_expanding", 0),
                    s.get("dominant_regime", ""),
                    s.get("timestamp", datetime.now(UTC).isoformat()),
                ),
            )


def load_regime_states() -> dict:
    """Return all regime states as {ticker: {p_compressed, p_normal, p_expanding, dominant_regime, timestamp}}."""  # noqa: E501
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM regime_states")
        rows = cur.fetchall()
    result = {}
    for r in rows:
        d = row_to_dict(r)
        ticker = d.pop("ticker")
        d.pop("updated_at", None)
        # Normalise key: regime_monitor stores "timestamp", not "recorded_at"
        d["timestamp"] = d.pop("recorded_at", "")
        result[ticker] = d
    return result
