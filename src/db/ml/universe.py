"""Universe cache persistence — single-row cache of S&P 500 + Nasdaq tickers."""

from datetime import UTC, datetime

from db.core.pool import get_conn


def save_universe_cache(tickers: list[str], source: str) -> None:
    """Upsert the universe cache (single row, id=1)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO universe_cache (id, tickers, source, fetched_at, updated_at)
               VALUES (1, %s, %s, %s, NOW())
               ON CONFLICT (id) DO UPDATE SET
                   tickers=EXCLUDED.tickers,
                   source=EXCLUDED.source,
                   fetched_at=EXCLUDED.fetched_at,
                   updated_at=NOW()""",
            (tickers, source, datetime.now(UTC)),
        )


def load_universe_cache(max_age_days: int) -> dict | None:
    """Return cached universe if fresh, else None.

    Returns: {tickers: list[str], source: str, count: int, fetched_at: str}
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM universe_cache WHERE id = 1")
        row = cur.fetchone()

    if row is None:
        return None

    fetched_at = row["fetched_at"]
    if isinstance(fetched_at, str):
        fetched_at = datetime.fromisoformat(fetched_at)

    # Strip timezone for comparison
    fetched_naive = fetched_at.replace(tzinfo=None) if fetched_at.tzinfo else fetched_at
    age_days = (datetime.now() - fetched_naive).days
    if age_days > max_age_days:
        return None

    tickers = list(row["tickers"])
    return {
        "tickers": tickers,
        "source": row["source"],
        "count": len(tickers),
        "fetched_at": fetched_at.isoformat(),
    }
