"""PostgreSQL connection pool for rally-scanner."""

import datetime as _dt
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

_pool: ThreadedConnectionPool | None = None


def init_pool(min_conn: int = 1, max_conn: int = 10) -> None:
    """Initialize the connection pool. Called once at startup."""
    global _pool
    _pool = ThreadedConnectionPool(min_conn, max_conn, os.environ["DATABASE_URL"])


@contextmanager
def get_conn():
    """Check out a connection from the pool, auto-commit or rollback on exit."""
    conn = _pool.getconn()
    try:
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def close_pool() -> None:
    """Close all connections in the pool."""
    if _pool:
        _pool.closeall()


def row_to_dict(row) -> dict:
    """Convert a psycopg2 RealDictRow to a plain dict with dates as ISO strings."""
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, _dt.datetime):
            d[k] = v.isoformat()
        elif isinstance(v, _dt.date):
            d[k] = v.isoformat()
    return d
