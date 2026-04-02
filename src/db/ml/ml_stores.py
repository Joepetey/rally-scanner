"""Concrete implementations of rally_ml storage protocols backed by PostgreSQL."""

from db.ml.models import load_manifest as _db_load_manifest
from db.ml.models import save_manifest_entry as _db_save_manifest
from db.ml.universe import load_universe_cache as _db_load_universe
from db.ml.universe import save_universe_cache as _db_save_universe


class PostgresManifestStore:
    """ManifestStore backed by the model_manifest PostgreSQL table."""

    def save_entry(self, ticker: str, meta: dict) -> None:
        _db_save_manifest(ticker, meta)

    def load_all(self) -> dict:
        return _db_load_manifest()


class PostgresUniverseCacheStore:
    """UniverseCacheStore backed by the universe_cache PostgreSQL table."""

    def load(self, max_age_days: int) -> dict | None:
        return _db_load_universe(max_age_days)

    def save(self, tickers: list[str], source: str) -> None:
        _db_save_universe(tickers, source)
