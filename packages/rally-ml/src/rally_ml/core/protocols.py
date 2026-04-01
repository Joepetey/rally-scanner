"""Protocols for external storage backends.

rally-ml does not depend on any database driver. The host application
(rally-scanner) injects concrete implementations at startup via the
``configure()`` functions in persistence.py and universe.py.
"""

from typing import Protocol


class ManifestStore(Protocol):
    """Read/write model training metadata."""

    def save_entry(self, ticker: str, meta: dict) -> None: ...

    def load_all(self) -> dict: ...


class UniverseCacheStore(Protocol):
    """Read/write cached ticker universe."""

    def load(self, max_age_days: int) -> dict | None: ...

    def save(self, tickers: list[str], source: str) -> None: ...
