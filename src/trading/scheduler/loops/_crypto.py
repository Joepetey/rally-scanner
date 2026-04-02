"""Shared helper: crypto position check used by multiple loops."""

from db.trading.positions import load_positions


def has_open_crypto_positions() -> bool:
    """True if any open position is a crypto asset."""
    from rally_ml.config import ASSETS

    positions = load_positions().get("positions", [])
    return any(
        ASSETS.get(p["ticker"]) and ASSETS[p["ticker"]].asset_class == "crypto"
        for p in positions
    )
