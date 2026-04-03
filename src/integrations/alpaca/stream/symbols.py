"""Symbol classification and format conversion for Alpaca streams."""

from rally_ml.config import ASSETS


def is_equity(ticker: str) -> bool:
    """True if ticker is an equity (not crypto). Uses suffix heuristic."""
    return not ticker.endswith("-USD")


def is_crypto(ticker: str) -> bool:
    """True if ticker is a known crypto asset in ASSETS config."""
    cfg = ASSETS.get(ticker)
    return bool(cfg and cfg.asset_class == "crypto")


def to_alpaca_crypto_symbol(ticker: str) -> str:
    """Convert internal crypto key to Alpaca format: BTC → BTC/USD."""
    return ASSETS[ticker].ticker.replace("-", "/")
