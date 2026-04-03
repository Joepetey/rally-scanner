"""Alpaca real-time market data stream package."""

import os

from rally_ml.config import PARAMS

from .manager import AlpacaStreamManager

__all__ = ["AlpacaStreamManager", "is_stream_enabled"]


def is_stream_enabled() -> bool:
    """True if streaming is enabled and Alpaca keys are configured."""
    if os.environ.get("ALPACA_STREAM_ENABLED", "1") == "0":
        return False
    if not PARAMS.stream_enabled:
        return False
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))
