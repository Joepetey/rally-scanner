"""Optional Sentry error monitoring — no-ops when SENTRY_DSN is unset."""

import logging
import os

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration


def init_sentry() -> None:
    """Initialize Sentry SDK if SENTRY_DSN is set, otherwise silently skip."""
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        return

    def _before_send(event: dict, hint: dict) -> dict | None:
        if "exc_info" in hint:
            exc_value = str(hint["exc_info"][1])
            if "42210000" in exc_value:  # Alpaca "not tradable"
                return None
        return event

    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("RAILWAY_ENVIRONMENT", "development"),
        release=os.environ.get("RAILWAY_GIT_COMMIT_SHA", "unknown"),
        integrations=[
            LoggingIntegration(level=logging.ERROR, event_level=logging.ERROR),
        ],
        before_send=_before_send,
        traces_sample_rate=0.0,
        server_name=os.environ.get("RAILWAY_SERVICE_NAME", "rally-scanner"),
    )
