"""Shared helpers for the services layer."""


def dollar_metrics(
    capital: float, size: float, entry: float, stop: float | None = None,
) -> dict:
    """Compute dollar allocation and risk from capital, size fraction, entry price, and stop."""
    metrics: dict = {}
    if capital > 0 and size > 0:
        metrics["dollar_allocation"] = capital * size
        if stop and entry > stop:
            metrics["dollar_risk"] = capital * size * (entry - stop) / entry
    return metrics
