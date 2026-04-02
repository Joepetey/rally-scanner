"""Signal queue processing and portfolio constraint filtering."""

import logging

from rally_ml.config import PARAMS, TICKER_TO_GROUP

from db.trading.signal_queue import (
    clear_expired_queue,
    dequeue_signals,
    get_unevaluated_skipped,
    update_skipped_outcome,
)

logger = logging.getLogger(__name__)


def filter_signals_by_constraints(
    open_positions: list[dict],
    signals: list[dict],
    max_exposure: float,
    max_group_positions: int,
    max_group_exposure: float,
) -> list[dict]:
    """Pure filter: return signals that pass portfolio + group exposure caps."""
    open_tickers = {pos["ticker"] for pos in open_positions}
    current_exposure = sum(pos.get("size", 0) for pos in open_positions)

    group_counts: dict[str, int] = {}
    group_exposures: dict[str, float] = {}
    for pos in open_positions:
        g = TICKER_TO_GROUP.get(pos["ticker"])
        if g:
            group_counts[g] = group_counts.get(g, 0) + 1
            group_exposures[g] = group_exposures.get(g, 0) + pos.get("size", 0)

    accepted: list[dict] = []
    for sig in signals:
        if sig["ticker"] in open_tickers:
            continue
        sig_size = sig.get("size", 0)
        if current_exposure + sig_size > max_exposure:
            continue
        g = TICKER_TO_GROUP.get(sig["ticker"])
        if g:
            if group_counts.get(g, 0) >= max_group_positions:
                continue
            if group_exposures.get(g, 0) + sig_size > max_group_exposure:
                continue
        accepted.append(sig)
        current_exposure += sig_size
        open_tickers.add(sig["ticker"])
        if g:
            group_counts[g] = group_counts.get(g, 0) + 1
            group_exposures[g] = group_exposures.get(g, 0) + sig_size

    return accepted


def process_signal_queue() -> list[dict]:
    """Return valid queued signals ready to re-attempt, clearing expired entries.

    Signals are ordered by P(rally) descending. The caller is responsible for
    attempting execution and calling remove_from_queue on success.
    """
    expired = clear_expired_queue(PARAMS.signal_queue_max_age_days)
    if expired:
        logger.info("Cleared %d expired signals from queue", expired)

    return dequeue_signals(PARAMS.signal_queue_max_age_days)


def update_skipped_outcomes(results: list[dict]) -> int:
    """Fill in outcome data for previously skipped signals using today's scan prices.

    Looks up each unevaluated skipped signal, finds a matching current price in
    results, computes the return from signal_date close to today's close, and
    persists it. Returns number of outcomes recorded.
    """
    unevaluated = get_unevaluated_skipped()
    if not unevaluated:
        return 0

    price_map = {r["ticker"]: r["close"] for r in results if r.get("status") == "ok"}
    recorded = 0
    for entry in unevaluated:
        ticker = entry["ticker"]
        if ticker not in price_map:
            continue
        current_price = price_map[ticker]
        signal_close = entry.get("close", 0)
        if signal_close <= 0:
            continue
        outcome_pct = round((current_price / signal_close - 1) * 100, 2)
        update_skipped_outcome(ticker, str(entry["signal_date"]), current_price, outcome_pct)
        recorded += 1

    if recorded:
        logger.info("Updated outcomes for %d skipped signals", recorded)
    return recorded
