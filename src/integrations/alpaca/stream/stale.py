"""Stale ticker detection for Alpaca streams."""

import time


def detect_stale_tickers(
    symbols: set[str],
    last_trade_time: dict[str, float],
    known_stale: set[str],
    stale_seconds: float = 300.0,
) -> tuple[list[str], list[str], list[str]]:
    """Return (new_stale, already_known_stale, never_traded) for subscribed tickers.

    new_stale     — previously received trades but went silent; callers should log WARNING.
    already_known — already reported and still stale; callers should log DEBUG only.
    never_traded  — never received a single trade since subscription; expected for
                    low-volume tickers. Callers should log INFO, not WARNING.

    Pure function — does NOT mutate known_stale. Caller is responsible for updating it.
    """
    now = time.monotonic()
    new_stale: list[str] = []
    already_known: list[str] = []
    never_traded: list[str] = []

    for ticker in symbols:
        last = last_trade_time.get(ticker)
        if last is None or (now - last) > stale_seconds:
            if ticker in known_stale:
                already_known.append(ticker)
            elif last is None:
                never_traded.append(ticker)
            else:
                new_stale.append(ticker)

    return new_stale, already_known, never_traded
