"""Stream supervisor with exponential backoff reconnection."""

import logging
import os
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Reconnection constants
IDLE_WAIT_SECONDS = 5.0
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
# When Alpaca rejects with "connection limit exceeded" / HTTP 429,
# retrying quickly just burns through the limit.  Wait much longer.
RATE_LIMIT_BACKOFF_SECONDS = 300.0
_RATE_LIMIT_ERRORS = ("connection limit exceeded", "429")

try:
    from alpaca.data.enums import DataFeed
    _STREAM_AVAILABLE = True
except ImportError:
    _STREAM_AVAILABLE = False


def run_stream_impl(
    symbols: set[str],
    stream_attr_setter: Callable,
    connected_event: threading.Event,
    label: str,
    make_stream: Callable,
    make_symbols: Callable,
    handler: Callable,
    lock: threading.Lock,
) -> None:
    """Generic blocking stream runner. Called from supervisor loop."""
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    if not symbols:
        logger.info("%s stream: no symbols to subscribe — idle", label)
        return

    alpaca_syms = make_symbols(symbols)
    stream = make_stream(api_key, secret_key)
    stream.subscribe_trades(handler, *alpaca_syms)
    logger.info(
        "%s stream: subscribing to %d symbols: %s",
        label, len(alpaca_syms), sorted(alpaca_syms),
    )

    with lock:
        stream_attr_setter(stream)

    logger.info("%s stream: connecting", label)

    try:
        stream.run()
        logger.info("%s stream: disconnected cleanly", label)
    finally:
        with lock:
            stream_attr_setter(None)
        connected_event.clear()
        logger.info("%s stream: connection closed", label)


def run_supervisor(
    run_fn: Callable[[], None],
    has_symbols_fn: Callable[[], bool],
    connected_event: threading.Event,
    stop_event: threading.Event,
    label: str,
) -> None:
    """Restart a stream on unexpected exit with exponential backoff."""
    backoff = INITIAL_BACKOFF_SECONDS
    attempt = 0

    while not stop_event.is_set():
        attempt += 1
        try:
            run_fn()
            if stop_event.is_set():
                break
            # Stream ran successfully for a while then disconnected — reset backoff
            backoff = INITIAL_BACKOFF_SECONDS
            if not has_symbols_fn():
                logger.debug("%s stream: no symbols, idle for %.0fs", label, IDLE_WAIT_SECONDS)
                attempt -= 1
                stop_event.wait(timeout=IDLE_WAIT_SECONDS)
                continue
            logger.warning(
                "%s stream exited unexpectedly (attempt %d), restarting in %.0fs",
                label, attempt, backoff,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if any(hint in err_msg for hint in _RATE_LIMIT_ERRORS):
                logger.error(
                    "%s stream: connection limit exceeded (attempt %d) "
                    "— backing off for %.0fs to let connections drain",
                    label, attempt, RATE_LIMIT_BACKOFF_SECONDS,
                )
                connected_event.clear()
                stop_event.wait(timeout=RATE_LIMIT_BACKOFF_SECONDS)
                if stop_event.is_set():
                    break
                backoff = INITIAL_BACKOFF_SECONDS
                continue
            logger.warning(
                "%s stream error (attempt %d): %s — restarting in %.0fs",
                label, attempt, e, backoff,
            )

        connected_event.clear()
        logger.info("%s stream: reconnect attempt %d in %.0fs", label, attempt + 1, backoff)
        stop_event.wait(timeout=backoff)
        if stop_event.is_set():
            break
        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
        logger.info("%s stream: reconnect attempt %d starting", label, attempt + 1)


def get_feed(feed_name: str) -> "DataFeed":
    """Resolve feed name string to Alpaca DataFeed enum."""
    if feed_name.lower() == "sip":
        return DataFeed.SIP
    return DataFeed.IEX
