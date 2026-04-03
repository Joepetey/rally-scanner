"""Subscription diff and update logic for Alpaca streams."""

import logging
import threading
from collections.abc import Callable

from .symbols import to_alpaca_crypto_symbol

logger = logging.getLogger(__name__)


def apply_diff(
    label: str,
    new_symbols: set[str],
    current_symbols: set[str],
    stream: object | None,
    connected: threading.Event,
    subscribe_handler: Callable,
    symbol_mapper: Callable[[set[str]], list[str]] = lambda s: list(s),
) -> set[str] | None:
    """Compute add/remove diff and apply to a live stream.

    Returns the new symbol set if changed, None if unchanged.
    """
    add = new_symbols - current_symbols
    remove = current_symbols - new_symbols
    if not (add or remove):
        return None

    logger.info(
        "%s stream subscriptions: +%d -%d (total %d)",
        label, len(add), len(remove), len(new_symbols),
    )

    if stream and connected.is_set():
        if add:
            try:
                mapped = symbol_mapper(add)
                stream.subscribe_trades(subscribe_handler, *mapped)
                logger.info("Subscribed %s: %s", label.lower(), sorted(add))
            except Exception as e:
                logger.warning("Failed to subscribe %s %s: %s", label.lower(), add, e)
        if remove:
            try:
                mapped = symbol_mapper(remove)
                stream.unsubscribe_trades(*mapped)
                logger.info("Unsubscribed %s: %s", label.lower(), sorted(remove))
            except Exception as e:
                logger.warning("Failed to unsubscribe %s %s: %s", label.lower(), remove, e)

    return new_symbols


def crypto_symbol_mapper(symbols: set[str]) -> list[str]:
    """Map internal crypto keys to Alpaca format for subscription calls."""
    return [to_alpaca_crypto_symbol(s) for s in symbols]
