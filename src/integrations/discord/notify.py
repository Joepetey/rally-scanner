"""Discord notification system — re-exports embeds and transport.

All embed builders live in embeds.py; HTTP transport in transport.py.
This module preserves the public API so existing imports keep working.
"""

from integrations.discord.embeds import (  # noqa: F401
    _approaching_alert_embed,
    _error_embed,
    _exit_embed,
    _fill_confirmation_embed,
    _let_it_ride_embed,
    _order_embed,
    _order_failure_embed,
    _positions_embed,
    _price_alert_embed,
    _regime_shift_embed,
    _retrain_embed,
    _risk_action_embed,
    _signal_embed,
    _stream_degraded_embed,
    _stream_recovered_embed,
)
from integrations.discord.transport import send_discord  # noqa: F401
