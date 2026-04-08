"""Typed event models emitted by TradingScheduler and consumed by Discord bot.

All events are Pydantic BaseModels for validation and serialization.
"""

from typing import Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Scan / regime / retrain event types
# ---------------------------------------------------------------------------

class ScanResult(BaseModel):
    signals: list[dict]
    exits: list[dict]
    orders: list[dict]
    positions_summary: dict
    scan_type: str = "daily"  # "daily", "morning", "midday", "cascade", "post_retrain"
    equity: float = 0.0  # account equity at scan time (for order embed dollar amounts)
    error: str | None = None


class WatchlistEvent(BaseModel):
    signals: list[dict]
    scan_type: str = "midday"


class RegimeEvent(BaseModel):
    transitions: list[dict]
    cascade_triggered: bool


class RetrainResult(BaseModel):
    tickers_retrained: list[str]
    duration_seconds: float
    manifest_size: int


class RiskActionEvent(BaseModel):
    actions: list[dict]


class StreamDegradedEvent(BaseModel):
    disconnected_minutes: int  # consecutive market-hours minutes stream has been down


class StreamRecoveredEvent(BaseModel):
    downtime_minutes: int  # how long stream was down before recovery


# ---------------------------------------------------------------------------
# Typed alert / exit / fill events
# ---------------------------------------------------------------------------

class AlertEvent(BaseModel):
    ticker: str
    alert_type: Literal["stop_breached", "target_breached", "near_stop", "near_target"]
    current_price: float
    level_price: float
    level_name: str
    entry_price: float
    pnl_pct: float
    distance_pct: float = 0.0


class ExitResult(BaseModel):
    ticker: str
    exit_reason: str
    fill_price: float | None = None
    order_id: str | None = None
    realized_pnl_pct: float | None = None
    bars_held: int | None = None


class FillNotification(BaseModel):
    ticker: str
    fill_price: float
    qty: float | None = None
    stop_price: float = 0.0
    target_price: float = 0.0


class HousekeepingResult(BaseModel):
    fills_confirmed: list[FillNotification]
    orders_placed: list[dict]
    positions_synced: bool


class LetItRideEvent(BaseModel):
    ticker: str
    entry_price: float
    target_price: float
    current_price: float
    trail_pct: float
    trail_order_id: str | None = None
    pnl_pct: float


# ---------------------------------------------------------------------------
# Union of all events the scheduler can emit
# ---------------------------------------------------------------------------

TradingEvent = (
    AlertEvent | ExitResult | HousekeepingResult | LetItRideEvent |
    ScanResult | WatchlistEvent | RegimeEvent | RetrainResult | RiskActionEvent |
    StreamDegradedEvent | StreamRecoveredEvent
)
