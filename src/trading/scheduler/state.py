"""Consolidated state for TradingScheduler.

All mutable flags, dedup trackers, pending results, and concurrency guards
live here instead of as bare attributes on the scheduler class.
"""

import asyncio
import zoneinfo
from dataclasses import dataclass, field
from datetime import datetime

_ET = zoneinfo.ZoneInfo("America/New_York")


@dataclass
class SchedulerState:
    """All mutable scheduler state in one place."""

    # Regime / watchlist / adaptive alerts
    regime_states: dict = field(default_factory=dict)
    watchlist_tickers: list[str] = field(default_factory=list)
    current_alert_interval: float = 0.0
    last_alert_check: datetime = field(
        default_factory=lambda: datetime.min.replace(tzinfo=_ET),
    )

    # Daily dedup: store ISO date string of last run
    ran_morning_scan: str = ""
    ran_morning_execute: str = ""
    ran_midday_1: str = ""
    ran_midday_2: str = ""
    ran_retrain: str = ""
    # Weekend crypto scan dedup: "{date}_{6h_slot}"
    ran_weekend_scan: str = ""

    # Pre-market scan results, consumed by market-open execution
    pending_signals: list[dict] = field(default_factory=list)
    pending_exits: list[dict] = field(default_factory=list)
    pending_scan_results: list[dict] = field(default_factory=list)
    pending_positions_embed: list[dict] = field(default_factory=list)

    # In-progress scan guard: prevent concurrent scans of the same type
    scan_in_progress: dict[str, bool] = field(default_factory=dict)

    # Concurrent exit guard: prevent double-exit for the same ticker
    exiting_tickers: set[str] = field(default_factory=set)
    exit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Snapshot fetch guard: prevent concurrent get_snapshots calls
    snapshot_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Housekeeping cycle counter for IEX coverage fallback
    housekeeping_cycles: int = 0

    # Stream health monitoring
    stream_degraded_cycles: int = 0
    stream_alert_sent: bool = False
