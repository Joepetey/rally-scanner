"""Stream callbacks and helpers extracted from TradingScheduler."""

from __future__ import annotations

import asyncio
import logging
from datetime import date as _date
from typing import TYPE_CHECKING

from config import ASSETS, PARAMS
from db.positions import (
    load_position_meta as _load_position_meta,
)
from db.positions import (
    load_positions,
)
from db.positions import (
    save_position_meta as _save_position_meta,
)
from db.positions import (
    save_watchlist as _db_save_watchlist,
)
from integrations.alpaca.executor import is_enabled as alpaca_enabled
from trading.positions import async_save_positions, update_position_for_price
from trading.risk_manager import compute_drawdown

if TYPE_CHECKING:
    from trading.engine import AlertEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone helpers (no scheduler state needed)
# ---------------------------------------------------------------------------


async def store_order_ids(results: list) -> None:
    """Persist Alpaca order IDs to position records after entry fills."""
    state = load_positions()
    pos_by_ticker = {pos["ticker"]: pos for pos in state["positions"]}
    for result in results:
        if result.success and result.order_id:
            pos = pos_by_ticker.get(result.ticker)
            if pos:
                if not pos.get("order_id"):
                    pos["order_id"] = result.order_id
                if result.qty:
                    pos["qty"] = result.qty
                if result.trail_order_id:
                    pos["trail_order_id"] = result.trail_order_id
    await async_save_positions(state)


def has_open_crypto_positions() -> bool:
    """True if any open position is a crypto asset."""
    positions = load_positions().get("positions", [])
    return any(
        ASSETS.get(p["ticker"]) and ASSETS[p["ticker"]].asset_class == "crypto"
        for p in positions
    )


def save_watchlist(results: list, positions: dict) -> None:
    """Persist scan results as watchlist for agent queries."""
    open_tickers = {p["ticker"] for p in positions.get("positions", [])}
    watchlist = []
    for r in sorted(results, key=lambda x: x.get("p_rally", 0), reverse=True):
        if r.get("status") != "ok":
            continue
        if r["ticker"] in open_tickers:
            continue
        watchlist.append({
            "ticker": r["ticker"],
            "p_rally": round(r.get("p_rally", 0) * 100, 1),
            "p_rally_raw": r.get("p_rally_raw", 0),
            "comp_score": r.get("comp_score", 0),
            "fail_dn": r.get("fail_dn", 0),
            "trend": r.get("trend", 0),
            "golden_cross": r.get("golden_cross", 0),
            "hmm_compressed": r.get("hmm_compressed", 0),
            "rv_pctile": r.get("rv_pctile", 0),
            "atr_pct": r.get("atr_pct", 0),
            "macd_hist": r.get("macd_hist", 0),
            "vol_ratio": r.get("vol_ratio", 1),
            "vix_pctile": r.get("vix_pctile", 0),
            "rsi": r.get("rsi", 0),
            "close": r.get("close", 0),
            "size": r.get("size", 0),
            "signal": bool(r.get("signal")),
        })
    _db_save_watchlist(watchlist, scan_date=_date.today())


async def should_use_fast_alerts(
    positions: list[dict], quotes: dict[str, dict],
) -> bool:
    """True if any position is near its stop or portfolio is drawing down."""
    for pos in positions:
        ticker = pos["ticker"]
        price = quotes.get(ticker, {}).get("price", 0)
        stop = max(pos.get("stop_price", 0), pos.get("trailing_stop", 0))
        if stop > 0 and price > 0:
            distance_pct = (price / stop - 1) * 100
            if 0 < distance_pct <= PARAMS.stop_proximity_pct:
                return True
    try:
        dd = compute_drawdown(100_000)
        if dd >= PARAMS.risk_tier1_dd:
            return True
    except Exception:
        logger.warning("Drawdown computation failed in should_use_fast_alerts", exc_info=True)
    return False


# ---------------------------------------------------------------------------
# Stream callbacks (need scheduler for engine, on_event, locks, stream)
# ---------------------------------------------------------------------------


def on_stream_trade(scheduler, ticker: str, price: float) -> None:
    """Called from stream thread per throttled trade event.

    Evaluates the ticker synchronously then dispatches async work
    back to the main event loop via run_coroutine_threadsafe.
    """
    if not scheduler._loop:
        return
    is_crypto = bool(ASSETS.get(ticker) and ASSETS[ticker].asset_class == "crypto")
    if not is_crypto and not scheduler._engine.is_market_open():
        return

    pos = _load_position_meta(ticker)
    if not pos:
        return

    if update_position_for_price(pos, price):
        _save_position_meta(pos)

    event = scheduler._engine.evaluate_single_ticker(ticker, price, pos)
    if event:
        asyncio.run_coroutine_threadsafe(
            handle_stream_alert(scheduler, event, pos, price),
            scheduler._loop,
        )


async def handle_stream_alert(
    scheduler, event: AlertEvent, pos: dict, price: float,
) -> None:
    """Process a stream-triggered alert on the main event loop."""
    await scheduler._on_event(event)

    if event.alert_type in ("stop_breached", "target_breached") and alpaca_enabled():
        fresh = _load_position_meta(event.ticker)
        if fresh is not None:
            pos = fresh
        await execute_breach_guarded(scheduler, event.ticker, pos, price, event.alert_type)


async def execute_breach_guarded(
    scheduler, ticker: str, pos: dict, price: float, alert_type: str,
) -> None:
    """Execute a breach exit with concurrent-exit guard."""
    async with scheduler._exit_lock:
        if ticker in scheduler._exiting_tickers:
            logger.info("Exit already in progress for %s \u2014 skipping duplicate", ticker)
            return
        scheduler._exiting_tickers.add(ticker)

    try:
        reason = alert_type.replace("_breached", "")
        exit_result = await scheduler._engine.execute_breach(ticker, pos, price, reason)
        if exit_result:
            await scheduler._on_event(exit_result)
            await scheduler.run_risk_evaluation()
            if scheduler._stream:
                all_pos = load_positions().get("positions", [])
                scheduler._stream.update_subscriptions({p["ticker"] for p in all_pos})
    finally:
        async with scheduler._exit_lock:
            scheduler._exiting_tickers.discard(ticker)
