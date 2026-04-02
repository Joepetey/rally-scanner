"""Risk action execution — close positions and tighten trailing stops.

Separated from evaluation (risk_evaluation.py) to keep side effects isolated.
"""

import logging

from db.trading.positions import load_position_meta, save_position_meta

from integrations.alpaca.broker import is_enabled as alpaca_enabled
from integrations.alpaca.exits import execute_exit
from trading.positions import async_close_position
from trading.risk.evaluation import RiskAction

logger = logging.getLogger(__name__)


def tighten_trailing_stop(ticker: str, new_stop: float) -> dict | None:
    """Tighten a position's trailing stop (only if new_stop > current).

    Returns the updated position dict, or None if not found or not tightened.
    """
    pos = load_position_meta(ticker)
    if pos is None:
        return None
    current = pos.get("trailing_stop", 0)
    if new_stop > current:
        pos["trailing_stop"] = round(new_stop, 2)
        save_position_meta(pos)
        logger.info(
            "Tightened trailing stop for %s: %.2f -> %.2f",
            ticker, current, new_stop,
        )
        return pos
    return None


async def _default_close_fn(ticker: str, pos: dict) -> float:
    """Default close implementation using Alpaca + DB."""
    price = pos.get("current_price", pos.get("entry_price", 0))
    if alpaca_enabled():
        trail_oid = pos.get("trail_order_id")
        order_result = await execute_exit(ticker, trail_order_id=trail_oid)
        if order_result.fill_price:
            price = order_result.fill_price

    closed = await async_close_position(ticker, price, "risk_reduction")
    return closed.get("realized_pnl_pct", 0) if closed else 0


async def _default_tighten_fn(ticker: str, new_trail: float) -> bool:
    """Default tighten implementation using DB."""
    return tighten_trailing_stop(ticker, new_trail)


async def execute_actions(
    actions: list[RiskAction],
    positions: list[dict],
    close_fn=None,
    tighten_fn=None,
) -> list[dict]:
    """Execute risk actions (close positions, tighten stops).

    Args:
        close_fn: async (ticker, pos) -> pnl_pct. Defaults to Alpaca + DB.
        tighten_fn: async (ticker, new_trail) -> bool. Defaults to DB.

    Returns list of result dicts for alerting.
    """
    if close_fn is None:
        close_fn = _default_close_fn
    if tighten_fn is None:
        tighten_fn = _default_tighten_fn

    results: list[dict] = []
    for action in actions:
        if action.action_type == "close_position":
            result = await _execute_close(action, positions, close_fn)
            results.append(result)
        elif action.action_type == "tighten_stop":
            result = await _execute_tighten(action, positions, tighten_fn)
            results.append(result)

    return results


async def _execute_close(
    action: RiskAction, positions: list[dict], close_fn,
) -> dict:
    """Close a position for risk reduction."""
    ticker = action.ticker
    pos = next((p for p in positions if p["ticker"] == ticker), None)
    if not pos:
        return {
            "ticker": ticker, "action": "close", "success": False,
            "error": "Position not found", "reason": action.reason,
        }

    try:
        pnl_pct = await close_fn(ticker, pos)
        price = pos.get("current_price", pos.get("entry_price", 0))
        return {
            "ticker": ticker, "action": "close", "success": True,
            "price": price, "pnl_pct": pnl_pct, "reason": action.reason,
        }
    except Exception as e:
        logger.exception("Risk close failed for %s", ticker)
        return {
            "ticker": ticker, "action": "close", "success": False,
            "error": str(e), "reason": action.reason,
        }


async def _execute_tighten(
    action: RiskAction, positions: list[dict], tighten_fn,
) -> dict:
    """Tighten a position's trailing stop."""
    ticker = action.ticker
    pos = next((p for p in positions if p["ticker"] == ticker), None)
    if not pos:
        return {
            "ticker": ticker, "action": "tighten", "success": False,
            "error": "Position not found", "reason": action.reason,
        }

    atr = pos.get("atr", 0)
    highest = pos.get(
        "highest_close", pos.get("current_price", pos.get("entry_price", 0)),
    )
    new_trail = round(highest - action.new_trail_atr_mult * atr, 2) if atr else 0
    old_trail = pos.get("trailing_stop", 0)

    if new_trail <= old_trail:
        return {
            "ticker": ticker, "action": "tighten", "success": True,
            "skipped": True, "reason": action.reason,
            "old_stop": old_trail, "new_stop": new_trail,
        }

    try:
        updated = await tighten_fn(ticker, new_trail)
        if not updated:
            return {
                "ticker": ticker, "action": "tighten", "success": False,
                "error": "Position not found in state", "reason": action.reason,
            }
        return {
            "ticker": ticker, "action": "tighten", "success": True,
            "old_stop": old_trail, "new_stop": new_trail,
            "reason": action.reason,
        }
    except Exception as e:
        logger.exception("Risk tighten failed for %s", ticker)
        return {
            "ticker": ticker, "action": "tighten", "success": False,
            "error": str(e), "reason": action.reason,
        }
