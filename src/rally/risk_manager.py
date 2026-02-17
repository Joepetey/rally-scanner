"""Proactive risk management — automatic stop tightening and position trimming.

Three tiers of escalating intervention based on drawdown, regime state, and VIX:
  Tier 1 (5-10% DD):  Tighten all trailing stops from 1.5×ATR to 1.0×ATR
  Tier 2 (10-15% DD): Close weakest position + Tier 1
  Tier 3 (per-position): If P(expanding) > 0.8, tighten that position's stop
  VIX spike: If VIX changed >20% vs previous day, tighten all stops by 30%
"""

import logging
from dataclasses import dataclass

from .config import PARAMS

logger = logging.getLogger(__name__)


@dataclass
class RiskAction:
    action_type: str     # "tighten_stop" or "close_position"
    ticker: str
    reason: str
    new_trail_atr_mult: float | None = None  # new trailing stop as ATR multiple
    old_trail_atr_mult: float | None = None


def check_vix_spike() -> dict:
    """Check if VIX had a significant daily change.

    Returns {is_spike: bool, change_pct: float, vix_level: float}
    """
    try:
        from datetime import datetime, timedelta

        from .data import fetch_vix

        start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        vix = fetch_vix(start=start)
        if len(vix) < 2:
            return {"is_spike": False, "change_pct": 0.0, "vix_level": 0.0}

        current = float(vix.iloc[-1])
        prev = float(vix.iloc[-2])
        if prev <= 0:
            return {"is_spike": False, "change_pct": 0.0, "vix_level": current}

        change_pct = (current - prev) / prev
        return {
            "is_spike": change_pct >= PARAMS.risk_vix_spike_pct,
            "change_pct": round(change_pct, 4),
            "vix_level": round(current, 2),
        }
    except Exception as e:
        logger.warning("VIX spike check failed: %s", e)
        return {"is_spike": False, "change_pct": 0.0, "vix_level": 0.0}


def evaluate(
    equity: float,
    positions: list[dict],
    regime_states: dict | None = None,
    drawdown: float | None = None,
) -> list[RiskAction]:
    """Evaluate portfolio risk and return list of actions to take.

    Args:
        equity: Current account equity
        positions: List of open position dicts from positions.json
        regime_states: {ticker: {p_expanding, ...}} from regime monitor
        drawdown: Current drawdown fraction (0.0-1.0). If None, computed.

    Returns:
        List of RiskAction to execute
    """
    if not PARAMS.proactive_risk_enabled or not positions:
        return []

    # Compute drawdown if not provided
    if drawdown is None:
        from .portfolio import compute_drawdown
        drawdown = compute_drawdown(equity)

    actions: list[RiskAction] = []
    regime_states = regime_states or {}

    # --- Tier 1: Tighten all stops at 5-10% drawdown ---
    if drawdown >= PARAMS.risk_tier1_dd:
        for pos in positions:
            actions.append(RiskAction(
                action_type="tighten_stop",
                ticker=pos["ticker"],
                reason=f"Tier 1: drawdown {drawdown:.1%}",
                new_trail_atr_mult=1.0,
                old_trail_atr_mult=1.5,
            ))

    # --- Tier 2: Close weakest position at 10-15% drawdown ---
    if drawdown >= PARAMS.risk_tier2_dd and len(positions) > 0:
        weakest = min(positions, key=lambda p: p.get("unrealized_pnl_pct", 0))
        actions.append(RiskAction(
            action_type="close_position",
            ticker=weakest["ticker"],
            reason=f"Tier 2: drawdown {drawdown:.1%}, weakest position",
        ))

    # --- Tier 3: Per-position regime tightening ---
    for pos in positions:
        ticker = pos["ticker"]
        regime = regime_states.get(ticker, {})
        p_expanding = regime.get("p_expanding", 0)
        if p_expanding >= PARAMS.risk_expanding_threshold:
            # Don't duplicate if already tightened by Tier 1
            already_tightened = any(
                a.ticker == ticker and a.action_type == "tighten_stop"
                for a in actions
            )
            if not already_tightened:
                actions.append(RiskAction(
                    action_type="tighten_stop",
                    ticker=ticker,
                    reason=f"Regime: P(expanding)={p_expanding:.0%}",
                    new_trail_atr_mult=0.5,
                    old_trail_atr_mult=1.5,
                ))
            else:
                # Override to even tighter stop
                for a in actions:
                    if a.ticker == ticker and a.action_type == "tighten_stop":
                        a.new_trail_atr_mult = 0.5
                        a.reason += f" + regime P(exp)={p_expanding:.0%}"
                        break

    # --- VIX spike: tighten all by 30% ---
    vix_info = check_vix_spike()
    if vix_info["is_spike"]:
        for pos in positions:
            ticker = pos["ticker"]
            existing = next(
                (a for a in actions if a.ticker == ticker and a.action_type == "tighten_stop"),
                None,
            )
            if existing:
                # Apply additional 30% tightening
                existing.new_trail_atr_mult = round(
                    existing.new_trail_atr_mult * 0.7, 2
                )
                existing.reason += f" + VIX spike {vix_info['change_pct']:.0%}"
            else:
                current_mult = 1.5
                actions.append(RiskAction(
                    action_type="tighten_stop",
                    ticker=ticker,
                    reason=f"VIX spike: {vix_info['change_pct']:.0%} change",
                    new_trail_atr_mult=round(current_mult * 0.7, 2),
                    old_trail_atr_mult=current_mult,
                ))

    # Deduplicate: keep only one close per ticker, skip tightens for closed tickers
    close_tickers = {a.ticker for a in actions if a.action_type == "close_position"}
    seen_closes: set[str] = set()
    deduped: list[RiskAction] = []
    for a in actions:
        if a.action_type == "close_position":
            if a.ticker not in seen_closes:
                seen_closes.add(a.ticker)
                deduped.append(a)
        else:
            # Don't tighten a position we're closing
            if a.ticker not in close_tickers:
                deduped.append(a)

    return deduped


async def execute_actions(
    actions: list[RiskAction],
    positions: list[dict],
) -> list[dict]:
    """Execute risk actions (close positions, tighten stops).

    Returns list of result dicts for alerting.
    """
    results: list[dict] = []

    for action in actions:
        if action.action_type == "close_position":
            result = await _execute_close(action, positions)
            results.append(result)
        elif action.action_type == "tighten_stop":
            result = await _execute_tighten(action, positions)
            results.append(result)

    return results


async def _execute_close(action: RiskAction, positions: list[dict]) -> dict:
    """Close a position for risk reduction."""
    ticker = action.ticker
    pos = next((p for p in positions if p["ticker"] == ticker), None)
    if not pos:
        return {"ticker": ticker, "action": "close", "success": False,
                "error": "Position not found", "reason": action.reason}

    price = pos.get("current_price", pos.get("entry_price", 0))

    try:
        from .alpaca_executor import is_enabled as alpaca_enabled

        if alpaca_enabled():
            from .alpaca_executor import execute_exit
            trail_oid = pos.get("trail_order_id")
            order_result = await execute_exit(ticker, trail_order_id=trail_oid)
            if order_result.fill_price:
                price = order_result.fill_price

        from .positions import async_close_position
        closed = await async_close_position(ticker, price, "risk_reduction")

        return {
            "ticker": ticker,
            "action": "close",
            "success": True,
            "price": price,
            "pnl_pct": closed.get("realized_pnl_pct", 0) if closed else 0,
            "reason": action.reason,
        }
    except Exception as e:
        logger.exception("Risk close failed for %s", ticker)
        return {"ticker": ticker, "action": "close", "success": False,
                "error": str(e), "reason": action.reason}


async def _execute_tighten(action: RiskAction, positions: list[dict]) -> dict:
    """Tighten a position's trailing stop."""
    ticker = action.ticker
    pos = next((p for p in positions if p["ticker"] == ticker), None)
    if not pos:
        return {"ticker": ticker, "action": "tighten", "success": False,
                "error": "Position not found", "reason": action.reason}

    atr = pos.get("atr", 0)
    highest = pos.get("highest_close", pos.get("current_price", pos.get("entry_price", 0)))
    new_trail = round(highest - action.new_trail_atr_mult * atr, 2) if atr else 0
    old_trail = pos.get("trailing_stop", 0)

    # Only tighten (never loosen)
    if new_trail <= old_trail:
        return {"ticker": ticker, "action": "tighten", "success": True,
                "skipped": True, "reason": action.reason,
                "old_stop": old_trail, "new_stop": new_trail}

    try:
        from .positions import tighten_trailing_stop
        updated = tighten_trailing_stop(ticker, new_trail)
        if not updated:
            return {"ticker": ticker, "action": "tighten", "success": False,
                    "error": "Position not found in state", "reason": action.reason}

        # Update broker trailing stop if Alpaca is enabled
        from .alpaca_executor import is_enabled as alpaca_enabled
        if alpaca_enabled():
            trail_oid = pos.get("trail_order_id")
            if trail_oid:
                from .alpaca_executor import cancel_order
                await cancel_order(trail_oid)

            qty = pos.get("qty")
            if qty:
                entry = pos.get("entry_price", 1)
                atr_pct = atr / entry if entry else 0.02
                trail_pct = round(max(action.new_trail_atr_mult * atr_pct * 100, 0.5), 2)
                from .alpaca_executor import place_trailing_stop
                new_oid = await place_trailing_stop(ticker, qty, trail_pct)
                if new_oid:
                    from .positions import async_save_positions, load_positions
                    state = load_positions()
                    for p in state.get("positions", []):
                        if p["ticker"] == ticker:
                            p["trail_order_id"] = new_oid
                            break
                    await async_save_positions(state)

        return {"ticker": ticker, "action": "tighten", "success": True,
                "old_stop": old_trail, "new_stop": new_trail,
                "reason": action.reason}
    except Exception as e:
        logger.exception("Risk tighten failed for %s", ticker)
        return {"ticker": ticker, "action": "tighten", "success": False,
                "error": str(e), "reason": action.reason}
