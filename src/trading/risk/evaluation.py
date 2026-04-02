"""Risk evaluation — drawdown, circuit breaker, VIX spike, and tier-based actions.

Pure evaluation logic with no broker or position-mutation side effects.
Execution lives in risk_execution.py.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from rally_ml.config import PARAMS
from rally_ml.core.data import fetch_vix

from db.trading.portfolio import get_high_water_mark, set_high_water_mark

logger = logging.getLogger(__name__)


def compute_drawdown_pure(equity: float, hwm: float) -> tuple[float, float]:
    """Pure math: compute drawdown from equity and high-water mark.

    Returns (drawdown_pct, new_hwm) where drawdown_pct is a positive fraction.
    """
    if equity <= 0:
        return 0.0, hwm
    if hwm <= 0:
        return 0.0, equity
    if equity >= hwm:
        return 0.0, equity
    return (hwm - equity) / hwm, hwm


def compute_drawdown(equity: float) -> float:
    """Compute current drawdown from equity high-water mark.

    Returns drawdown as a positive fraction (e.g. 0.10 = 10% drawdown).
    """
    hwm = get_high_water_mark()
    dd, new_hwm = compute_drawdown_pure(equity, hwm)
    if new_hwm != hwm:
        set_high_water_mark(new_hwm)
    return dd


def is_circuit_breaker_active(equity: float) -> bool:
    """Check if the drawdown circuit breaker should block new entries."""
    if not PARAMS.circuit_breaker_enabled:
        return False
    dd = compute_drawdown(equity)
    if dd >= PARAMS.max_drawdown_pct:
        logger.warning(
            "Circuit breaker ACTIVE: drawdown %.1f%% >= threshold %.1f%%",
            dd * 100, PARAMS.max_drawdown_pct * 100,
        )
        return True
    return False


@dataclass
class RiskAction:
    action_type: str     # "tighten_stop" or "close_position"
    ticker: str
    reason: str
    new_trail_atr_mult: float | None = None
    old_trail_atr_mult: float | None = None


def check_vix_spike() -> dict:
    """Check if VIX had a significant daily change.

    Returns {is_spike: bool, change_pct: float, vix_level: float}
    """
    try:
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
    vix_info: dict | None = None,
) -> list[RiskAction]:
    """Evaluate portfolio risk and return list of actions to take."""
    if not PARAMS.proactive_risk_enabled or not positions:
        return []

    if drawdown is None:
        drawdown = compute_drawdown(equity)

    actions: list[RiskAction] = []
    regime_states = regime_states or {}

    # Tier 1: Tighten all stops at 5-10% drawdown
    if drawdown >= PARAMS.risk_tier1_dd:
        for pos in positions:
            actions.append(RiskAction(
                action_type="tighten_stop",
                ticker=pos["ticker"],
                reason=f"Tier 1: drawdown {drawdown:.1%}",
                new_trail_atr_mult=1.0,
                old_trail_atr_mult=PARAMS.trailing_stop_atr_mult,
            ))

    # Tier 2: Close weakest position at 10-15% drawdown
    if drawdown >= PARAMS.risk_tier2_dd:
        weakest = min(positions, key=lambda p: p.get("unrealized_pnl_pct", 0))
        actions.append(RiskAction(
            action_type="close_position",
            ticker=weakest["ticker"],
            reason=f"Tier 2: drawdown {drawdown:.1%}, weakest position",
        ))

    # Tier 3: Per-position regime tightening
    for pos in positions:
        ticker = pos["ticker"]
        regime = regime_states.get(ticker, {})
        p_expanding = regime.get("p_expanding", 0)
        if p_expanding >= PARAMS.risk_expanding_threshold:
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
                for a in actions:
                    if a.ticker == ticker and a.action_type == "tighten_stop":
                        a.new_trail_atr_mult = 0.5
                        a.reason += f" + regime P(exp)={p_expanding:.0%}"
                        break

    # VIX spike: tighten all by 30%
    if vix_info is None:
        vix_info = check_vix_spike()
    if vix_info["is_spike"]:
        for pos in positions:
            ticker = pos["ticker"]
            existing = next(
                (a for a in actions if a.ticker == ticker and a.action_type == "tighten_stop"),
                None,
            )
            if existing:
                existing.new_trail_atr_mult = round(
                    existing.new_trail_atr_mult * 0.7, 2
                )
                existing.reason += f" + VIX spike {vix_info['change_pct']:.0%}"
            else:
                base_mult = PARAMS.trailing_stop_atr_mult
                actions.append(RiskAction(
                    action_type="tighten_stop",
                    ticker=ticker,
                    reason=f"VIX spike: {vix_info['change_pct']:.0%} change",
                    new_trail_atr_mult=round(base_mult * 0.7, 2),
                    old_trail_atr_mult=base_mult,
                ))

    # Deduplicate: one close per ticker, skip tightens for closed tickers
    close_tickers = {a.ticker for a in actions if a.action_type == "close_position"}
    seen_closes: set[str] = set()
    deduped: list[RiskAction] = []
    for a in actions:
        if a.action_type == "close_position":
            if a.ticker not in seen_closes:
                seen_closes.add(a.ticker)
                deduped.append(a)
        elif a.ticker not in close_tickers:
            deduped.append(a)

    return deduped
