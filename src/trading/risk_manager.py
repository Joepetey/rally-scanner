"""Risk management — re-export facade.

Actual logic lives in focused submodules:
- risk_evaluation: drawdown, circuit breaker, VIX spike, tier-based actions
- risk_execution: close positions, tighten trailing stops
"""

from trading.risk_evaluation import (
    RiskAction,
    check_vix_spike,
    compute_drawdown,
    compute_drawdown_pure,
    evaluate,
    is_circuit_breaker_active,
)
from trading.risk_execution import (
    execute_actions,
    tighten_trailing_stop,
)

__all__ = [
    "RiskAction",
    "check_vix_spike",
    "compute_drawdown",
    "compute_drawdown_pure",
    "evaluate",
    "execute_actions",
    "is_circuit_breaker_active",
    "tighten_trailing_stop",
]
