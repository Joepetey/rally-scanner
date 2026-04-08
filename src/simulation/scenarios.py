"""Scenario configuration for BTC-USD paper trading simulation.

Each scenario sets tight stops/targets so a small, predictable price injection
forces the intended exit condition:

  target — tight take-profit at entry * 1.01; inject above to trigger
  stop   — tight stop-loss at entry * 0.99; inject below to trigger
  trail  — far stop/target; Phase 1 injects a high price so the trailing stop
            tightens, Phase 2 injects below the new trailing stop
  time   — far stop/target; bars_held is advanced past time_stop_bars so
            update_existing_positions() triggers the time-stop exit
"""

from datetime import datetime

from rally_ml.config import PARAMS

TICKER = "BTC"
_SIM_SIZE = 0.05  # 5% of equity — small enough to always fit within any cap
VALID_SCENARIOS = {"target", "stop", "trail", "time", "let_it_ride"}


def get_signal_for_scenario(scenario: str, entry_price: float) -> dict:
    """Return a synthetic BTC signal dict for execute_entries()."""
    pos = setup_position_for_scenario(scenario, entry_price)
    return {
        "ticker": TICKER,
        "close": entry_price,
        "signal": True,
        "p_rally": 0.95,
        "size": _SIM_SIZE,
        "range_low": pos["stop_price"],
        "atr_pct": PARAMS.default_atr_pct,
        "date": pos["entry_date"],
        "status": "ok",
    }


def setup_position_for_scenario(scenario: str, entry_price: float) -> dict:
    """Return an initial position meta dict with tight stops/targets for the scenario."""
    p = PARAMS
    atr_val = round(entry_price * p.default_atr_pct, 4)
    today = datetime.now().strftime("%Y-%m-%d")

    if scenario == "target":
        # Tight take-profit 1% above entry; stop is far (won't trigger naturally)
        target_price = round(entry_price * 1.01, 2)
        stop_price = round(entry_price * 0.97, 2)
    elif scenario == "stop":
        # Tight stop 1% below entry; target is far (won't trigger naturally)
        stop_price = round(entry_price * 0.99, 2)
        target_price = round(entry_price * 1.05, 2)
    elif scenario == "trail":
        # Far hard stop and target; trailing stop is what triggers
        stop_price = round(entry_price * 0.97, 2)
        target_price = round(entry_price * 1.10, 2)
    elif scenario == "time":
        # Far stop and target; position expires via bars_held
        stop_price = round(entry_price * 0.97, 2)
        target_price = round(entry_price * 1.10, 2)
    elif scenario == "let_it_ride":
        # Tight target 1% above entry; Phase 1 hits target → converts to
        # trailing stop. Phase 2 drops below trailing stop → exit.
        target_price = round(entry_price * 1.01, 2)
        stop_price = round(entry_price * 0.97, 2)
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    return {
        "ticker": TICKER,
        "entry_price": entry_price,
        "entry_date": today,
        "stop_price": stop_price,
        "target_price": target_price,
        "trailing_stop": round(entry_price - p.trailing_stop_atr_mult * atr_val, 4),
        "highest_close": entry_price,
        "atr": atr_val,
        "bars_held": 0,
        "size": _SIM_SIZE,
        "p_rally": 0.95,
        "current_price": entry_price,
        "unrealized_pnl_pct": 0.0,
        "status": "open",
    }


def get_inject_prices(scenario: str, pos: dict) -> list[float]:
    """Return ordered inject prices for stream-based scenarios.

    trail  → [high_price_phase1, low_price_phase2]
    others → [single_exit_trigger_price]

    Note: for trail, Phase 2 price is computed from the *initial* pos trailing_stop.
    After Phase 1 injection tightens the stop, the runner re-reads pos from DB and
    recomputes the actual low price before injecting.
    """
    entry = pos["entry_price"]
    p = PARAMS

    if scenario == "target":
        return [round(pos["target_price"] * 1.002, 2)]

    elif scenario == "stop":
        effective_stop = max(pos["stop_price"], pos.get("trailing_stop", 0))
        return [round(effective_stop * 0.997, 2)]

    elif scenario == "trail":
        # Phase 1: +2.5% above entry — tightens trailing stop without hitting target
        high_price = round(entry * 1.025, 2)
        # Phase 2 placeholder: runner re-reads trailing_stop after Phase 1 completes
        atr_val = pos.get("atr", entry * p.default_atr_pct)
        new_trail_est = round(high_price - p.trailing_stop_atr_mult * atr_val, 4)
        low_price_est = round(new_trail_est * 0.996, 2)
        return [high_price, low_price_est]

    elif scenario == "time":
        return [entry]  # price is irrelevant for time stop

    elif scenario == "let_it_ride":
        # Phase 1: inject above target to trigger let-it-ride conversion
        above_target = round(pos["target_price"] * 1.002, 2)
        # Phase 2: after conversion, inject below trailing stop to trigger exit
        # Runner re-reads trailing_stop after Phase 1 completes
        atr_val = pos.get("atr", entry * p.default_atr_pct)
        trail_est = round(above_target - p.trailing_stop_atr_mult * atr_val, 4)
        low_est = round(trail_est * 0.996, 2)
        return [above_target, low_est]

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")
