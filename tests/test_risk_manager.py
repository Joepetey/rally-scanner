"""Tests for proactive risk management."""

from rally.config import PARAMS
from rally.risk_manager import evaluate


def _make_positions(n=3, pnl_pcts=None):
    """Create n test positions with optional PnL percentages."""
    if pnl_pcts is None:
        pnl_pcts = [2.0, -1.0, -5.0][:n]
    positions = []
    for i, pnl in enumerate(pnl_pcts):
        ticker = ["AAPL", "MSFT", "GOOG", "META", "NVDA"][i % 5]
        positions.append({
            "ticker": ticker,
            "entry_price": 100.0,
            "current_price": 100.0 * (1 + pnl / 100),
            "unrealized_pnl_pct": pnl,
            "stop_price": 95.0,
            "trailing_stop": 97.0,
            "highest_close": 102.0,
            "atr": 2.0,
            "size": 0.10,
            "bars_held": 3,
        })
    return positions


def test_no_actions_when_healthy(monkeypatch):
    """No actions when drawdown is low and no regime issues."""
    monkeypatch.setattr(PARAMS, "proactive_risk_enabled", True)
    positions = _make_positions(2, [3.0, 1.0])

    # Mock VIX check to return no spike
    import rally.risk_manager as rm
    monkeypatch.setattr(rm, "check_vix_spike",
                        lambda: {"is_spike": False, "change_pct": 0.0, "vix_level": 20.0})

    actions = evaluate(100_000, positions, drawdown=0.02)
    assert len(actions) == 0


def test_tier1_tightens_all_stops(monkeypatch):
    """Tier 1 (5-10% DD): all stops tightened to 1.0×ATR."""
    monkeypatch.setattr(PARAMS, "proactive_risk_enabled", True)
    import rally.risk_manager as rm
    monkeypatch.setattr(rm, "check_vix_spike",
                        lambda: {"is_spike": False, "change_pct": 0.0, "vix_level": 20.0})

    positions = _make_positions(3)
    actions = evaluate(100_000, positions, drawdown=0.06)

    tightens = [a for a in actions if a.action_type == "tighten_stop"]
    assert len(tightens) == 3
    for a in tightens:
        assert a.new_trail_atr_mult == 1.0
        assert "Tier 1" in a.reason


def test_tier2_closes_weakest(monkeypatch):
    """Tier 2 (10-15% DD): weakest position closed + all stops tightened."""
    monkeypatch.setattr(PARAMS, "proactive_risk_enabled", True)
    import rally.risk_manager as rm
    monkeypatch.setattr(rm, "check_vix_spike",
                        lambda: {"is_spike": False, "change_pct": 0.0, "vix_level": 20.0})

    positions = _make_positions(3, [2.0, -1.0, -5.0])
    actions = evaluate(100_000, positions, drawdown=0.11)

    closes = [a for a in actions if a.action_type == "close_position"]
    assert len(closes) == 1
    assert closes[0].ticker == "GOOG"  # worst PnL at -5%
    assert "Tier 2" in closes[0].reason

    # GOOG should NOT also have a tighten action
    tightens = [a for a in actions if a.action_type == "tighten_stop"]
    tighten_tickers = {a.ticker for a in tightens}
    assert "GOOG" not in tighten_tickers


def test_expanding_regime_tightens_individual(monkeypatch):
    """Per-position: P(expanding) > 0.8 tightens that position's stop."""
    monkeypatch.setattr(PARAMS, "proactive_risk_enabled", True)
    import rally.risk_manager as rm
    monkeypatch.setattr(rm, "check_vix_spike",
                        lambda: {"is_spike": False, "change_pct": 0.0, "vix_level": 20.0})

    positions = _make_positions(2, [3.0, 1.0])
    regime_states = {
        "AAPL": {"p_expanding": 0.85, "p_compressed": 0.05, "p_normal": 0.10},
        "MSFT": {"p_expanding": 0.30, "p_compressed": 0.40, "p_normal": 0.30},
    }

    actions = evaluate(100_000, positions, regime_states=regime_states, drawdown=0.02)

    assert len(actions) == 1
    assert actions[0].ticker == "AAPL"
    assert actions[0].new_trail_atr_mult == 0.5
    assert "regime" in actions[0].reason.lower() or "Regime" in actions[0].reason


def test_vix_spike_tightens_all(monkeypatch):
    """VIX spike: all stops tightened by 30%."""
    monkeypatch.setattr(PARAMS, "proactive_risk_enabled", True)
    import rally.risk_manager as rm
    monkeypatch.setattr(rm, "check_vix_spike",
                        lambda: {"is_spike": True, "change_pct": 0.25, "vix_level": 35.0})

    positions = _make_positions(2, [2.0, 1.0])
    actions = evaluate(100_000, positions, drawdown=0.02)

    tightens = [a for a in actions if a.action_type == "tighten_stop"]
    assert len(tightens) == 2
    for a in tightens:
        # 1.5 * 0.7 = 1.05
        assert a.new_trail_atr_mult == 1.05
        assert "VIX" in a.reason


def test_disabled_returns_empty(monkeypatch):
    """Proactive risk disabled returns no actions."""
    monkeypatch.setattr(PARAMS, "proactive_risk_enabled", False)
    actions = evaluate(100_000, _make_positions(3), drawdown=0.15)
    assert actions == []


def test_empty_positions_returns_empty(monkeypatch):
    """No positions returns no actions."""
    monkeypatch.setattr(PARAMS, "proactive_risk_enabled", True)
    actions = evaluate(100_000, [], drawdown=0.10)
    assert actions == []
