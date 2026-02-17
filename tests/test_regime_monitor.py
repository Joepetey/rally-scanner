"""Tests for regime shift monitoring."""

from rally.regime_monitor import (
    _classify_regime,
    _is_significant_transition,
    _load_regime_states,
    _save_regime_states,
    is_cascade,
)


def test_classify_regime_compressed():
    assert _classify_regime(0.8, 0.1, 0.1) == "compressed"


def test_classify_regime_normal():
    assert _classify_regime(0.1, 0.7, 0.2) == "normal"


def test_classify_regime_expanding():
    assert _classify_regime(0.05, 0.15, 0.80) == "expanding"


def test_significant_transition_compressed_to_expanding():
    assert _is_significant_transition("compressed", "expanding", 0.1, 0.1, 0.8)


def test_significant_transition_compressed_to_normal_with_high_p_expanding():
    assert _is_significant_transition("compressed", "normal", 0.2, 0.4, 0.45)


def test_significant_transition_compressed_to_normal_low_p_expanding():
    """Not significant: compressed → normal but P(expanding) < 0.4."""
    assert not _is_significant_transition("compressed", "normal", 0.2, 0.6, 0.2)


def test_significant_transition_normal_to_expanding_high():
    assert _is_significant_transition("normal", "expanding", 0.05, 0.25, 0.70)


def test_significant_transition_normal_to_expanding_low():
    """Not significant: normal → expanding but P(expanding) < 0.6."""
    assert not _is_significant_transition("normal", "expanding", 0.1, 0.4, 0.5)


def test_not_significant_normal_to_compressed():
    assert not _is_significant_transition("normal", "compressed", 0.7, 0.2, 0.1)


def test_state_persistence(tmp_path, monkeypatch):
    """States are saved and loaded correctly."""
    state_file = tmp_path / "regime_states.json"
    import rally.regime_monitor as rm
    monkeypatch.setattr(rm, "REGIME_STATE_FILE", state_file)

    states = {
        "AAPL": {
            "p_compressed": 0.8,
            "p_normal": 0.1,
            "p_expanding": 0.1,
            "dominant_regime": "compressed",
            "timestamp": "2024-01-01T00:00:00",
        },
    }
    _save_regime_states(states)
    loaded = _load_regime_states()
    assert loaded["AAPL"]["dominant_regime"] == "compressed"
    assert loaded["AAPL"]["p_compressed"] == 0.8


def test_state_persistence_empty(tmp_path, monkeypatch):
    """Empty file returns empty dict."""
    state_file = tmp_path / "regime_states.json"
    import rally.regime_monitor as rm
    monkeypatch.setattr(rm, "REGIME_STATE_FILE", state_file)

    loaded = _load_regime_states()
    assert loaded == {}


def test_cascade_threshold_met():
    transitions = [
        {"ticker": "AAPL", "prev_regime": "compressed", "new_regime": "expanding",
         "p_compressed": 0.1, "p_normal": 0.1, "p_expanding": 0.8},
        {"ticker": "MSFT", "prev_regime": "compressed", "new_regime": "expanding",
         "p_compressed": 0.1, "p_normal": 0.2, "p_expanding": 0.7},
        {"ticker": "GOOG", "prev_regime": "compressed", "new_regime": "normal",
         "p_compressed": 0.2, "p_normal": 0.5, "p_expanding": 0.3},
    ]
    assert is_cascade(transitions)


def test_cascade_threshold_not_met():
    transitions = [
        {"ticker": "AAPL", "prev_regime": "compressed", "new_regime": "expanding",
         "p_compressed": 0.1, "p_normal": 0.1, "p_expanding": 0.8},
    ]
    assert not is_cascade(transitions)


def test_cascade_empty():
    assert not is_cascade([])
