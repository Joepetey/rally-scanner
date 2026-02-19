"""Regime shift monitoring — detect HMM state transitions across assets.

Periodically checks HMM regime probabilities for all assets and alerts
on significant transitions (e.g. compressed → expanding).
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from ..config import PARAMS
from ..utils import atomic_json_write

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REGIME_STATE_FILE = PROJECT_ROOT / "models" / "regime_states.json"


def _classify_regime(p_compressed: float, p_normal: float, p_expanding: float) -> str:
    """Return the dominant regime name from state probabilities."""
    probs = {"compressed": p_compressed, "normal": p_normal, "expanding": p_expanding}
    return max(probs, key=probs.get)


def _load_regime_states() -> dict:
    """Load previous regime states from disk."""
    if not REGIME_STATE_FILE.exists():
        return {}
    try:
        with open(REGIME_STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_regime_states(states: dict) -> None:
    """Save regime states to disk atomically."""
    atomic_json_write(REGIME_STATE_FILE, states)


def check_regime_shifts(tickers: list[str] | None = None) -> list[dict]:
    """Check HMM regime states for all trained assets, detect transitions.

    Returns list of transition events:
        {ticker, prev_regime, new_regime, p_compressed, p_normal, p_expanding}
    """
    from datetime import timedelta

    from ..core.data import fetch_daily_batch, fetch_vix_safe
    from ..core.features import build_features
    from ..core.hmm import predict_hmm_probs
    from ..core.persistence import load_manifest, load_model

    manifest = load_manifest()
    if not manifest:
        return []

    if tickers:
        check_tickers = [t for t in tickers if t in manifest]
    else:
        check_tickers = sorted(manifest.keys())

    if not check_tickers:
        return []

    # Fetch data (need ~300 bars for features + HMM)
    lookback_days = 500
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    vix_data = fetch_vix_safe(start=start, verbose=False)

    try:
        ohlcv_cache = fetch_daily_batch(check_tickers, start=start)
    except Exception:
        ohlcv_cache = {}

    prev_states = _load_regime_states()
    new_states = {}
    transitions = []

    for ticker in check_tickers:
        try:
            artifacts = load_model(ticker)
            hmm_model = artifacts.get("hmm_model")
            hmm_scaler = artifacts.get("hmm_scaler")
            state_order = artifacts.get("state_order")

            if hmm_model is None:
                continue

            df = ohlcv_cache.get(ticker)
            if df is None or len(df) < 300:
                continue

            from ..core.data import merge_vix
            if vix_data is not None:
                df = merge_vix(df, vix_data)

            df = build_features(df, live=True)

            hmm_probs = predict_hmm_probs(hmm_model, hmm_scaler, state_order, df)
            last = hmm_probs.iloc[-1]

            p_c = float(last["P_compressed"])
            p_n = float(last["P_normal"])
            p_e = float(last["P_expanding"])
            new_regime = _classify_regime(p_c, p_n, p_e)

            new_states[ticker] = {
                "p_compressed": round(p_c, 4),
                "p_normal": round(p_n, 4),
                "p_expanding": round(p_e, 4),
                "dominant_regime": new_regime,
                "timestamp": datetime.now().isoformat(),
            }

            # Compare to previous state
            prev = prev_states.get(ticker)
            if prev:
                prev_regime = prev["dominant_regime"]
                if prev_regime != new_regime and _is_significant_transition(
                    prev_regime, new_regime, p_c, p_n, p_e
                ):
                    transitions.append({
                        "ticker": ticker,
                        "prev_regime": prev_regime,
                        "new_regime": new_regime,
                        "p_compressed": round(p_c, 4),
                        "p_normal": round(p_n, 4),
                        "p_expanding": round(p_e, 4),
                    })

        except Exception as e:
            logger.warning("Regime check failed for %s: %s", ticker, e)

    # Merge new states into previous (preserve tickers we didn't check)
    prev_states.update(new_states)
    _save_regime_states(prev_states)

    if transitions:
        logger.info(
            "Regime shifts detected: %s",
            ", ".join(f"{t['ticker']} {t['prev_regime']}→{t['new_regime']}" for t in transitions),
        )

    return transitions


def _is_significant_transition(
    prev: str, new: str,
    p_c: float, p_n: float, p_e: float,
) -> bool:
    """Determine if a regime transition is significant enough to alert on.

    Significant transitions:
      - compressed → expanding (always)
      - compressed → normal with P(expanding) > 0.4
      - normal → expanding with P(expanding) > 0.6
    """
    if prev == "compressed" and new == "expanding":
        return True
    if prev == "compressed" and new == "normal" and p_e > 0.4:
        return True
    if prev == "normal" and new == "expanding" and p_e > 0.6:
        return True
    return False


def is_cascade(transitions: list[dict]) -> bool:
    """Check if enough simultaneous shifts trigger an early scan."""
    expanding_shifts = sum(
        1 for t in transitions
        if t["new_regime"] == "expanding"
        or (t["prev_regime"] == "compressed" and t["new_regime"] != "compressed")
    )
    return expanding_shifts >= PARAMS.regime_cascade_threshold


def get_regime_states() -> dict:
    """Return the current cached regime states."""
    return _load_regime_states()
