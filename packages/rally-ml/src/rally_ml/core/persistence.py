"""
Model persistence — save and load trained model artifacts via joblib.

Directory structure:
    models/
        AAPL.joblib         # all model artifacts for AAPL
        MSFT.joblib
        ...

Manifest metadata is stored via an injected ManifestStore (configured at
application startup).  Call ``configure(store)`` before using save_model()
or load_manifest().
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import joblib

from ..config import AssetConfig
from .protocols import ManifestStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("APP_ROOT", Path.cwd()))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))

_store: ManifestStore | None = None


def configure(store: ManifestStore) -> None:
    """Inject a ManifestStore implementation (call once at startup)."""
    global _store
    _store = store


def save_model(ticker: str, artifacts: dict, asset_config: AssetConfig) -> None:
    """
    Save model artifacts for one asset.

    artifacts must contain:
        lr_model, lr_scaler, iso_calibrator,
        hmm_model, hmm_scaler, state_order,
        feature_cols, train_start, train_end
    """
    MODELS_DIR.mkdir(exist_ok=True)

    artifacts["asset_config"] = asset_config.model_dump()
    artifacts["saved_at"] = datetime.now().isoformat()

    path = MODELS_DIR / f"{ticker}.joblib"
    tmp_path = path.with_suffix(".tmp")
    joblib.dump(artifacts, tmp_path)
    tmp_path.rename(path)

    if _store is not None:
        _store.save_entry(ticker, {
            "saved_at": artifacts["saved_at"],
            "train_start": artifacts["train_start"],
            "train_end": artifacts["train_end"],
            "r_up": asset_config.r_up,
            "d_dn": asset_config.d_dn,
        })
    else:
        logger.warning("ManifestStore not configured — metadata not persisted for %s", ticker)


def load_model(ticker: str) -> dict:
    """Load model artifacts for one asset."""
    path = MODELS_DIR / f"{ticker}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model for {ticker}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Corrupt model artifact for {ticker}: {e}") from e


def load_manifest() -> dict:
    """Load the manifest of all trained models."""
    if _store is None:
        raise RuntimeError("ManifestStore not configured — call configure() at startup")
    return _store.load_all()
