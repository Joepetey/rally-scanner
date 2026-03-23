"""
Model persistence — save and load trained model artifacts via joblib.

Directory structure:
    models/
        AAPL.joblib         # all model artifacts for AAPL
        MSFT.joblib
        ...

Manifest metadata is stored in the `model_manifest` PostgreSQL table.
"""

from datetime import datetime
from pathlib import Path

import joblib

from config import AssetConfig
from db.models import load_manifest as _db_load_manifest
from db.models import save_manifest_entry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


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
    joblib.dump(artifacts, path)

    save_manifest_entry(ticker, {
        "saved_at": artifacts["saved_at"],
        "train_start": artifacts["train_start"],
        "train_end": artifacts["train_end"],
        "r_up": asset_config.r_up,
        "d_dn": asset_config.d_dn,
    })


def load_model(ticker: str) -> dict:
    """Load model artifacts for one asset."""
    path = MODELS_DIR / f"{ticker}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model for {ticker}")
    return joblib.load(path)


def load_manifest() -> dict:
    """Load the manifest of all trained models."""
    return _db_load_manifest()
