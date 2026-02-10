"""
Model persistence — save and load trained model artifacts via joblib.

Directory structure:
    models/
        manifest.json       # {ticker: {saved_at, train_start, train_end, r_up, d_dn}}
        AAPL.joblib         # all model artifacts for AAPL
        MSFT.joblib
        ...
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib

from config import AssetConfig

MODELS_DIR = Path(__file__).parent / "models"


def save_model(ticker: str, artifacts: dict, asset_config: AssetConfig) -> None:
    """
    Save model artifacts for one asset.

    artifacts must contain:
        lr_model, lr_scaler, iso_calibrator,
        hmm_model, hmm_scaler, state_order,
        feature_cols, train_start, train_end
    """
    MODELS_DIR.mkdir(exist_ok=True)

    artifacts["asset_config"] = asdict(asset_config)
    artifacts["saved_at"] = datetime.now().isoformat()

    path = MODELS_DIR / f"{ticker}.joblib"
    joblib.dump(artifacts, path)

    # Update manifest
    manifest_path = MODELS_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest[ticker] = {
        "saved_at": artifacts["saved_at"],
        "train_start": artifacts["train_start"],
        "train_end": artifacts["train_end"],
        "r_up": asset_config.r_up,
        "d_dn": asset_config.d_dn,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_model(ticker: str) -> dict:
    """Load model artifacts for one asset."""
    path = MODELS_DIR / f"{ticker}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model for {ticker}")
    return joblib.load(path)


def load_manifest() -> dict:
    """Load the manifest of all trained models."""
    manifest_path = MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)
