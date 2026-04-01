"""Rally scanner configuration — assets, ML parameters, and trading presets.

Submodules:
  config.assets   — AssetConfig, ASSETS, ASSET_GROUPS, TICKER_TO_GROUP
  config.params   — Params, PARAMS, PipelineConfig, PIPELINE
  config.trading  — TradingConfig, CONFIGS, CONFIGS_BY_NAME
"""

from rally_ml.config.assets import ASSET_GROUPS, ASSETS, TICKER_TO_GROUP, AssetConfig
from rally_ml.config.params import PARAMS, PIPELINE, Params, PipelineConfig
from rally_ml.config.trading import CONFIGS, CONFIGS_BY_NAME, TradingConfig

__all__ = [
    "AssetConfig", "ASSETS", "ASSET_GROUPS", "TICKER_TO_GROUP",
    "Params", "PARAMS", "PipelineConfig", "PIPELINE",
    "TradingConfig", "CONFIGS", "CONFIGS_BY_NAME",
]
