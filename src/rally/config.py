"""
Rally Phase-Transition Detector — Configuration
All thresholds, parameters, and asset definitions in one place.
"""

from dataclasses import dataclass, field


@dataclass
class AssetConfig:
    ticker: str
    asset_class: str  # "equity" or "crypto"
    r_up: float       # min rally magnitude
    d_dn: float       # max adverse excursion allowed


ASSETS: dict[str, AssetConfig] = {
    # --- Indices ---
    "SPY": AssetConfig(ticker="SPY", asset_class="equity", r_up=0.02, d_dn=0.01),
    "QQQ": AssetConfig(ticker="QQQ", asset_class="equity", r_up=0.025, d_dn=0.012),
    # --- Mega-cap tech ---
    "AAPL": AssetConfig(ticker="AAPL", asset_class="equity", r_up=0.03, d_dn=0.015),
    "MSFT": AssetConfig(ticker="MSFT", asset_class="equity", r_up=0.03, d_dn=0.015),
    "NVDA": AssetConfig(ticker="NVDA", asset_class="equity", r_up=0.05, d_dn=0.025),
    "AMZN": AssetConfig(ticker="AMZN", asset_class="equity", r_up=0.035, d_dn=0.018),
    "GOOG": AssetConfig(ticker="GOOG", asset_class="equity", r_up=0.03, d_dn=0.015),
    "META": AssetConfig(ticker="META", asset_class="equity", r_up=0.04, d_dn=0.02),
    "TSLA": AssetConfig(ticker="TSLA", asset_class="equity", r_up=0.06, d_dn=0.03),
    # --- Other popular ---
    "AMD": AssetConfig(ticker="AMD", asset_class="equity", r_up=0.05, d_dn=0.025),
    "NFLX": AssetConfig(ticker="NFLX", asset_class="equity", r_up=0.04, d_dn=0.02),
    "JPM": AssetConfig(ticker="JPM", asset_class="equity", r_up=0.03, d_dn=0.015),
    # --- Crypto ---
    "BTC": AssetConfig(ticker="BTC-USD", asset_class="crypto", r_up=0.04, d_dn=0.02),
    "ETH": AssetConfig(ticker="ETH-USD", asset_class="crypto", r_up=0.05, d_dn=0.025),
}


@dataclass
class Params:
    # --- Timeframes ---
    lookbacks: tuple = (20, 40, 80)
    forward_horizon: int = 10           # H bars for label

    # --- Volatility compression ---
    atr_period: int = 20
    bb_period: int = 20
    rv_period: int = 20
    percentile_window: int = 252

    # --- Range structure ---
    range_period: int = 40

    # --- Failed breakdown ---
    trap_range_period: int = 20         # shorter range for breakdown detection
    trap_lookforward: int = 5           # k bars to confirm failure
    rsi_period: int = 14
    rv_trap_percentile: float = 0.70    # RV must be below this percentile

    # --- Trend ---
    ma_long: int = 200

    # --- Model ---
    walk_forward_train_years: int = 5
    walk_forward_test_years: int = 1
    train_live_features: bool = True    # use live-compatible features during retrain

    # --- Trading rules ---
    p_rally_threshold: float = 0.50
    comp_score_threshold: float = 0.55
    fail_dn_score_threshold: float = -1.0  # disabled: trap is a model feature, not a gate
    vol_target_k: float = 0.10         # risk budget scalar
    max_risk_frac: float = 0.25        # max position as fraction of equity
    min_position_size: float = 0.01    # minimum position size (skip if below)
    profit_atr_mult: float = 2.0       # take-profit in ATR multiples
    time_stop_bars: int = 10
    rv_exit_pct: float = 0.80          # RV percentile for exhaustion exit


PARAMS = Params()


@dataclass
class PipelineConfig:
    # --- Parallelism ---
    n_workers: int = 8              # max parallel training workers
    # --- OHLCV disk cache ---
    cache_dir: str = "data_cache"   # parquet cache directory
    cache_enabled: bool = True
    # --- HMM ---
    hmm_n_iter: int = 100           # reduced from 200 (converges in ~50-80)
    hmm_tol: float = 1e-3           # early stopping tolerance
    # --- Model freshness ---
    skip_fresh_days: int = 7        # skip if model trained < N days ago
    skip_fresh_enabled: bool = False


PIPELINE = PipelineConfig()
