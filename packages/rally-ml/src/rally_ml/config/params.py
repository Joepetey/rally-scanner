"""ML hyperparameters and pipeline configuration."""

from pydantic import BaseModel


class Params(BaseModel):
    # --- Timeframes ---
    lookbacks: tuple[int, ...] = (20, 40, 80)
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

    # --- Trading rules (conservative — best risk-adjusted per universe backtest) ---
    p_rally_threshold: float = 0.55
    comp_score_threshold: float = 0.60
    fail_dn_score_threshold: float = -1.0  # disabled: trap is a model feature, not a gate
    vol_target_k: float = 0.08         # risk budget scalar
    max_risk_frac: float = 0.15        # max position as fraction of equity
    min_position_size: float = 0.01    # minimum position size (skip if below)
    profit_atr_mult: float = 1.0       # take-profit in ATR multiples
    time_stop_bars: int = 3
    rv_exit_pct: float = 0.80          # RV percentile for exhaustion exit
    trailing_stop_atr_mult: float = 1.5  # trailing stop distance in ATR multiples
    fallback_stop_pct: float = 0.03      # fallback hard stop (3% below entry)
    limit_order_buffer_pct: float = 0.002  # limit price = midpoint(bid,ask) * (1 + buffer)
    profit_lock_pct: float = 0.02        # raise hard stop to +2% once that profit level is touched
    default_atr_pct: float = 0.02        # fallback ATR % when real ATR unavailable

    # --- Portfolio-level risk ---
    max_portfolio_exposure: float = 1.0  # max total exposure (1.0 = 100% of equity)
    max_drawdown_pct: float = 0.15       # circuit breaker threshold
    circuit_breaker_enabled: bool = True
    max_group_positions: int = 3         # max positions per asset group
    max_group_exposure: float = 0.50     # max exposure per asset group

    # --- Regime monitoring (Phase 1) ---
    regime_check_enabled: bool = True
    regime_cascade_threshold: int = 3    # N simultaneous shifts to trigger early scan

    # --- Proactive risk reduction (Phase 2) ---
    proactive_risk_enabled: bool = True
    risk_tier1_dd: float = 0.05          # tighten stops at 5% drawdown
    risk_tier2_dd: float = 0.10          # trim weakest at 10% drawdown
    risk_expanding_threshold: float = 0.80  # P(expanding) for per-position tightening
    risk_vix_spike_pct: float = 0.20     # VIX % daily change trigger

    # --- Signal queue & rotation ---
    signal_queue_max_age_days: int = 2       # discard queued signal after N days
    partial_sizing_enabled: bool = True      # scale down size to fit available capital
    rotation_enabled: bool = True            # exit weakest pos to fund better signal
    rotation_p_rally_margin: float = 0.08   # new signal must exceed weakest by this margin
    cooldown_days: int = 10                # skip re-entry on tickers closed within N days

    # --- Cash parking (SGOV) ---
    sgov_min_idle_fraction: float = 0.02  # don't park if idle < 2% of equity

    # --- Real-time streaming ---
    # Override via env vars: ALPACA_STREAM_ENABLED=0, ALPACA_DATA_FEED=sip,
    # STREAM_EVAL_THROTTLE_SECONDS=5
    stream_enabled: bool = True
    stream_eval_throttle_seconds: float = 10.0
    stream_data_feed: str = "iex"  # "iex" (free) or "sip" (paid consolidated)

    # --- Adaptive scan frequency (Phase 3) ---
    morning_scan_enabled: bool = True
    midday_scans_enabled: bool = True
    adaptive_alerts_enabled: bool = True
    fast_alert_interval: int = 5         # minutes between alerts in fast mode
    base_alert_interval: int = 15        # minutes between alerts in normal mode
    vix_fast_threshold: float = 30.0     # VIX level to trigger fast alerts
    stop_proximity_pct: float = 1.0      # % from stop to trigger fast alerts
    watchlist_p_rally_min: float = 0.35  # P(rally) threshold for watchlist


PARAMS = Params()


class PipelineConfig(BaseModel):
    # --- Parallelism ---
    n_workers: int = 4              # safe with OMP_NUM_THREADS=2 (4 workers × 2 threads = 8)
    # --- OHLCV disk cache ---
    cache_dir: str = "models/data_cache"  # under models/ for Railway volume
    cache_enabled: bool = True
    # --- HMM ---
    hmm_n_iter: int = 50            # converges in ~30-50 with diag covariance
    hmm_tol: float = 1e-3           # early stopping tolerance
    # --- Model freshness ---
    skip_fresh_days: int = 7        # skip if model trained < N days ago
    skip_fresh_enabled: bool = False


PIPELINE = PipelineConfig()
