"""Trading configuration presets — used by scanner and backtest."""

from pydantic import BaseModel


class TradingConfig(BaseModel):
    name: str
    p_rally: float
    comp_score: float
    max_risk: float
    vol_k: float
    profit_atr: float
    time_stop: int
    leverage: float
    cash_yield: float  # annual yield on idle capital


CONFIGS: list[TradingConfig] = [
    # --- Baseline (current) ---
    TradingConfig(name="Baseline", p_rally=0.50, comp_score=0.55, max_risk=0.25,
                  vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=1.0, cash_yield=0.0),

    # --- Baseline + cash yield ---
    TradingConfig(name="Base+Cash4%", p_rally=0.50, comp_score=0.55, max_risk=0.25,
                  vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=1.0, cash_yield=0.04),

    # --- Aggressive: lower thresholds, longer holds ---
    TradingConfig(name="Aggressive", p_rally=0.40, comp_score=0.45, max_risk=0.30,
                  vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=1.0, cash_yield=0.0),

    TradingConfig(name="Aggr+Cash4%", p_rally=0.40, comp_score=0.45, max_risk=0.30,
                  vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=1.0, cash_yield=0.04),

    # --- Concentrated: fewer trades, bigger bets ---
    TradingConfig(name="Concentrated", p_rally=0.55, comp_score=0.60, max_risk=0.40,
                  vol_k=0.15, profit_atr=2.5, time_stop=12, leverage=1.0, cash_yield=0.0),

    # --- Leveraged variants ---
    TradingConfig(name="Base 2x Lev", p_rally=0.50, comp_score=0.55, max_risk=0.25,
                  vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=2.0, cash_yield=0.04),

    TradingConfig(name="Base 3x Lev", p_rally=0.50, comp_score=0.55, max_risk=0.25,
                  vol_k=0.10, profit_atr=2.0, time_stop=10, leverage=3.0, cash_yield=0.04),

    TradingConfig(name="Aggr 2x Lev", p_rally=0.40, comp_score=0.45, max_risk=0.30,
                  vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=2.0, cash_yield=0.04),

    # --- Max return: aggressive + 3x ---
    TradingConfig(name="Aggr 3x Lev", p_rally=0.40, comp_score=0.45, max_risk=0.30,
                  vol_k=0.12, profit_atr=3.0, time_stop=15, leverage=3.0, cash_yield=0.04),

    # --- Conservative: minimize drawdown ---
    TradingConfig(name="Conservative", p_rally=0.55, comp_score=0.60, max_risk=0.15,
                  vol_k=0.08, profit_atr=2.0, time_stop=8, leverage=1.0, cash_yield=0.05),

    TradingConfig(name="Cons 2x Lev", p_rally=0.55, comp_score=0.60, max_risk=0.15,
                  vol_k=0.08, profit_atr=2.0, time_stop=8, leverage=2.0, cash_yield=0.05),
]

CONFIGS_BY_NAME: dict[str, TradingConfig] = {
    c.name.lower().replace(" ", "_"): c for c in CONFIGS
}
CONFIGS_BY_NAME.update({
    "baseline": CONFIGS[0],
    "conservative": CONFIGS[9],
    "aggressive": CONFIGS[2],
    "concentrated": CONFIGS[4],
})
