# Rally Phase-Transition Detector

A quantitative system that detects the transition from volatility compression to directional rally across equities and crypto. It combines statistical learning (logistic regression + isotonic calibration), hidden Markov models for regime detection, and a multi-factor feature pipeline to generate probabilistic entry signals with structured risk management.

---

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Feature Engineering](#feature-engineering)
3. [Regime Detection (HMM)](#regime-detection-hmm)
4. [Label Construction](#label-construction)
5. [Model Architecture](#model-architecture)
6. [Trading System](#trading-system)
7. [Auto-Calibration](#auto-calibration)
8. [System Architecture](#system-architecture)
9. [Usage](#usage)
10. [Configuration](#configuration)
11. [Caveats & Limitations](#caveats--limitations)

---

## Theoretical Foundation

### The Compression-Expansion Cycle

Markets alternate between periods of low volatility (compression) and high volatility (expansion). This is not random — it is a well-documented empirical phenomenon rooted in market microstructure:

1. **Compression phase**: Participants reach temporary equilibrium. Realized volatility, ATR, and Bollinger Band width contract. Range narrows. This creates a "coiled spring" effect as limit orders accumulate around the range boundaries.

2. **Transition event**: A catalyst (earnings, macro data, flow imbalance) breaks the equilibrium. The accumulated order book pressure unwinds directionally.

3. **Expansion phase**: Volatility expands as the new trend establishes. Range widens, momentum builds, and volume confirms the direction.

The system targets the **transition point** — the moment when compression has reached an extreme and the probability of directional expansion is elevated.

### Bear Trap as a Directional Signal

A "failed breakdown" or bear trap occurs when price briefly penetrates the lower range boundary during compression, then quickly recovers above it. This pattern is significant because:

- The breakdown triggers stop-losses from existing longs and attracts new shorts
- When the breakdown fails to follow through, those shorts are trapped
- Forced short-covering creates upward pressure that can catalyze the compression-to-rally transition
- The failure itself is information: sellers couldn't sustain control despite the volatility environment favoring downside

The system quantifies this via `FAIL_DN_SCORE` — a continuous measure of trap strength based on the speed and magnitude of recovery relative to ATR.

### Regime-Dependent Probability

Not all compression resolves upward. The system uses a Hidden Markov Model to classify the current volatility regime (compressed / normal / expanding) and uses the transition probability from compressed to expanding as an additional feature. This captures the *dynamics* of the regime, not just its level.

---

## Feature Engineering

The system computes 13 features organized into base features (10) and HMM-derived features (3).

### Base Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `COMP_SCORE` | Composite compression score: `1 - mean(p_ATR, p_BB, p_RV)` where each is a 252-day percentile rank | Higher values = more compressed. Combines three independent volatility measures into a single compression gauge |
| `RangeWidth` | `(RangeHigh - RangeLow) / Close` over 40-bar window | Absolute range context — tight ranges precede breakouts |
| `RangeTightness` | `RangeWidth / ATR_pct` | Range relative to current volatility — distinguishes "tight range in low vol" from "tight range in high vol" |
| `FAIL_DN_SCORE` | Continuous bear-trap strength: recovery magnitude / ATR, with RSI bonus | Core signal: failed breakdowns during compression predict upside |
| `Trend` | Binary: Close > 200-day MA | Trend filter — compression resolving with trend has higher base rate |
| `COMP_x_FAIL` | `COMP_SCORE * FAIL_DN_SCORE` | Interaction: traps during compression are more meaningful than traps in normal vol |
| `p_VOL` | 252-day percentile rank of volume ratio (current / 20-day avg) | Volume confirmation — breakout bars with elevated volume have higher conviction |
| `GOLDEN_CROSS` | Binary: 50-day MA > 200-day MA | Momentum context: trend acceleration filter |
| `MACD_HIST` | `(EMA12 - EMA26 - Signal9) / Close` | Normalized momentum: positive histogram = strengthening momentum |
| `VIX_PCTILE` | 252-day percentile rank of VIX close | Market-wide fear gauge: elevated VIX during compression can indicate capitulation |

### HMM-Derived Features

| Feature | Description |
|---------|-------------|
| `P_compressed` | Posterior probability of being in the compressed regime |
| `P_expanding` | Posterior probability of being in the expanding regime |
| `HMM_transition_signal` | `P_compressed * P(compressed -> expanding)` from the HMM transition matrix |

### Trap Engine Details

The bear trap detector (`compute_failed_breakdown`) works as follows:

1. Compute a 20-bar rolling range low (shorter than the main 40-bar range for more frequent signals)
2. Identify breakdown bars where `Low <= RangeLow_prev + 0.3 * ATR` (probe tolerance)
3. Filter: only consider breakdowns where realized volatility is below the 70th percentile (compressed environment)
4. Look forward 5 bars for recovery: `score = (Close_j - RangeLow) / ATR`
5. Apply RSI bonus: if RSI is rising during recovery, multiply score by 1.5x

In **live mode**, the score is lagged by 5 bars to eliminate forward-looking bias — you see the confirmed trap from 5 days ago.

---

## Regime Detection (HMM)

A 3-state Gaussian Hidden Markov Model is fit on three volatility observables: `[RV, ATR_pct, BB_width]`.

**Why HMM over simple thresholds?**
- Regimes are latent — you can't observe them directly, only their emissions (volatility metrics)
- The HMM captures transition dynamics: P(compressed -> expanding) varies over time based on the fitted transition matrix
- It provides probabilistic state assignment rather than hard labels, which gives the logistic regression smoother features to work with

**Fitting procedure:**
1. Standardize features with `StandardScaler`
2. Fit `GaussianHMM(n_components=3, covariance_type="full")` on training data only
3. Label states post-hoc by sorting mean RV emission: lowest RV = compressed, highest = expanding
4. Extract filtered posterior probabilities via the forward algorithm

The HMM is **refit per fold** in walk-forward training to prevent information leakage.

---

## Label Construction

The target variable `RALLY_ST` is a binary label defined over a forward horizon H (default: 10 bars):

```
RALLY_ST(t) = 1  if:
    max(Close[t+1 : t+H]) / Close[t] - 1  >=  r_up    (sufficient upside)
AND
    min(Low[t+1 : t+H])   / Close[t] - 1   >= -d_dn    (limited drawdown)
```

This is **not** a simple "did price go up" label. It requires:
- **Magnitude**: the rally must reach at least `r_up` (e.g., +3% for AAPL)
- **Path quality**: the drawdown during the rally must not exceed `d_dn` (e.g., -1.5% for AAPL)

This encodes the trading reality: a +5% rally that first drops -8% is not tradeable with reasonable stops.

Thresholds are **asset-specific** — higher-volatility names (TSLA: r_up=6%, d_dn=3%) require proportionally larger moves to qualify.

---

## Model Architecture

### Walk-Forward Logistic Regression

The model uses **walk-forward validation** to avoid lookahead bias:

```
Fold 1: Train 2000-2004, Test 2005
Fold 2: Train 2001-2005, Test 2006
Fold 3: Train 2002-2006, Test 2007
...
```

- Training window: 5 years (rolling)
- Test window: 1 year
- All out-of-sample predictions are concatenated to form the full backtest series

**Why logistic regression?**
- Produces calibrated probabilities (after isotonic correction)
- Interpretable coefficients reveal which features drive predictions
- Low parameter count reduces overfitting on the small positive-class counts typical of rare-event prediction
- `class_weight="balanced"` handles the ~5-15% positive label rate

### Isotonic Calibration

Raw logistic regression probabilities are often poorly calibrated for rare events. The system applies isotonic regression as a post-hoc calibration step:

1. Hold out the last 20% of training data as a calibration set
2. Fit `IsotonicRegression(y_min=0, y_max=1)` mapping raw probabilities to observed frequencies
3. Apply to test set predictions

This ensures that when the model says "65% probability of rally," it actually rallies roughly 65% of the time.

---

## Trading System

### Entry Conditions

A long entry signal fires when **all** conditions are met:

| Condition | Default Threshold |
|-----------|------------------|
| Calibrated P(rally) | > 50% |
| Compression score | > 0.55 |
| Trend = 1 (equities only) | Close > 200-day MA |

### Position Sizing

Volatility-targeted sizing scales position size by conviction and inverse volatility:

```
size = k * (P_rally - 0.5) / ATR_pct
```

Where `k = 0.10` (risk budget scalar). Clamped to [0%, 25%] of equity. This means:
- Higher conviction (higher P_rally) -> larger position
- Higher volatility (higher ATR_pct) -> smaller position
- A 70% probability signal in a 1% daily ATR stock gets ~2x the size of a 55% signal in a 2% ATR stock

### Exit Rules (5 layers)

Exits are checked in priority order each bar:

1. **Structural stop-loss**: Price hits RangeLow (the prior consolidation floor). Fallback: -3% hard stop if RangeLow is invalid.
2. **Profit target**: Price reaches entry + 2.0 * ATR. Captures the expected expansion magnitude.
3. **Trailing stop**: After 2+ bars, exit if close < (highest_close - 1.5 * ATR). Ratchets up as price advances, never down.
4. **Time stop**: Exit after 10 bars regardless. The thesis is a short-duration event — if it hasn't materialized, the setup has failed.
5. **Volatility exhaustion**: Exit if RV percentile > 80% and close < prior close. The expansion phase is overextended.

### Portfolio Construction

When running all assets, positions share a single equity pool:
- Each position is sized from current portfolio equity
- Total exposure is capped at 100%
- PnL is realized at exit and added back to the equity pool

---

## Auto-Calibration

For the expanded universe (500+ stocks), hand-tuning `r_up` and `d_dn` per asset is impractical. The auto-calibrator derives thresholds from historical volatility:

```
median_daily_rv = median(20-day rolling RV over last 252 days)
rv_10 = median_daily_rv * sqrt(10)          # scale to 10-bar horizon
r_up  = rv_10 * 0.80                        # ~80% of a 1-sigma move
d_dn  = r_up / 2.0                          # maintain 2:1 reward/risk ratio
```

Clamped to: `r_up` in [1.5%, 10%], `d_dn` in [0.8%, 5%].

This preserves the 2:1 ratio observed in the hand-tuned original 14 assets and scales naturally with each stock's volatility profile.

---

## System Architecture

```
market-rally/
  config.py          Parameters, asset definitions, thresholds
  data.py            yfinance daily OHLCV + VIX fetcher with session caching
  features.py        10 base features: compression, range, traps, trend, volume, MACD, VIX
  hmm.py             3-state Gaussian HMM for regime detection (+3 features)
  labels.py          RALLY_ST binary label (forward-looking)
  model.py           Walk-forward logistic regression + isotonic calibration
  trading.py         Signal generation, vol-targeted position sizing, trade simulation
  backtest.py        Equity curve computation, metrics, diagnostic reports
  calibrate.py       Auto-calibrate r_up/d_dn from historical volatility
  persistence.py     Save/load trained models with joblib
  positions.py       Track open positions in JSON across scanner runs
  universe.py        S&P 500 + Nasdaq 100 universe from Wikipedia (cached 30 days)
  main.py            CLI: single-asset or 14-asset backtest with walk-forward
  retrain.py         CLI: weekly model retraining for full universe
  scanner.py         CLI: daily scan — load models, fetch prices, generate alerts
  backtest_universe.py  Full universe 20-year backtest with multiple config sweeps
  optimize.py        Parameter sweep across 11 named configurations
```

### Data Flow

```
                 retrain.py (weekly)
                      |
  yfinance ──> data.py ──> features.py ──> labels.py
                                |               |
                            hmm.py          model.py
                                |               |
                                └──> models/*.joblib
                                         |
                 scanner.py (daily) ─────┘
                      |
              load model + fetch latest data
                      |
              predict P(rally) ──> generate signals
                      |
              positions.json ──> terminal alerts
```

---

## Usage

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Weekly Retraining

Train models for the full S&P 500 + Nasdaq 100 universe:

```bash
python retrain.py                              # all ~550 tickers
python retrain.py --tickers AAPL MSFT SPY      # specific tickers only
python retrain.py --validate                   # compare auto-cal vs hand-tuned
```

### Daily Scanning

```bash
python scanner.py                              # scan all trained assets
python scanner.py --positions                  # include open position tracking
python scanner.py --config conservative        # use conservative thresholds
python scanner.py --tickers AAPL NVDA MSFT     # scan specific tickers
```

Scanner output includes:
- **Market breadth**: % above 50/200-day MA, golden crosses, VIX sentiment
- **New signals**: tickers meeting all entry criteria with sizing and levels
- **Watchlist**: near-miss tickers (P_rally > 35%) approaching signal threshold
- **Full probability ranking**: all scanned assets sorted by rally probability
- **Open positions**: current holdings with PnL, stops, targets (with `--positions`)

### Backtesting

```bash
# Single asset
python main.py --asset SPY --plot

# Original 14 assets with portfolio aggregation
python main.py --run-all --plot

# Full universe backtest (500+ assets, ~2-4 hours)
python backtest_universe.py --plot

# Parameter optimization sweep
python optimize.py
```

---

## Configuration

### Named Configs (scanner.py)

| Config | P(rally) | Comp | MaxSize | ProfitATR | TimeStop |
|--------|----------|------|---------|-----------|----------|
| `baseline` | 50% | 0.55 | 25% | 2.0x | 10 bars |
| `conservative` | 60% | 0.60 | 15% | 1.5x | 8 bars |
| `aggressive` | 45% | 0.50 | 30% | 2.5x | 12 bars |
| `concentrated` | 55% | 0.55 | 40% | 2.0x | 10 bars |

### Key Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `forward_horizon` | 10 | Label lookforward window (bars) |
| `atr_period` | 20 | ATR calculation period |
| `bb_period` | 20 | Bollinger Band period |
| `rv_period` | 20 | Realized volatility period |
| `percentile_window` | 252 | Rolling percentile lookback (~1 year) |
| `range_period` | 40 | Main range calculation period |
| `trap_range_period` | 20 | Shorter range for breakdown detection |
| `trap_lookforward` | 5 | Bars to confirm failed breakdown |
| `rv_trap_percentile` | 0.70 | Max RV percentile to qualify trap |
| `walk_forward_train_years` | 5 | Training window per fold |
| `walk_forward_test_years` | 1 | Test window per fold |
| `vol_target_k` | 0.10 | Position sizing risk scalar |
| `profit_atr_mult` | 2.0 | Take-profit in ATR multiples |
| `time_stop_bars` | 10 | Maximum holding period |
| `rv_exit_pct` | 0.80 | RV percentile for exhaustion exit |

---

## Caveats & Limitations

**This is a research/educational tool, not investment advice.**

- **Survivorship bias**: The S&P 500 universe is today's constituents tested historically. Stocks that went bankrupt or were delisted are excluded, inflating results.
- **No transaction costs**: Backtests do not include commissions, slippage, or bid-ask spread. Real execution will reduce returns.
- **Look-ahead in calibration**: Auto-calibrated thresholds use the full price history, including the test period. In live use this is not an issue (you calibrate on available data), but it flatters backtest results.
- **Data quality**: yfinance data may have gaps, splits, or adjustments that differ from institutional-grade feeds.
- **Regime dependence**: The system performs best in markets that exhibit clear compression-expansion cycles. Extended trending or choppy markets may produce poor results.
- **Model staleness**: Models should be retrained weekly. Stale models (>30 days) may have degraded calibration.
- **Single timeframe**: All analysis is on daily bars. Intraday dynamics are not captured.

---

## Dependencies

- Python 3.10+
- numpy, pandas, scikit-learn, matplotlib
- yfinance (Yahoo Finance API)
- hmmlearn (Hidden Markov Models)
- joblib (model persistence)
- lxml (Wikipedia universe scraping)
