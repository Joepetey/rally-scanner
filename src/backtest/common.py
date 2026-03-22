"""Backtest primitives — walk-forward training, signal generation, trade simulation,
and portfolio-level performance evaluation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import CONFIGS, CONFIGS_BY_NAME, PARAMS, TradingConfig
from core.features import ALL_FEATURE_COLS
from core.hmm import fit_hmm, predict_hmm_probs


def generate_signals_fast(preds: pd.DataFrame, cfg: TradingConfig,
                          require_trend: bool) -> pd.Series:
    signal = (
        (preds["P_RALLY"] > cfg.p_rally)
        & (preds["COMP_SCORE"] > cfg.comp_score)
    )
    if require_trend:
        signal = signal & (preds["Trend"] == 1.0)
    return signal


def simulate_trades_fast(preds: pd.DataFrame, signal: pd.Series,
                         cfg: TradingConfig, close_only_tp: bool = False) -> pd.DataFrame:
    """Simulate trades with config-specific parameters.

    Args:
        close_only_tp: If True, profit target is only triggered when the *close*
            reaches the target (matching production behaviour).  Default False uses
            the intraday *high*, which is optimistic but standard for limit-order
            backtests.
    """
    p = PARAMS
    n = len(preds)
    close = preds["Close"].values
    high = preds["High"].values
    low = preds["Low"].values
    atr = preds["ATR"].values
    rv_pct = preds["p_RV"].values if "p_RV" in preds.columns else np.full(n, 0.5)

    trades = []
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    stop_price = 0.0
    trailing_stop = 0.0
    size = 0.0
    bars_held = 0
    highest_close = 0.0

    for i in range(n):
        if in_trade:
            bars_held += 1
            if close[i] > highest_close:
                highest_close = close[i]
                new_trail = highest_close - p.trailing_stop_atr_mult * atr[entry_idx]
                trailing_stop = max(trailing_stop, new_trail)

            exit_reason = None
            exit_price = close[i]

            tp_level = entry_price + cfg.profit_atr * atr[entry_idx]
            tp_check = close[i] >= tp_level if close_only_tp else high[i] >= tp_level

            if low[i] <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
            elif tp_check:
                exit_price = tp_level if not close_only_tp else close[i]
                exit_reason = "profit_target"
            elif bars_held >= 2 and close[i] < trailing_stop:
                exit_reason = "trail_stop"
            elif bars_held >= cfg.time_stop:
                exit_reason = "time_stop"
            elif rv_pct[i] > 0.80 and close[i] < close[i - 1]:
                exit_reason = "vol_exhaustion"

            if exit_reason:
                pnl_pct = exit_price / entry_price - 1
                trades.append({
                    "entry_date": preds.index[entry_idx],
                    "exit_date": preds.index[i],
                    "pnl_pct": pnl_pct,
                    "pnl_sized": pnl_pct * size,
                    "size": size,
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })
                in_trade = False

        if not in_trade and signal.iloc[i]:
            entry_idx = i
            entry_price = close[i]
            stop_price = preds["RangeLow"].iloc[i]
            if np.isnan(stop_price) or stop_price >= entry_price:
                stop_price = entry_price * (1 - p.fallback_stop_pct)

            p_rally = preds["P_RALLY"].iloc[i]
            atr_pct = preds["ATR_pct"].iloc[i]
            raw_size = cfg.vol_k * (p_rally - cfg.p_rally) / atr_pct if atr_pct > 0 else 0
            size = max(0.0, min(raw_size, cfg.max_risk))
            if size >= 0.01:
                in_trade = True
                bars_held = 0
                highest_close = entry_price
                trailing_stop = entry_price - p.trailing_stop_atr_mult * atr[i]

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def simulate_portfolio(all_trades: pd.DataFrame, cfg: TradingConfig,
                       initial_capital: float = 100_000) -> dict:
    """Simulate portfolio with leverage and cash yield on idle capital."""
    if all_trades.empty:
        return {"cagr": 0, "max_dd": 0, "sharpe": 0, "total_return": 0,
                "n_trades": 0, "win_rate": 0, "pf": 0, "n_years": 0}

    trades = all_trades.sort_values("entry_date").reset_index(drop=True)
    min_date = trades["entry_date"].min()
    max_date = trades["exit_date"].max()
    dates = pd.date_range(min_date, max_date, freq="B")

    equity = initial_capital
    equity_series = pd.Series(np.nan, index=dates)
    open_positions = []
    daily_pnl = pd.Series(0.0, index=dates)
    trade_queue = list(trades.itertuples(index=False))
    trade_idx = 0
    daily_cash_yield = (1 + cfg.cash_yield) ** (1 / 252) - 1

    for date in dates:
        # Close positions
        still_open = []
        for pos in open_positions:
            if date >= pos["exit_date"]:
                sized_pnl = pos["pnl_pct"] * pos["allocated"] * cfg.leverage
                equity += sized_pnl
                daily_pnl[date] += sized_pnl
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open new positions
        total_allocated = sum(p["allocated"] for p in open_positions)
        exposure = total_allocated / equity if equity > 0 else 0
        while trade_idx < len(trade_queue):
            t = trade_queue[trade_idx]
            if t.entry_date > date:
                break
            if t.entry_date == date:
                alloc = t.size * equity
                new_exposure = exposure + alloc / equity
                if new_exposure <= 1.0:
                    open_positions.append({
                        "entry_date": t.entry_date,
                        "exit_date": t.exit_date,
                        "pnl_pct": t.pnl_pct,
                        "allocated": alloc,
                    })
                    exposure = new_exposure
            trade_idx += 1

        # Cash yield on idle capital
        idle = max(0, equity - total_allocated)
        cash_income = idle * daily_cash_yield
        equity += cash_income
        daily_pnl[date] += cash_income

        equity_series[date] = equity

    equity_series = equity_series.ffill().bfill()
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak

    total_return = equity_series.iloc[-1] / initial_capital - 1
    total_days = (dates[-1] - dates[0]).days
    n_years = total_days / 365.25
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    max_dd = drawdown.min()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    sized_pnl = trades["pnl_sized"] * cfg.leverage
    gross_profit = sized_pnl[sized_pnl > 0].sum()
    gross_loss = -sized_pnl[sized_pnl < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_return": total_return,
        "n_trades": len(trades),
        "n_years": n_years,
        "win_rate": (trades["pnl_pct"] > 0).mean(),
        "pf": pf,
        "equity_series": equity_series,
        "drawdown": drawdown,
    }


# ---------------------------------------------------------------------------
# Walk-forward training (used by universe_bt.py)
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    coefs: dict
    intercept: float
    predictions: pd.DataFrame  # index=date, columns=[P_RALLY, RALLY_ST]


def walk_forward_train(df: pd.DataFrame) -> list[FoldResult]:
    """
    Walk-forward: train on `train_years`, test on `test_years`, roll forward.
    Fits HMM on training data per fold, uses state probs as features.
    Returns list of FoldResult (one per fold).
    """
    p = PARAMS
    train_yrs = p.walk_forward_train_years
    test_yrs = p.walk_forward_test_years

    target_col = "RALLY_ST"

    years = df.index.year.unique().sort_values()
    min_year = int(years.min())
    max_year = int(years.max())

    results: list[FoldResult] = []

    fold_start = min_year
    while fold_start + train_yrs + test_yrs - 1 <= max_year:
        train_end_year = fold_start + train_yrs - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + test_yrs - 1

        train_mask = (df.index.year >= fold_start) & (df.index.year <= train_end_year)
        test_mask = (df.index.year >= test_start_year) & (df.index.year <= test_end_year)

        df_train_raw = df.loc[train_mask]
        df_test_raw = df.loc[test_mask]

        if len(df_train_raw) < 100 or len(df_test_raw) < 20:
            fold_start += test_yrs
            continue

        # Fit HMM on training data only
        hmm_model, hmm_scaler, state_order = fit_hmm(df_train_raw)

        # Predict HMM state probabilities for train and test
        hmm_train = predict_hmm_probs(hmm_model, hmm_scaler, state_order, df_train_raw)
        hmm_test = predict_hmm_probs(hmm_model, hmm_scaler, state_order, df_test_raw)

        # Merge HMM features
        df_train_full = df_train_raw.join(hmm_train)
        df_test_full = df_test_raw.join(hmm_test)

        # Determine which features are available
        feature_cols = ALL_FEATURE_COLS

        # Drop rows with NaN in features or label
        train_valid = df_train_full[feature_cols + [target_col]].notna().all(axis=1)
        test_valid = df_test_full[feature_cols + [target_col]].notna().all(axis=1)
        df_train = df_train_full.loc[train_valid]
        df_test = df_test_full.loc[test_valid]

        if len(df_train) < 100 or len(df_test) < 20:
            fold_start += test_yrs
            continue

        X_train = df_train[feature_cols].values
        y_train = df_train[target_col].values
        X_test = df_test[feature_cols].values
        y_test = df_test[target_col].values

        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Fit logistic regression
        lr = LogisticRegression(
            C=1.0,
            l1_ratio=0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
        lr.fit(X_train_s, y_train)

        # Isotonic calibration (fit on last 20% of training set as validation)
        val_split = int(len(X_train_s) * 0.8)
        X_val_s = X_train_s[val_split:]
        y_val = y_train[val_split:]
        raw_val_probs = lr.predict_proba(X_val_s)[:, 1]

        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(raw_val_probs, y_val)

        # Calibrated probabilities on test set
        raw_test_probs = lr.predict_proba(X_test_s)[:, 1]
        cal_test_probs = iso.predict(raw_test_probs)

        preds = pd.DataFrame({
            "P_RALLY": cal_test_probs,
            "P_RALLY_RAW": raw_test_probs,
            "RALLY_ST": y_test,
        }, index=df_test.index)

        # Attach features for trading rules
        for col in feature_cols:
            preds[col] = df_test[col].values
        # Attach price data for trading
        for col in ["Open", "High", "Low", "Close", "ATR", "ATR_pct", "RV",
                     "p_RV", "RangeHigh", "RangeLow", "MA200", "RSI",
                     "VIX_Close"]:
            if col in df_test.columns:
                preds[col] = df_test[col].values

        coefs = dict(zip(feature_cols, lr.coef_[0] / scaler.scale_))
        intercept = float(lr.intercept_[0])

        results.append(FoldResult(
            train_start=str(fold_start),
            train_end=str(train_end_year),
            test_start=str(test_start_year),
            test_end=str(test_end_year),
            coefs=coefs,
            intercept=intercept,
            predictions=preds,
        ))

        fold_start += test_yrs

    return results


def combine_predictions(folds: list[FoldResult]) -> pd.DataFrame:
    """Concatenate all out-of-sample predictions across folds."""
    return pd.concat([f.predictions for f in folds], axis=0).sort_index()
