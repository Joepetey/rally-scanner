"""
Walk-forward logistic regression with isotonic calibration + HMM regime features.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import PARAMS
from .features import FEATURE_COLS
from .hmm import fit_hmm, predict_hmm_probs

# Extended feature set: original + HMM regime probabilities
HMM_FEATURE_COLS = ["P_compressed", "P_expanding", "HMM_transition_signal"]
ALL_FEATURE_COLS = FEATURE_COLS + HMM_FEATURE_COLS


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
                     "p_RV", "RangeHigh", "RangeLow", "MA200", "RSI"]:
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
