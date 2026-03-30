"""Shared model-fitting pipeline: HMM → logistic regression → isotonic calibration.

Used by both pipeline/retrain.py (train the latest fold) and
backtest/common.py (walk-forward folds). Any change to the model
architecture — regularisation, solver, calibration strategy — happens here once.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from core.hmm import fit_hmm, predict_hmm_probs

logger = logging.getLogger(__name__)


def fit_model(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "RALLY_ST",
    min_train: int = 100,
    min_positives: int = 3,
) -> dict | None:
    """Fit HMM + logistic regression + isotonic calibration on training data.

    Returns an artifacts dict ready for save_model() or direct inference, or
    None if the data is insufficient (too few rows or too few positive labels).

    HMM failure is handled gracefully — if fit_hmm raises (convergence or
    singular covariance), training continues without HMM features.
    """
    # Fit HMM; fall back to None on convergence / singular-covariance errors
    try:
        hmm_model, hmm_scaler, state_order = fit_hmm(df_train)
    except (ValueError, np.linalg.LinAlgError):
        hmm_model, hmm_scaler, state_order = None, None, None

    hmm_probs = predict_hmm_probs(hmm_model, hmm_scaler, state_order, df_train)
    df_full = df_train.join(hmm_probs)

    # Drop rows where any feature or the target label is NaN
    valid = df_full[feature_cols + [target_col]].notna().all(axis=1)
    df = df_full.loc[valid]

    if len(df) < min_train:
        return None

    X = df[feature_cols].values
    y = df[target_col].values

    if y.sum() < min_positives:
        return None

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lr = LogisticRegression(
        C=1.0, l1_ratio=0, solver="lbfgs",
        max_iter=1000, class_weight="balanced",
    )
    lr.fit(X_s, y)

    # Isotonic calibration on the last 20% of training data as a held-out
    # validation set — corrects probability ranking without leaking test data
    val_split = int(len(X_s) * 0.8)
    X_val_s = X_s[val_split:]
    y_val = y[val_split:]
    raw_val_probs = lr.predict_proba(X_val_s)[:, 1]

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    if y_val.sum() >= 3:
        iso.fit(raw_val_probs, y_val)
    else:
        # Not enough positives for calibration — identity mapping
        iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    return {
        "lr_model": lr,
        "lr_scaler": scaler,
        "iso_calibrator": iso,
        "hmm_model": hmm_model,
        "hmm_scaler": hmm_scaler,
        "state_order": state_order,
        "feature_cols": feature_cols,
        # Normalised coefficients — useful for fold-level diagnostics in backtest
        "coefs": dict(zip(feature_cols, lr.coef_[0] / scaler.scale_, strict=False)),
        "intercept": float(lr.intercept_[0]),
    }
