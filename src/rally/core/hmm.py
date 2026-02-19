"""
Hidden Markov Model for volatility regime detection.

Fits a 3-state Gaussian HMM on volatility features to identify:
  State 0: Compressed (low vol)
  State 1: Normal
  State 2: Expanding (high vol)

States are labeled post-hoc by sorting on mean RV emission.
"""

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from ..config import PIPELINE

HMM_FEATURES = ["RV", "ATR_pct", "BB_width"]
N_STATES = 3


def fit_hmm(df_train: pd.DataFrame) -> tuple[GaussianHMM, StandardScaler, np.ndarray]:
    """
    Fit a 3-state Gaussian HMM on volatility features from training data.
    Returns (model, scaler, state_order) where state_order maps
    internal states to [compressed=0, normal=1, expanding=2].
    """
    cols = [c for c in HMM_FEATURES if c in df_train.columns]
    X = df_train[cols].dropna().values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=PIPELINE.hmm_n_iter,
            tol=PIPELINE.hmm_tol,
            random_state=42,
            verbose=False,
        )
        model.fit(X_s)

    # Label states by mean RV (first feature) — lowest RV = compressed
    rv_means = model.means_[:, 0]  # RV is first column
    state_order = np.argsort(rv_means)  # [compressed, normal, expanding]

    return model, scaler, state_order


def predict_hmm_probs(
    model: GaussianHMM | None, scaler: StandardScaler | None,
    state_order: np.ndarray | None, df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute filtered state probabilities for each bar.
    Returns DataFrame with columns: P_compressed, P_normal, P_expanding.
    If model is None, returns zeros (graceful degradation).
    """
    if model is None:
        return pd.DataFrame(
            0.0, index=df.index,
            columns=["P_compressed", "P_normal", "P_expanding", "HMM_transition_signal"],
        )

    cols = [c for c in HMM_FEATURES if c in df.columns]
    mask = df[cols].notna().all(axis=1)
    X = df.loc[mask, cols].values
    X_s = scaler.transform(X)

    # Forward algorithm gives filtered probabilities
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        posteriors = model.predict_proba(X_s)

    # Reorder columns: compressed, normal, expanding
    posteriors_ordered = posteriors[:, state_order]

    result = pd.DataFrame(
        np.nan,
        index=df.index,
        columns=["P_compressed", "P_normal", "P_expanding"],
    )
    result.loc[mask, "P_compressed"] = posteriors_ordered[:, 0]
    result.loc[mask, "P_normal"] = posteriors_ordered[:, 1]
    result.loc[mask, "P_expanding"] = posteriors_ordered[:, 2]

    # Transition probability: P(expanding next | compressed now)
    trans = model.transmat_
    compressed_idx = state_order[0]
    expanding_idx = state_order[2]
    p_trans = trans[compressed_idx, expanding_idx]

    # Weighted transition signal: P(compressed) * P(compressed->expanding)
    result["HMM_transition_signal"] = result["P_compressed"] * p_trans

    return result
