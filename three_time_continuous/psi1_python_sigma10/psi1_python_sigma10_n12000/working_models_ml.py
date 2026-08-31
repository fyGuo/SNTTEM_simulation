"""ML-based working models with cross-fitting (SuperLearner equivalent).

Cross-fitting strategy: train on id_1, predict for id_2, and vice versa.
SuperLearner is replaced by a simple ensemble of RF, GradientBoosting, and
Logistic/Linear regression, averaged with equal weights.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def _classify(X_train, y_train, X_pred):
    """Ensemble classification: RF + GBM + logistic (GAM-like), equal-weight average."""
    models = [
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
        GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0
        ),
        make_pipeline(PolynomialFeatures(degree=3), LogisticRegression(max_iter=2000)),
    ]
    preds = []
    for m in models:
        m.fit(X_train, y_train)
        preds.append(m.predict_proba(X_pred)[:, 1])
    return np.mean(preds, axis=0)


def _regress(X_train, y_train, X_pred):
    """Ensemble regression: RF + GBM + polynomial OLS, equal-weight average."""
    models = [
        RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0),
        GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0
        ),
        make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
    ]
    preds = []
    for m in models:
        m.fit(X_train, y_train)
        preds.append(m.predict(X_pred))
    return np.mean(preds, axis=0)


def working_model(
    data,
    id_1,
    id_2,
    ps0=True,
    ps1=True,
    ps2=True,
    mu03=True,
    mu13=True,
    mu23=True,
):
    """Fit nuisance models using cross-fitting and ML ensembles.

    For each nuisance quantity, models are trained on id_1 and predict id_2,
    then trained on id_2 and predict id_1.

    Args:
        data: Full dataset (pandas DataFrame).
        id_1, id_2: Integer index arrays for the two cross-fitting folds.
        ps0/ps1/ps2: Whether to fit propensity score models (bool).
        mu03/mu13/mu23: Whether to fit outcome regression models (bool).

    Returns:
        data with columns ps0, ps1, ps2, mu23, dr_mu23, mu13, dr_mu13, mu03 added.
    """
    data = data.copy()
    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Propensity score at time 0: P(A0=1 | L0) ---
    if ps0:
        X1, X2 = d1[["L0"]].values, d2[["L0"]].values
        data.loc[data.index[id_2], "ps0"] = _classify(X1, d1["A0"].values, X2)
        data.loc[data.index[id_1], "ps0"] = _classify(X2, d2["A0"].values, X1)
    else:
        data["ps0"] = data["A0"].mean()

    # --- Propensity score at time 1: P(A1=1 | L1, A0) ---
    if ps1:
        X1 = d1[["L1", "A0"]].values
        X2 = d2[["L1", "A0"]].values
        data.loc[data.index[id_2], "ps1"] = _classify(X1, d1["A1"].values, X2)
        data.loc[data.index[id_1], "ps1"] = _classify(X2, d2["A1"].values, X1)
    else:
        data["ps1"] = data["A1"].mean()

    # --- Propensity score at time 2: P(A2=1 | L2, A1) ---
    if ps2:
        X1 = d1[["L2", "A1"]].values
        X2 = d2[["L2", "A1"]].values
        data.loc[data.index[id_2], "ps2"] = _classify(X1, d1["A2"].values, X2)
        data.loc[data.index[id_1], "ps2"] = _classify(X2, d2["A2"].values, X1)
    else:
        data["ps2"] = data["A2"].mean()

    # --- Outcome regression mu23: E[Y | L2,A1,L1,A0,L0, A2==A1] ---
    if mu23:
        mask1 = d1["A2"].values == d1["A1"].values
        mask2 = d2["A2"].values == d2["A1"].values
        cols = ["L2", "A1", "L1", "A0", "L0"]
        X1_tr = d1.loc[mask1, cols].values
        y1_tr = d1.loc[mask1, "Y"].values
        X2_tr = d2.loc[mask2, cols].values
        y2_tr = d2.loc[mask2, "Y"].values
        data.loc[data.index[id_2], "mu23"] = _regress(X1_tr, y1_tr, d2[cols].values)
        data.loc[data.index[id_1], "mu23"] = _regress(X2_tr, y2_tr, d1[cols].values)
    else:
        data["mu23"] = data["Y"].mean()

    # --- Doubly-robust pseudo-outcome for mu13 ---
    for fold_idx, fold_name in [(id_1, "id_1"), (id_2, "id_2")]:
        idx = data.index[fold_idx]
        ps2_f = data.loc[idx, "ps2"].values
        A1_f = data.loc[idx, "A1"].values
        A2_f = data.loc[idx, "A2"].values
        Y_f = data.loc[idx, "Y"].values
        mu23_f = data.loc[idx, "mu23"].values
        weight2 = (A1_f == A2_f).astype(float) / (
            (1 - ps2_f) * (1 - A1_f) + ps2_f * A1_f
        )
        data.loc[idx, "dr_mu23"] = weight2 * (Y_f - mu23_f) + mu23_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu13: E[dr_mu23 | L1,A0,L0, A1==A0] ---
    if mu13:
        mask1 = d1["A1"].values == d1["A0"].values
        mask2 = d2["A1"].values == d2["A0"].values
        cols = ["L1", "A0", "L0"]
        X1_tr = d1.loc[mask1, cols].values
        y1_tr = d1.loc[mask1, "dr_mu23"].values
        X2_tr = d2.loc[mask2, cols].values
        y2_tr = d2.loc[mask2, "dr_mu23"].values
        data.loc[data.index[id_2], "mu13"] = _regress(X1_tr, y1_tr, d2[cols].values)
        data.loc[data.index[id_1], "mu13"] = _regress(X2_tr, y2_tr, d1[cols].values)
    else:
        data["mu13"] = data["dr_mu23"].mean()

    # --- Doubly-robust pseudo-outcome for mu03 ---
    for fold_idx in [id_1, id_2]:
        idx = data.index[fold_idx]
        ps1_f = data.loc[idx, "ps1"].values
        A0_f = data.loc[idx, "A0"].values
        A1_f = data.loc[idx, "A1"].values
        dr_mu23_f = data.loc[idx, "dr_mu23"].values
        mu13_f = data.loc[idx, "mu13"].values
        weight1 = (A0_f == A1_f).astype(float) / (
            (1 - ps1_f) * (1 - A1_f) + ps1_f * A1_f
        )
        data.loc[idx, "dr_mu13"] = weight1 * (dr_mu23_f - mu13_f) + mu13_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu03: E[dr_mu13 | L0, A0==0] ---
    if mu03:
        mask1 = d1["A0"].values == 0
        mask2 = d2["A0"].values == 0
        X1_tr = d1.loc[mask1, ["L0"]].values
        y1_tr = d1.loc[mask1, "dr_mu13"].values
        X2_tr = d2.loc[mask2, ["L0"]].values
        y2_tr = d2.loc[mask2, "dr_mu13"].values
        data.loc[data.index[id_2], "mu03"] = _regress(X1_tr, y1_tr, d2[["L0"]].values)
        data.loc[data.index[id_1], "mu03"] = _regress(X2_tr, y2_tr, d1[["L0"]].values)
    else:
        data["mu03"] = data["dr_mu13"].mean()

    return data
