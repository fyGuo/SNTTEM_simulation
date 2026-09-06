"""ML-based working models with cross-fitting (SuperLearner equivalent).

Cross-fitting strategy: train on id_1, predict for id_2, and vice versa.
SuperLearner is replaced by a simple ensemble of RF, GradientBoosting, and
Logistic/Linear regression, averaged with equal weights, for propensity
scores (ps0..ps4).

The outcome regression (mu05..mu45) is fit on the FULL fold with A_t as an
explicit covariate, rather than only the A_t==A_{t-1} subset (~ph_t fraction
of the fold) -- that subsetting was starving the mu-fit for the sparse
constant-treatment-path corner (ph=0.3, p_I=0.2) and producing catastrophic
Robins'-estimator outliers there (see CLAUDE.md). Prediction plugs in the
counterfactual A_t=A_{t-1} value -- an interpolation within A_t's observed
{0,1} support, not extrapolation, since A_t is binary and both values appear
in the full training fold. This variant still uses the full RF+GBM+poly
ensemble (_classify/_regress) for mu, same as for ps0..ps4 -- isolating the
full-data/plug-in change from the separate RF-only simplification
(_classify_rf/_regress_rf, currently unused but kept for comparison).
"""
import numpy as np
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


def _classify_rf(X_train, y_train, X_pred):
    """RF-only classification -- alternative to _classify() for the outcome
    regression, currently unused (this variant uses the full ensemble
    instead -- see module docstring), kept for comparison.
    """
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    model.fit(X_train, y_train)
    return model.predict_proba(X_pred)[:, 1]


def _regress_rf(X_train, y_train, X_pred):
    """RF-only regression -- alternative to _regress(), currently unused. See
    _classify_rf() docstring.
    """
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    model.fit(X_train, y_train)
    return model.predict(X_pred)


def _classify_true(X_train, y_train, X_pred):
    """Plain logistic regression -- the correctly-specified working model.

    Used only for Three-step-ipw's nuisance fits (see CLAUDE.md): same
    covariate sets as working_model()'s RF/GBM/poly ensemble, but fit with a
    single parametric logistic regression instead.
    """
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model.predict_proba(X_pred)[:, 1]


def _regress_true(X_train, y_train, X_pred):
    """Plain linear regression -- the correctly-specified working model.

    Used only for Three-step-ipw's nuisance fits (see CLAUDE.md): same
    covariate sets as working_model()'s RF/GBM/poly ensemble, but fit with a
    single parametric linear regression instead.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_pred)


def working_model_true(
    data,
    id_1,
    id_2,
    ps0=True,
    ps1=True,
    ps2=True,
    ps3=True,
    ps4=True,
    mu05=True,
    mu15=True,
    mu25=True,
    mu35=True,
    mu45=True,
):
    """Fit Three-step-ipw's nuisance models with plain parametric regressions.

    Mirrors working_model() exactly -- same cross-fitting, same covariate
    sets, same recursive doubly-robust pseudo-outcome chain -- except every
    RF/GBM/poly ensemble call is replaced by a single logistic regression
    (propensity scores and mu45, since Y is binary) or linear regression
    (mu05/mu15/mu25/mu35, continuous pseudo-outcomes). Used only to build the
    nuisances fed to Three-step-ipw; Three-step-g and Robins' still use
    working_model()'s RF ensemble.

    Args:
        data: Full dataset (pandas DataFrame).
        id_1, id_2: Integer index arrays for the two cross-fitting folds.
        ps0/ps1/ps2/ps3/ps4: Whether to fit propensity score models (bool).
        mu05/mu15/mu25/mu35/mu45: Whether to fit outcome regression models (bool).

    Returns:
        data with columns ps0..ps4, mu45, dr_mu45, mu35, dr_mu35, mu25, dr_mu25,
        mu15, dr_mu15, mu05 added.
    """
    data = data.copy()
    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Propensity score at time 0: P(A0=1 | L0) ---
    if ps0:
        X1, X2 = d1[["L0"]].values, d2[["L0"]].values
        data.loc[data.index[id_2], "ps0"] = _classify_true(X1, d1["A0"].values, X2)
        data.loc[data.index[id_1], "ps0"] = _classify_true(X2, d2["A0"].values, X1)
    else:
        data["ps0"] = data["A0"].mean()

    # --- Propensity score at time 1: P(A1=1 | L1, A0) ---
    if ps1:
        X1 = d1[["L1", "A0"]].values
        X2 = d2[["L1", "A0"]].values
        data.loc[data.index[id_2], "ps1"] = _classify_true(X1, d1["A1"].values, X2)
        data.loc[data.index[id_1], "ps1"] = _classify_true(X2, d2["A1"].values, X1)
    else:
        data["ps1"] = data["A1"].mean()

    # --- Propensity score at time 2: P(A2=1 | L2, A1) ---
    if ps2:
        X1 = d1[["L2", "A1"]].values
        X2 = d2[["L2", "A1"]].values
        data.loc[data.index[id_2], "ps2"] = _classify_true(X1, d1["A2"].values, X2)
        data.loc[data.index[id_1], "ps2"] = _classify_true(X2, d2["A2"].values, X1)
    else:
        data["ps2"] = data["A2"].mean()

    # --- Propensity score at time 3: P(A3=1 | L3, A2) ---
    if ps3:
        X1 = d1[["L3", "A2"]].values
        X2 = d2[["L3", "A2"]].values
        data.loc[data.index[id_2], "ps3"] = _classify_true(X1, d1["A3"].values, X2)
        data.loc[data.index[id_1], "ps3"] = _classify_true(X2, d2["A3"].values, X1)
    else:
        data["ps3"] = data["A3"].mean()

    # --- Propensity score at time 4: P(A4=1 | L4, A3) ---
    if ps4:
        X1 = d1[["L4", "A3"]].values
        X2 = d2[["L4", "A3"]].values
        data.loc[data.index[id_2], "ps4"] = _classify_true(X1, d1["A4"].values, X2)
        data.loc[data.index[id_1], "ps4"] = _classify_true(X2, d2["A4"].values, X1)
    else:
        data["ps4"] = data["A4"].mean()

    # --- Outcome regression mu45: E[Y | L4,A3,L3,A2,L2,A1,L1,A0,L0, A4==A3] ---
    if mu45:
        mask1 = d1["A4"].values == d1["A3"].values
        mask2 = d2["A4"].values == d2["A3"].values
        cols = ["L4", "A3", "L3", "A2", "L2", "A1", "L1", "A0", "L0"]
        X1_tr = d1.loc[mask1, cols].values
        y1_tr = d1.loc[mask1, "Y"].values
        X2_tr = d2.loc[mask2, cols].values
        y2_tr = d2.loc[mask2, "Y"].values
        data.loc[data.index[id_2], "mu45"] = _classify_true(X1_tr, y1_tr, d2[cols].values)
        data.loc[data.index[id_1], "mu45"] = _classify_true(X2_tr, y2_tr, d1[cols].values)
    else:
        data["mu45"] = data["Y"].mean()

    # --- Doubly-robust pseudo-outcome for mu35 ---
    for fold_idx in [id_1, id_2]:
        idx = data.index[fold_idx]
        ps4_f = data.loc[idx, "ps4"].values
        A3_f = data.loc[idx, "A3"].values
        A4_f = data.loc[idx, "A4"].values
        Y_f = data.loc[idx, "Y"].values
        mu45_f = data.loc[idx, "mu45"].values
        weight4 = (A4_f == A3_f).astype(float) / (
            (1 - ps4_f) * (1 - A3_f) + ps4_f * A3_f
        )
        data.loc[idx, "dr_mu45"] = weight4 * (Y_f - mu45_f) + mu45_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu35: E[dr_mu45 | L3,A2,L2,A1,L1,A0,L0, A3==A2] ---
    if mu35:
        mask1 = d1["A3"].values == d1["A2"].values
        mask2 = d2["A3"].values == d2["A2"].values
        cols = ["L3", "A2", "L2", "A1", "L1", "A0", "L0"]
        X1_tr = d1.loc[mask1, cols].values
        y1_tr = d1.loc[mask1, "dr_mu45"].values
        X2_tr = d2.loc[mask2, cols].values
        y2_tr = d2.loc[mask2, "dr_mu45"].values
        data.loc[data.index[id_2], "mu35"] = _regress_true(X1_tr, y1_tr, d2[cols].values)
        data.loc[data.index[id_1], "mu35"] = _regress_true(X2_tr, y2_tr, d1[cols].values)
    else:
        data["mu35"] = data["dr_mu45"].mean()

    # --- Doubly-robust pseudo-outcome for mu25 ---
    for fold_idx in [id_1, id_2]:
        idx = data.index[fold_idx]
        ps3_f = data.loc[idx, "ps3"].values
        A2_f = data.loc[idx, "A2"].values
        A3_f = data.loc[idx, "A3"].values
        dr_mu45_f = data.loc[idx, "dr_mu45"].values
        mu35_f = data.loc[idx, "mu35"].values
        weight3 = (A3_f == A2_f).astype(float) / (
            (1 - ps3_f) * (1 - A2_f) + ps3_f * A2_f
        )
        data.loc[idx, "dr_mu35"] = weight3 * (dr_mu45_f - mu35_f) + mu35_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu25: E[dr_mu35 | L2,A1,L1,A0,L0, A2==A1] ---
    if mu25:
        mask1 = d1["A2"].values == d1["A1"].values
        mask2 = d2["A2"].values == d2["A1"].values
        cols = ["L2", "A1", "L1", "A0", "L0"]
        X1_tr = d1.loc[mask1, cols].values
        y1_tr = d1.loc[mask1, "dr_mu35"].values
        X2_tr = d2.loc[mask2, cols].values
        y2_tr = d2.loc[mask2, "dr_mu35"].values
        data.loc[data.index[id_2], "mu25"] = _regress_true(X1_tr, y1_tr, d2[cols].values)
        data.loc[data.index[id_1], "mu25"] = _regress_true(X2_tr, y2_tr, d1[cols].values)
    else:
        data["mu25"] = data["dr_mu35"].mean()

    # --- Doubly-robust pseudo-outcome for mu15 ---
    for fold_idx in [id_1, id_2]:
        idx = data.index[fold_idx]
        ps2_f = data.loc[idx, "ps2"].values
        A1_f = data.loc[idx, "A1"].values
        A2_f = data.loc[idx, "A2"].values
        dr_mu35_f = data.loc[idx, "dr_mu35"].values
        mu25_f = data.loc[idx, "mu25"].values
        weight2 = (A2_f == A1_f).astype(float) / (
            (1 - ps2_f) * (1 - A1_f) + ps2_f * A1_f
        )
        data.loc[idx, "dr_mu25"] = weight2 * (dr_mu35_f - mu25_f) + mu25_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu15: E[dr_mu25 | L1,A0,L0, A1==A0] ---
    if mu15:
        mask1 = d1["A1"].values == d1["A0"].values
        mask2 = d2["A1"].values == d2["A0"].values
        cols = ["L1", "A0", "L0"]
        X1_tr = d1.loc[mask1, cols].values
        y1_tr = d1.loc[mask1, "dr_mu25"].values
        X2_tr = d2.loc[mask2, cols].values
        y2_tr = d2.loc[mask2, "dr_mu25"].values
        data.loc[data.index[id_2], "mu15"] = _regress_true(X1_tr, y1_tr, d2[cols].values)
        data.loc[data.index[id_1], "mu15"] = _regress_true(X2_tr, y2_tr, d1[cols].values)
    else:
        data["mu15"] = data["dr_mu25"].mean()

    # --- Doubly-robust pseudo-outcome for mu05 ---
    for fold_idx in [id_1, id_2]:
        idx = data.index[fold_idx]
        ps1_f = data.loc[idx, "ps1"].values
        A0_f = data.loc[idx, "A0"].values
        A1_f = data.loc[idx, "A1"].values
        dr_mu25_f = data.loc[idx, "dr_mu25"].values
        mu15_f = data.loc[idx, "mu15"].values
        weight1 = (A1_f == A0_f).astype(float) / (
            (1 - ps1_f) * (1 - A0_f) + ps1_f * A0_f
        )
        data.loc[idx, "dr_mu15"] = weight1 * (dr_mu25_f - mu15_f) + mu15_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu05: E[dr_mu15 | L0, A0==0] ---
    if mu05:
        mask1 = d1["A0"].values == 0
        mask2 = d2["A0"].values == 0
        X1_tr = d1.loc[mask1, ["L0"]].values
        y1_tr = d1.loc[mask1, "dr_mu15"].values
        X2_tr = d2.loc[mask2, ["L0"]].values
        y2_tr = d2.loc[mask2, "dr_mu15"].values
        data.loc[data.index[id_2], "mu05"] = _regress_true(X1_tr, y1_tr, d2[["L0"]].values)
        data.loc[data.index[id_1], "mu05"] = _regress_true(X2_tr, y2_tr, d1[["L0"]].values)
    else:
        data["mu05"] = data["dr_mu15"].mean()

    return data


def working_model(
    data,
    id_1,
    id_2,
    ps0=True,
    ps1=True,
    ps2=True,
    ps3=True,
    ps4=True,
    mu05=True,
    mu15=True,
    mu25=True,
    mu35=True,
    mu45=True,
):
    """Fit nuisance models using cross-fitting and ML ensembles.

    For each nuisance quantity, models are trained on id_1 and predict id_2,
    then trained on id_2 and predict id_1.

    Args:
        data: Full dataset (pandas DataFrame).
        id_1, id_2: Integer index arrays for the two cross-fitting folds.
        ps0/ps1/ps2/ps3/ps4: Whether to fit propensity score models (bool).
        mu05/mu15/mu25/mu35/mu45: Whether to fit outcome regression models (bool).

    Returns:
        data with columns ps0..ps4, mu45, dr_mu45, mu35, dr_mu35, mu25, dr_mu25,
        mu15, dr_mu15, mu05 added.
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

    # --- Propensity score at time 3: P(A3=1 | L3, A2) ---
    if ps3:
        X1 = d1[["L3", "A2"]].values
        X2 = d2[["L3", "A2"]].values
        data.loc[data.index[id_2], "ps3"] = _classify(X1, d1["A3"].values, X2)
        data.loc[data.index[id_1], "ps3"] = _classify(X2, d2["A3"].values, X1)
    else:
        data["ps3"] = data["A3"].mean()

    # --- Propensity score at time 4: P(A4=1 | L4, A3) ---
    if ps4:
        X1 = d1[["L4", "A3"]].values
        X2 = d2[["L4", "A3"]].values
        data.loc[data.index[id_2], "ps4"] = _classify(X1, d1["A4"].values, X2)
        data.loc[data.index[id_1], "ps4"] = _classify(X2, d2["A4"].values, X1)
    else:
        data["ps4"] = data["A4"].mean()

    # --- Outcome regression mu45: E[Y | L4,A4,A3,L3,A2,L2,A1,L1,A0,L0], A4:=A3 plug-in ---
    # Trained on the FULL fold with A4 as an explicit covariate (not just the
    # A4==A3 subset) -- at ph=0.3 that subset is only ~30% of the fold, which
    # starved the mu-fit for the constant-treatment-path corner and produced
    # catastrophic Robins'-estimator outliers there (see CLAUDE.md). Since A4
    # is binary, plugging in the counterfactual A4=A3 at predict time is an
    # interpolation within A4's observed support, not an out-of-range
    # extrapolation.
    if mu45:
        cols = ["L4", "A4", "A3", "L3", "A2", "L2", "A1", "L1", "A0", "L0"]
        X1_tr = d1[cols].values
        y1_tr = d1["Y"].values
        X2_tr = d2[cols].values
        y2_tr = d2["Y"].values
        d1_cf = d1[cols].copy()
        d1_cf["A4"] = d1_cf["A3"]
        d2_cf = d2[cols].copy()
        d2_cf["A4"] = d2_cf["A3"]
        data.loc[data.index[id_2], "mu45"] = _classify(X1_tr, y1_tr, d2_cf[cols].values)
        data.loc[data.index[id_1], "mu45"] = _classify(X2_tr, y2_tr, d1_cf[cols].values)
    else:
        data["mu45"] = data["Y"].mean()

    # --- Doubly-robust pseudo-outcome for mu35 ---
    for fold_idx in [id_1, id_2]:
        idx = data.index[fold_idx]
        ps4_f = data.loc[idx, "ps4"].values
        A3_f = data.loc[idx, "A3"].values
        A4_f = data.loc[idx, "A4"].values
        Y_f = data.loc[idx, "Y"].values
        mu45_f = data.loc[idx, "mu45"].values
        weight4 = (A4_f == A3_f).astype(float) / (
            (1 - ps4_f) * (1 - A3_f) + ps4_f * A3_f
        )
        data.loc[idx, "dr_mu45"] = weight4 * (Y_f - mu45_f) + mu45_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu35: E[dr_mu45 | L3,A3,A2,L2,A1,L1,A0,L0], A3:=A2 plug-in ---
    if mu35:
        cols = ["L3", "A3", "A2", "L2", "A1", "L1", "A0", "L0"]
        X1_tr = d1[cols].values
        y1_tr = d1["dr_mu45"].values
        X2_tr = d2[cols].values
        y2_tr = d2["dr_mu45"].values
        d1_cf = d1[cols].copy()
        d1_cf["A3"] = d1_cf["A2"]
        d2_cf = d2[cols].copy()
        d2_cf["A3"] = d2_cf["A2"]
        data.loc[data.index[id_2], "mu35"] = _regress(X1_tr, y1_tr, d2_cf[cols].values)
        data.loc[data.index[id_1], "mu35"] = _regress(X2_tr, y2_tr, d1_cf[cols].values)
    else:
        data["mu35"] = data["dr_mu45"].mean()

    # --- Doubly-robust pseudo-outcome for mu25 ---
    for fold_idx in [id_1, id_2]:
        idx = data.index[fold_idx]
        ps3_f = data.loc[idx, "ps3"].values
        A2_f = data.loc[idx, "A2"].values
        A3_f = data.loc[idx, "A3"].values
        dr_mu45_f = data.loc[idx, "dr_mu45"].values
        mu35_f = data.loc[idx, "mu35"].values
        weight3 = (A3_f == A2_f).astype(float) / (
            (1 - ps3_f) * (1 - A2_f) + ps3_f * A2_f
        )
        data.loc[idx, "dr_mu35"] = weight3 * (dr_mu45_f - mu35_f) + mu35_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu25: E[dr_mu35 | L2,A2,A1,L1,A0,L0], A2:=A1 plug-in ---
    if mu25:
        cols = ["L2", "A2", "A1", "L1", "A0", "L0"]
        X1_tr = d1[cols].values
        y1_tr = d1["dr_mu35"].values
        X2_tr = d2[cols].values
        y2_tr = d2["dr_mu35"].values
        d1_cf = d1[cols].copy()
        d1_cf["A2"] = d1_cf["A1"]
        d2_cf = d2[cols].copy()
        d2_cf["A2"] = d2_cf["A1"]
        data.loc[data.index[id_2], "mu25"] = _regress(X1_tr, y1_tr, d2_cf[cols].values)
        data.loc[data.index[id_1], "mu25"] = _regress(X2_tr, y2_tr, d1_cf[cols].values)
    else:
        data["mu25"] = data["dr_mu35"].mean()

    # --- Doubly-robust pseudo-outcome for mu15 ---
    for fold_idx in [id_1, id_2]:
        idx = data.index[fold_idx]
        ps2_f = data.loc[idx, "ps2"].values
        A1_f = data.loc[idx, "A1"].values
        A2_f = data.loc[idx, "A2"].values
        dr_mu35_f = data.loc[idx, "dr_mu35"].values
        mu25_f = data.loc[idx, "mu25"].values
        weight2 = (A2_f == A1_f).astype(float) / (
            (1 - ps2_f) * (1 - A1_f) + ps2_f * A1_f
        )
        data.loc[idx, "dr_mu25"] = weight2 * (dr_mu35_f - mu25_f) + mu25_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu15: E[dr_mu25 | L1,A1,A0,L0], A1:=A0 plug-in ---
    if mu15:
        cols = ["L1", "A1", "A0", "L0"]
        X1_tr = d1[cols].values
        y1_tr = d1["dr_mu25"].values
        X2_tr = d2[cols].values
        y2_tr = d2["dr_mu25"].values
        d1_cf = d1[cols].copy()
        d1_cf["A1"] = d1_cf["A0"]
        d2_cf = d2[cols].copy()
        d2_cf["A1"] = d2_cf["A0"]
        data.loc[data.index[id_2], "mu15"] = _regress(X1_tr, y1_tr, d2_cf[cols].values)
        data.loc[data.index[id_1], "mu15"] = _regress(X2_tr, y2_tr, d1_cf[cols].values)
    else:
        data["mu15"] = data["dr_mu25"].mean()

    # --- Doubly-robust pseudo-outcome for mu05 ---
    for fold_idx in [id_1, id_2]:
        idx = data.index[fold_idx]
        ps1_f = data.loc[idx, "ps1"].values
        A0_f = data.loc[idx, "A0"].values
        A1_f = data.loc[idx, "A1"].values
        dr_mu25_f = data.loc[idx, "dr_mu25"].values
        mu15_f = data.loc[idx, "mu15"].values
        weight1 = (A1_f == A0_f).astype(float) / (
            (1 - ps1_f) * (1 - A0_f) + ps1_f * A0_f
        )
        data.loc[idx, "dr_mu15"] = weight1 * (dr_mu25_f - mu15_f) + mu15_f

    d1 = data.iloc[id_1].copy()
    d2 = data.iloc[id_2].copy()

    # --- Outcome regression mu05: E[dr_mu15 | L0,A0], A0:=0 plug-in (A_{-1} := 0 convention) ---
    if mu05:
        cols = ["L0", "A0"]
        X1_tr = d1[cols].values
        y1_tr = d1["dr_mu15"].values
        X2_tr = d2[cols].values
        y2_tr = d2["dr_mu15"].values
        d1_cf = d1[cols].copy()
        d1_cf["A0"] = 0.0
        d2_cf = d2[cols].copy()
        d2_cf["A0"] = 0.0
        data.loc[data.index[id_2], "mu05"] = _regress(X1_tr, y1_tr, d2_cf[cols].values)
        data.loc[data.index[id_1], "mu05"] = _regress(X2_tr, y2_tr, d1_cf[cols].values)
    else:
        data["mu05"] = data["dr_mu15"].mean()

    return data
