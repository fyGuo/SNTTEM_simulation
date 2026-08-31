"""M-estimation and estimating equations for SNTTEM g-estimation.

Two estimators are implemented:
  - Three-step  ("three_step"): conventional sequential regression approach
  - Robins' g-estimator ("robins"): SNTTEM-based estimator

Both solve a system of three estimating equations for (psi03, psi13, psi23).

MISSPECIFICATION STUDY: gamma13 and gamma23 below are deliberately fitted with
the WRONG functional form. The DGP (generate_data.py) is unchanged and still
generates data under the true blips -- only these estimating equations use the
misspecified forms. gamma03 keeps its true form.
"""
import os
import numpy as np
from scipy.optimize import fsolve

# True blips come from the DGP module (gamma03 always used at its true form).
from generate_data import gamma03, gamma13 as gamma13_true, gamma23 as gamma23_true

# Additive misspecification constant, read at import time from the environment so
# each run (a fresh process, incl. loky workers that inherit the env) can set it:
#     GAMMA_ADD=1.5 python run_simulation.py
# The +GAMMA_ADD term is NOT multiplied by psi, so the fitted blip is nonzero
# (=GAMMA_ADD) even at psi = 0 and cannot be absorbed by rescaling psi.
GAMMA_ADD = float(os.environ.get("GAMMA_ADD", "3"))


def gamma13_wrong(L1, psi13):
    """MISSPECIFIED blip at time 1 (true form: psi13 * (1 + L1))."""
    return psi13 * (L1 + 10) / (L1 + 4) + GAMMA_ADD


def gamma23_wrong(L2, psi23):
    """MISSPECIFIED blip at time 2 (true form: psi23 * (L2 + 1))."""
    return psi23 * (L2 + 10) / (L2 + 4) + GAMMA_ADD


def _unpack(df):
    v = lambda col: df[col].values
    return (
        v("A0"), v("A1"), v("A2"),
        v("I1"), v("I2"),
        v("Y"),
        v("L0"), v("L1"), v("L2"),
        v("ps0"), v("ps1"), v("ps2"),
        v("mu23"), v("mu13"), v("mu03"),
    )


def ee_three_step(theta, df, g13=gamma13_true, g23=gamma23_true):
    """Conventional (three-step) estimating equations summed over observations.

    g13, g23 select the blip functional form used for fitting (true or wrong).
    """
    psi03, psi13, psi23 = theta
    A0, A1, A2, I1, I2, Y, L0, L1, L2, ps0, ps1, ps2, mu23, mu13, mu03 = _unpack(df)

    w2 = (A2 == A1).astype(float) / ((1 - ps2) * (1 - A1) + ps2 * A1)
    w1 = (A1 == A0).astype(float) / ((1 - ps1) * (1 - A0) + ps1 * A0)

    eq1 = np.sum(
        (A0 - ps0) * (
            w2 * w1 * (Y - mu23)
            + w1 * (mu23 - mu13)
            + mu13 * np.exp(-gamma03(L0, psi03) * A0)
            - mu03
        )
    )
    eq2 = np.sum(
        I1 * (A1 - ps1) * (
            w2 * (Y - mu23)
            + mu23 * np.exp(-g13(L1, psi13) * (A1 - A0))
            - mu13
        )
    )
    eq3 = np.sum(
        I2 * (A2 - ps2) * (Y * np.exp(-g23(L2, psi23) * (A2 - A1)) - mu23)
    )
    return np.array([eq1, eq2, eq3])


def ee_robins(theta, df, g13=gamma13_true, g23=gamma23_true):
    """SNTTEM Robins' g-estimating equations summed over observations.

    g13, g23 select the blip functional form used for fitting (true or wrong).
    """
    psi03, psi13, psi23 = theta
    A0, A1, A2, I1, I2, Y, L0, L1, L2, ps0, ps1, ps2, mu23, mu13, mu03 = _unpack(df)

    w2 = (A2 == A1).astype(float) / ((1 - ps2) * (1 - A1) + ps2 * A1)
    w1 = (A1 == A0).astype(float) / ((1 - ps1) * (1 - A0) + ps1 * A0)

    # Use indicator-based weights: w^{1-I} so ineligible subjects get weight 1
    w2_I = w2 ** (1 - I2)
    w1_I = w1 ** (1 - I1)

    eq1 = np.sum(
        (A0 - ps0) * (
            w2_I * w1_I * (Y * np.exp(-g23(L2, psi23) * (A2 - A1)) - mu23)
            + w1_I * (mu23 * np.exp(-g13(L1, psi13) * (A1 - A0)) - mu13)
            + mu13 * np.exp(-gamma03(L0, psi03) * A0)
            - mu03
        )
    )
    eq2 = np.sum(
        I1 * (A1 - ps1) * (
            w2_I * (Y * np.exp(-g23(L2, psi23) * (A2 - A1)) - mu23)
            + mu23 * np.exp(-g13(L1, psi13) * (A1 - A0))
            - mu13
        )
    )
    eq3 = np.sum(
        I2 * (A2 - ps2) * (Y * np.exp(-g23(L2, psi23) * (A2 - A1)) - mu23)
    )
    return np.array([eq1, eq2, eq3])


def m_estimate(ee_fn, df, start=(0.0, 0.0, 0.0),
               g13=gamma13_true, g23=gamma23_true):
    """Find the root of sum_i ee(theta, data_i) = 0 via scipy fsolve.

    g13, g23 are forwarded to ee_fn to choose the blip form (true or wrong).
    """
    sol = fsolve(ee_fn, x0=np.array(start, dtype=float), args=(df, g13, g23),
                 full_output=True)
    estimates = sol[0]
    return estimates


def gmm_combine(est_con, est_gest):
    """Compute the optimal GMM linear combination of two estimators.

    Args:
        est_con:  (n_iter, 3) array of Three-step estimates (used as pilot).
        est_gest: (n_iter, 3) array of Robins' estimates (used as pilot).

    Returns:
        (n_iter, 3) array of GMM-combined estimates using the same inputs.
    """
    stacked = np.hstack([est_con, est_gest])   # (n_iter, 6)
    cov_mat = np.cov(stacked, rowvar=False)    # (6, 6)
    inv_mat = np.linalg.pinv(cov_mat)

    A = np.vstack([np.eye(3), np.eye(3)])      # (6, 3)
    mat2 = A.T @ inv_mat @ A                   # (3, 3)
    invcov = np.linalg.pinv(mat2)
    weight = invcov @ A.T @ inv_mat            # (3, 6)

    combined = (weight @ stacked.T).T          # (n_iter, 3)
    return combined
