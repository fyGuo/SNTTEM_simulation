"""M-estimation and estimating equations for SNTTEM g-estimation.

Two estimators are implemented:
  - Three-step  ("three_step"): conventional sequential regression approach
  - Robins' g-estimator ("robins"): SNTTEM-based estimator

Both solve a system of three estimating equations for (psi03, psi13, psi23).
"""
import numpy as np
from scipy.optimize import fsolve

from generate_data import gamma03, gamma13, gamma23


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


def ee_three_step(theta, df):
    """Conventional (three-step) estimating equations summed over observations."""
    psi03, psi13, psi23 = theta
    A0, A1, A2, I1, I2, Y, L0, L1, L2, ps0, ps1, ps2, mu23, mu13, mu03 = _unpack(df)

    w2 = (A2 == A1).astype(float) / ((1 - ps2) * (1 - A1) + ps2 * A1)
    w1 = (A1 == A0).astype(float) / ((1 - ps1) * (1 - A0) + ps1 * A0)

    eq1 = np.sum(
        (A0 - ps0) * (
            w2 * w1 * (Y - mu23)
            + w1 * (mu23 - mu13)
            + mu13 -gamma03(L0, psi03) * A0
            - mu03
        )
    )
    eq2 = np.sum(
        I1 * (A1 - ps1) * (
            w2 * (Y - mu23)
            + mu23 -gamma13(L1, psi13) * (A1 - A0)
            - mu13
        )
    )
    eq3 = np.sum(
        I2 * (A2 - ps2) * (Y -gamma23(L2, psi23) * (A2 - A1) - mu23)
    )
    return np.array([eq1, eq2, eq3])


def ee_robins(theta, df):
    """SNTTEM Robins' g-estimating equations summed over observations."""
    psi03, psi13, psi23 = theta
    A0, A1, A2, I1, I2, Y, L0, L1, L2, ps0, ps1, ps2, mu23, mu13, mu03 = _unpack(df)

    w2 = (A2 == A1).astype(float) / ((1 - ps2) * (1 - A1) + ps2 * A1)
    w1 = (A1 == A0).astype(float) / ((1 - ps1) * (1 - A0) + ps1 * A0)

    # Use indicator-based weights: w^{1-I} so ineligible subjects get weight 1
    w2_I = w2 ** (1 - I2)
    w1_I = w1 ** (1 - I1)

    eq1 = np.sum(
        (A0 - ps0) * (
            w2_I * w1_I * (Y -gamma23(L2, psi23) * (A2 - A1) - mu23)
            + w1_I * (mu23 -gamma13(L1, psi13) * (A1 - A0) - mu13)
            + mu13-gamma03(L0, psi03) * A0
            - mu03
        )
    )
    eq2 = np.sum(
        I1 * (A1 - ps1) * (
            w2_I * (Y-gamma23(L2, psi23) * (A2 - A1) - mu23)
            + mu23 -gamma13(L1, psi13) * (A1 - A0)
            - mu13
        )
    )
    eq3 = np.sum(
        I2 * (A2 - ps2) * (Y -gamma23(L2, psi23) * (A2 - A1) - mu23)
    )
    return np.array([eq1, eq2, eq3])


def m_estimate(ee_fn, df, start=(0.0, 0.0, 0.0)):
    """Find the root of sum_i ee(theta, data_i) = 0 via scipy fsolve."""
    sol = fsolve(ee_fn, x0=np.array(start, dtype=float), args=(df,), full_output=True)
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
