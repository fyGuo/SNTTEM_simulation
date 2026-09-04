"""M-estimation and estimating equations for SNTTEM g-estimation.

Three estimators are implemented:
  - Three-step-g   ("three_step_g"):   conventional AIPW sequential regression
  - Three-step-ipw ("three_step_ipw"): conventional IPW sequential regression
  - Robins' g-estimator ("robins"):    SNTTEM-based estimator

All three solve a system of five estimating equations for
(psi05, psi15, psi25, psi35, psi45).
"""
import numpy as np
from scipy.optimize import fsolve

from generate_data import gamma05, gamma15, gamma25, gamma35, gamma45


def _unpack(df):
    v = lambda col: df[col].values
    return (
        v("A0"), v("A1"), v("A2"), v("A3"), v("A4"),
        v("I1"), v("I2"), v("I3"), v("I4"),
        v("Y"),
        v("L0"), v("L1"), v("L2"), v("L3"), v("L4"),
        v("ps0"), v("ps1"), v("ps2"), v("ps3"), v("ps4"),
        v("mu45"), v("mu35"), v("mu25"), v("mu15"), v("mu05"),
    )


def ee_three_step_g(theta, df):
    """Conventional (three-step-g / AIPW) estimating equations summed over observations."""
    psi05, psi15, psi25, psi35, psi45 = theta
    (A0, A1, A2, A3, A4,
     I1, I2, I3, I4,
     Y,
     L0, L1, L2, L3, L4,
     ps0, ps1, ps2, ps3, ps4,
     mu45, mu35, mu25, mu15, mu05) = _unpack(df)

    w4 = (A4 == A3).astype(float) / ((1 - ps4) * (1 - A3) + ps4 * A3)
    w3 = (A3 == A2).astype(float) / ((1 - ps3) * (1 - A2) + ps3 * A2)
    w2 = (A2 == A1).astype(float) / ((1 - ps2) * (1 - A1) + ps2 * A1)
    w1 = (A1 == A0).astype(float) / ((1 - ps1) * (1 - A0) + ps1 * A0)

    eq1 = np.sum(
        (A0 - ps0) * ((
            w4 * w3 * w2 * w1 * (Y - mu45)
            + w3 * w2 * w1 * (mu45 - mu35)
            + w2 * w1 * (mu35 - mu25)
            + w1 * (mu25 - mu15)
            + mu15) * np.exp(-gamma05(L0, psi05) * A0)
            - mu05
        )
    )
    eq2 = np.sum(
        I1 * (A1 - ps1) * ((
            w4 * w3 * w2 * (Y - mu45)
            + w3 * w2 * (mu45 - mu35)
            + w2 * (mu35 - mu25)
            + mu25) * np.exp(-gamma15(L1, psi15) * (A1 - A0))
            - mu15
        )
    )
    eq3 = np.sum(
        I2 * (A2 - ps2) *( (
            w4 * w3 * (Y - mu45)
            + w3 * (mu45 - mu35)
            + mu35)* np.exp(-gamma25(L2, psi25) * (A2 - A1))
            - mu25)
        )

    eq4 = np.sum(
        I3 * (A3 - ps3) * (
            (w4 * (Y - mu45)+ mu45)* np.exp(-gamma35(L3, psi35) * (A3 - A2)) - mu35
        )
    )
    eq5 = np.sum(
        I4 * (A4 - ps4) * (Y * np.exp(-gamma45(L4, psi45) * (A4 - A3)) - mu45)
    )
    return np.array([eq1, eq2, eq3, eq4, eq5])

def ee_three_step_ipw(theta, df):
    """Conventional (three-step-ipw) estimating equations summed over observations."""
    psi05, psi15, psi25, psi35, psi45 = theta
    (A0, A1, A2, A3, A4,
     I1, I2, I3, I4,
     Y,
     L0, L1, L2, L3, L4,
     ps0, ps1, ps2, ps3, ps4,
     mu45, mu35, mu25, mu15, mu05) = _unpack(df)

    w4 = (A4 == A3).astype(float) / ((1 - ps4) * (1 - A3) + ps4 * A3)
    w3 = (A3 == A2).astype(float) / ((1 - ps3) * (1 - A2) + ps3 * A2)
    w2 = (A2 == A1).astype(float) / ((1 - ps2) * (1 - A1) + ps2 * A1)
    w1 = (A1 == A0).astype(float) / ((1 - ps1) * (1 - A0) + ps1 * A0)

    eq1 = np.sum(
        (A0 - ps0) * ((
            w4 * w3 * w2 * w1 * (Y  * np.exp(-gamma05(L0, psi05) * A0)
            - mu05)
        )
    ))
    eq2 = np.sum(
        I1 * (A1 - ps1) * (
            w4 * w3 * w2 * (Y * np.exp(-gamma15(L1, psi15) * (A1 - A0))
            - mu15)
        )
    )
    eq3 = np.sum(
        I2 * (A2 - ps2) *(
            w4 * w3 * (Y* np.exp(-gamma25(L2, psi25) * (A2 - A1))
            - mu25))
        )

    eq4 = np.sum(
        I3 * (A3 - ps3) * (
            w4 * (Y* np.exp(-gamma35(L3, psi35) * (A3 - A2)) - mu35)
        )
    )
    eq5 = np.sum(
        I4 * (A4 - ps4) * (Y * np.exp(-gamma45(L4, psi45) * (A4 - A3)) - mu45)
    )
    return np.array([eq1, eq2, eq3, eq4, eq5])


def ee_robins(theta, df):
    """SNTTEM Robins' g-estimating equations summed over observations."""
    psi05, psi15, psi25, psi35, psi45 = theta
    (A0, A1, A2, A3, A4,
     I1, I2, I3, I4,
     Y,
     L0, L1, L2, L3, L4,
     ps0, ps1, ps2, ps3, ps4,
     mu45, mu35, mu25, mu15, mu05) = _unpack(df)

    w4 = (A4 == A3).astype(float) / ((1 - ps4) * (1 - A3) + ps4 * A3)
    w3 = (A3 == A2).astype(float) / ((1 - ps3) * (1 - A2) + ps3 * A2)
    w2 = (A2 == A1).astype(float) / ((1 - ps2) * (1 - A1) + ps2 * A1)
    w1 = (A1 == A0).astype(float) / ((1 - ps1) * (1 - A0) + ps1 * A0)

    # Use indicator-based weights: w^{1-I} so ineligible subjects get weight 1
    w4_I = w4 ** (1 - I4)
    w3_I = w3 ** (1 - I3)
    w2_I = w2 ** (1 - I2)
    w1_I = w1 ** (1 - I1)

    # Blip-removed increments at each time point, shared by all five equations
    b4 = Y * np.exp(-gamma45(L4, psi45) * (A4 - A3)) - mu45
    b3 = mu45 * np.exp(-gamma35(L3, psi35) * (A3 - A2)) - mu35
    b2 = mu35 * np.exp(-gamma25(L2, psi25) * (A2 - A1)) - mu25
    b1 = mu25 * np.exp(-gamma15(L1, psi15) * (A1 - A0)) - mu15
    b0 = mu15 * np.exp(-gamma05(L0, psi05) * A0) - mu05

    eq1 = np.sum(
        (A0 - ps0) * (
            w4_I * w3_I * w2_I * w1_I * b4
            + w3_I * w2_I * w1_I * b3
            + w2_I * w1_I * b2
            + w1_I * b1
            + b0
        )
    )
    eq2 = np.sum(
        I1 * (A1 - ps1) * (
            w4_I * w3_I * w2_I * b4
            + w3_I * w2_I * b3
            + w2_I * b2
            + b1
        )
    )
    eq3 = np.sum(
        I2 * (A2 - ps2) * (
            w4_I * w3_I * b4
            + w3_I * b3
            + b2
        )
    )
    eq4 = np.sum(
        I3 * (A3 - ps3) * (
            w4_I * b4
            + b3
        )
    )
    eq5 = np.sum(
        I4 * (A4 - ps4) * b4
    )
    return np.array([eq1, eq2, eq3, eq4, eq5])


def m_estimate(ee_fn, df, start=(0.0, 0.0, 0.0, 0.0, 0.0)):
    """Find the root of sum_i ee(theta, data_i) = 0 via scipy fsolve."""
    sol = fsolve(ee_fn, x0=np.array(start, dtype=float), args=(df,), full_output=True)
    estimates = sol[0]
    return estimates


def gmm_combine(est_con, est_gest):
    """Compute the optimal GMM linear combination of two estimators.

    Args:
        est_con:  (n_iter, 5) array of Three-step estimates (used as pilot).
        est_gest: (n_iter, 5) array of Robins' estimates (used as pilot).

    Returns:
        (n_iter, 5) array of GMM-combined estimates using the same inputs.
    """
    stacked = np.hstack([est_con, est_gest])   # (n_iter, 10)
    cov_mat = np.cov(stacked, rowvar=False)    # (10, 10)
    inv_mat = np.linalg.pinv(cov_mat)

    A = np.vstack([np.eye(5), np.eye(5)])      # (10, 5)
    mat2 = A.T @ inv_mat @ A                   # (5, 5)
    invcov = np.linalg.pinv(mat2)
    weight = invcov @ A.T @ inv_mat            # (5, 10)

    combined = (weight @ stacked.T).T          # (n_iter, 5)
    return combined
