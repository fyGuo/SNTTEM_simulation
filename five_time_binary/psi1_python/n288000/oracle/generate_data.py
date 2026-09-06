import numpy as np
import pandas as pd
from scipy.special import expit


def generate_data(n, psi05, psi15, psi25, psi35, psi45,
                  p_I1=0.5, p_I2=0.5, p_I3=0.5, p_I4=0.5,
                  ph1=0.05, ph2=0.05, ph3=0.05, ph4=0.05,
                  rng=None):
    """Generate longitudinal data for SNTTEM simulation.

    Five time points with binary treatment (A0..A4), covariates (L0..L4),
    eligibility indicators (I1..I4), and binary outcome Y.

    Eligibility is controlled per time point: p_I1 for I1, ..., p_I4 for I4.
    Treatment persistence is controlled per time point: ph1 is the probability
    A1 == A0, ph2 the probability A2 == A1, ..., ph4 the probability A4 == A3.

    Intercept is -2.6, not the -1.5 used in the psi0 (null) mechanism: with
    five compounded blip terms at psi=1, the -1.5 intercept lets exp(log_p)
    exceed 1 (worst case log_p = -1.5 + 2.5 = 1.0), which makes rng.binomial
    raise and every iteration in the cell come back NaN. -2.6 keeps
    exp(log_p) < 0.91 across the full production grid (n=72000, 300 iters,
    all 6 (p_I, ph) cells) with zero exceedances -- see CLAUDE.md "Known
    issues".
    """
    if rng is None:
        rng = np.random.default_rng()

    I0 = np.ones(n)
    L0 = rng.uniform(0, 0.5, n)
    A0 = rng.binomial(1, expit(-1 + L0), n)

    I1 = rng.binomial(1, p_I1, n)
    L1 = rng.normal(-0.5 * A0, 1, n)
    L1 = np.clip(L1, -1.5, -0.5)
    # A1 = A0 with prob ph1, else 1-A0
    A1 = np.where(rng.binomial(1, ph1, n) == 1, A0, 1 - A0)

    I2 = rng.binomial(1, p_I2, n)
    L2 = rng.normal(-0.5 * A1, 1, n)
    L2 = np.clip(L2, -1.5, -0.5)
    # A2 = A1 with prob ph2, else 1-A1
    A2 = np.where(rng.binomial(1, ph2, n) == 1, A1, 1 - A1)

    I3 = rng.binomial(1, p_I3, n)
    L3 = rng.normal(-0.5 * A2, 1, n)
    L3 = np.clip(L3, -1.5, -0.5)
    # A3 = A2 with prob ph3, else 1-A2
    A3 = np.where(rng.binomial(1, ph3, n) == 1, A2, 1 - A2)

    I4 = rng.binomial(1, p_I4, n)
    L4 = rng.normal(-0.5 * A3, 1, n)
    L4 = np.clip(L4, -1.5, -0.5)
    # A4 = A3 with prob ph4, else 1-A3
    A4 = np.where(rng.binomial(1, ph4, n) == 1, A3, 1 - A3)

    log_p = (
        -2.6
        + psi05 * L0 * A0
        + I1 * psi15 * (1 + L1) * (A1 - A0)
        + (1 - I1) * psi15 * L1 / 3 * (A1 - A0)
        + I2 * psi25 * (1 + L2) * (A2 - A1)
        + (1 - I2) * psi25 * L2 / 3 * (A2 - A1)
        + I3 * psi35 * (1 + L3) * (A3 - A2)
        + (1 - I3) * psi35 * L3 / 3 * (A3 - A2)
        + I4 * psi45 * (1 + L4) * (A4 - A3)
        + (1 - I4) * psi45 * L4 / 3 * (A4 - A3)
    )
    Y = rng.binomial(1, np.exp(log_p), n)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "I0": I0,
            "L0": L0,
            "A0": A0.astype(float),
            "I1": I1.astype(float),
            "L1": L1,
            "A1": A1.astype(float),
            "I2": I2.astype(float),
            "L2": L2,
            "A2": A2.astype(float),
            "I3": I3.astype(float),
            "L3": L3,
            "A3": A3.astype(float),
            "I4": I4.astype(float),
            "L4": L4,
            "A4": A4.astype(float),
            "Y": Y.astype(float),
        }
    )
    return df


# Blip functions: effect of changing treatment by one unit at each time point
def gamma45(L4, psi45):
    return psi45 * (L4 + 1)


def gamma35(L3, psi35):
    return psi35 * (L3 + 1)


def gamma25(L2, psi25):
    return psi25 * (L2 + 1)


def gamma15(L1, psi15):
    return psi15 * (1 + L1)


def gamma05(L0, psi05):
    return psi05 * L0
