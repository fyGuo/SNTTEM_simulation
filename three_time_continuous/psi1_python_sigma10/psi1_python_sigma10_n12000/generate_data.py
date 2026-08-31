import numpy as np
import pandas as pd
from scipy.special import expit


def generate_data(n, psi03, psi13, psi23, p_I1=0.5, p_I2=0.5, ph1=0.05, ph2=0.05,
                  rng=None, sigma=0):
    """Generate longitudinal data for SNTTEM simulation (continuous outcome).

    Three time points with binary treatment (A0, A1, A2), covariates (L0, L1, L2),
    eligibility indicators (I1, I2), and continuous outcome Y.

    Eligibility is controlled per time point: p_I1 for I1, p_I2 for I2.
    Treatment persistence is controlled per time point: ph1 is the probability
    A1 == A0, ph2 the probability A2 == A1.
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

    mu_Y = (
        -1.5
        + psi03 * L0 * A0
        + psi13 * (1 + L1) * (A1 - A0)
        + psi23 * (1 + L2) * (A2 - A1)
    )
    sigma_Y = sigma * (A0 + A1 + A2)
    Y = rng.normal(mu_Y, sigma_Y, n)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "I0": I0,
            "L0": L0,
            "A0": A0.astype(float),
            "L1": L1,
            "I1": I1.astype(float),
            "A1": A1.astype(float),
            "I2": I2.astype(float),
            "L2": L2,
            "A2": A2.astype(float),
            "Y": Y.astype(float),
        }
    )
    return df


# Blip functions: effect of changing treatment by one unit at each time point
def gamma23(L2, psi23):
    return psi23 * (L2 + 1)


def gamma13(L1, psi13):
    return psi13 * (1 + L1)


def gamma03(L0, psi03):
    return psi03 * L0
