"""Exact closed-form nuisance functions implied by the known DGP (psi1: all
psi_j5 = 1.0, intercept -2.6).

Propensity scores are unchanged from psi0_python/n72000/oracle/'s derivation
-- A_t's generation doesn't depend on psi at all, so ps0..ps4 have the same
closed form here as there.

mu05..mu45 are NOT constant here (unlike psi0, where psi=0 collapses Y to an
unconditional Bernoulli(exp(-1.5))). Derivation: each mu_{j5} is defined
recursively in working_models_ml.py's working_model() as
E[dr_mu_{(j+1)5} | history up to time j, A_j == A_{j-1}] (with mu45 :=
E[Y | full history, A4==A3] as the base case). Substituting the DGP's log_p
formula and working through the recursion (using that mu_{(j+1)5}, by
induction, doesn't depend on L_j/I_j/A_j -- only on history strictly before
step j -- so the A_j==A_{j-1} masking exactly zeroes out step j's own psi_j5
blip term, with no other term affected) gives, for j=0..4:

    mu05 = exp(c)
    mu15 = exp(c + psi05*L0*A0)
    mu25 = exp(c + psi05*L0*A0 + [psi15 term](L1,A0,A1,I1))
    mu35 = mu25's exponent + [psi25 term](L2,A1,A2,I2)
    mu45 = mu35's exponent + [psi35 term](L3,A2,A3,I3)

where c = -2.6 (generate_data.py's intercept) and [psi_j5 term] is exactly
the corresponding summand in generate_data.py's log_p:
I_t*psi*(1+L_t)*(A_t-A_{t-1}) + (1-I_t)*psi*L_t/3*(A_t-A_{t-1}). Note mu45
never involves psi45 -- that blip is handled entirely by the estimating
equation's own exp(-gamma45*...) term, not by the mu nuisance, matching the
K=5 "deepest equation has a different shape" note in CLAUDE.md.

This was verified (not just derived) via a score-at-truth check analogous to
../../../psi0_python/n72000/check_ipw_ee_oracle.py: at theta =
(1,1,1,1,1), all three estimating equations' per-observation mean is ~0
(within Monte Carlo noise) on a large-n draw using these nuisances -- see
check_oracle_derivation.py in this folder.
"""
import numpy as np
from scipy.special import expit

INTERCEPT = -2.6


def oracle_nuisances(df, ph1, ph2, ph3, ph4,
                      psi05=1.0, psi15=1.0, psi25=1.0, psi35=1.0):
    df = df.copy()
    df["ps0"] = expit(-1 + df["L0"].values)
    df["ps1"] = ph1 * df["A0"] + (1 - ph1) * (1 - df["A0"])
    df["ps2"] = ph2 * df["A1"] + (1 - ph2) * (1 - df["A1"])
    df["ps3"] = ph3 * df["A2"] + (1 - ph3) * (1 - df["A2"])
    df["ps4"] = ph4 * df["A3"] + (1 - ph4) * (1 - df["A3"])

    L0, A0 = df["L0"].values, df["A0"].values
    I1, L1, A1 = df["I1"].values, df["L1"].values, df["A1"].values
    I2, L2, A2 = df["I2"].values, df["L2"].values, df["A2"].values
    I3, L3, A3 = df["I3"].values, df["L3"].values, df["A3"].values

    term05 = psi05 * L0 * A0
    term15 = (I1 * psi15 * (1 + L1) * (A1 - A0)
              + (1 - I1) * psi15 * L1 / 3 * (A1 - A0))
    term25 = (I2 * psi25 * (1 + L2) * (A2 - A1)
              + (1 - I2) * psi25 * L2 / 3 * (A2 - A1))
    term35 = (I3 * psi35 * (1 + L3) * (A3 - A2)
              + (1 - I3) * psi35 * L3 / 3 * (A3 - A2))

    df["mu05"] = np.exp(INTERCEPT * np.ones(len(df)))
    df["mu15"] = np.exp(INTERCEPT + term05)
    df["mu25"] = np.exp(INTERCEPT + term05 + term15)
    df["mu35"] = np.exp(INTERCEPT + term05 + term15 + term25)
    df["mu45"] = np.exp(INTERCEPT + term05 + term15 + term25 + term35)
    return df
