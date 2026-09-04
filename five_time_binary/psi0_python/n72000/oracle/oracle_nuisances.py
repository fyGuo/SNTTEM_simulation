"""Exact closed-form nuisance functions implied by the known DGP.

Used to attach ps0..ps4 and mu05..mu45 directly from the DGP (generate_data.py)
at the true psi (all zero, as in this psi0 simulation) instead of fitting any
model from data -- no cross-fitting, no ML, no parametric working model.
Every estimator (Three-step-g, Three-step-ipw, Robins') is then given
nuisances with zero specification error and zero estimation noise, isolating
each estimating equation's own finite-sample behavior. See CLAUDE.md and
../check_ipw_ee_oracle.py / ../check_ipw_ee_oracle_mc.py, which this folder's
run_simulation.py productionizes to the full 6-cell x 300-iteration grid for
all three estimators (those check scripts covered Three-step-ipw only, at one
cell).

Closed-form truth used here (all valid because true psi05..psi45 = 0):
  - ps0  = expit(-1 + L0)                          (A0 depends on L0 for real)
  - ps_t = ph_t*A_{t-1} + (1-ph_t)*(1-A_{t-1})     (t=1..4; A_t doesn't
    depend on L_t at all in the DGP -- persistence is a coin flip on ph_t
    that ignores L_t)
  - mu05 = mu15 = mu25 = mu35 = mu45 = exp(-1.5)   (log_p = -1.5 exactly
    when every psi is 0, so Y is Bernoulli(exp(-1.5)) independent of
    everything)
"""
import numpy as np
from scipy.special import expit

TRUE_MU = np.exp(-1.5)


def oracle_nuisances(df, ph1, ph2, ph3, ph4):
    df = df.copy()
    df["ps0"] = expit(-1 + df["L0"].values)
    df["ps1"] = ph1 * df["A0"] + (1 - ph1) * (1 - df["A0"])
    df["ps2"] = ph2 * df["A1"] + (1 - ph2) * (1 - df["A1"])
    df["ps3"] = ph3 * df["A2"] + (1 - ph3) * (1 - df["A2"])
    df["ps4"] = ph4 * df["A3"] + (1 - ph4) * (1 - df["A3"])
    for col in ["mu05", "mu15", "mu25", "mu35", "mu45"]:
        df[col] = TRUE_MU
    return df
