"""Step 1 of the Three-step-ipw bias investigation: is the estimating
equation itself correct?

Rather than fitting ML nuisance models, plug in the EXACT nuisance functions
implied by the known DGP (generate_data.py) at the true psi (all zero, as in
the ph=0.3 scale-test runs), and check whether the estimating equations
evaluate to (approximately) zero at the truth on a very large sample --
removing nuisance-estimation noise entirely so any remaining discrepancy is
about the equation's specification, not about ML fitting error.

Closed-form truth used here (all valid because true psi05..psi45 = 0):
  - ps0 = expit(-1 + L0)                          (A0 depends on L0 for real)
  - ps_t = ph_t*A_{t-1} + (1-ph_t)*(1-A_{t-1})     (t=1..4; A_t doesn't
    depend on L_t at all in the DGP -- persistence is a coin flip on ph_t
    that ignores L_t)
  - mu05 = mu15 = mu25 = mu35 = mu45 = exp(-1.5)   (log_p = -1.5 exactly
    when every psi is 0, so Y is Bernoulli(exp(-1.5)) independent of
    everything)

Two checks per (p_I, ph) cell, each on one large-n draw (no repeated
iterations needed -- with n in the millions, Monte Carlo noise on the score
mean is already tiny):
  1. Evaluate each estimating equation at theta = 0 directly and report the
     per-observation mean +/- its Monte Carlo SE (is the score unbiased at
     the truth?).
  2. Solve the system via fsolve and report the root (does psi-hat land on
     0?).
"""
import numpy as np
import pandas as pd
from scipy.special import expit

from generate_data import generate_data
from estimators import ee_three_step_g, ee_three_step_ipw, ee_robins, m_estimate

N = 5_000_000
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


def check_cell(p_I, ph, seed):
    rng = np.random.default_rng(seed)
    df = generate_data(
        N, psi05=0.0, psi15=0.0, psi25=0.0, psi35=0.0, psi45=0.0,
        p_I1=p_I, p_I2=p_I, p_I3=p_I, p_I4=p_I,
        ph1=ph, ph2=ph, ph3=ph, ph4=ph, rng=rng,
    )
    df = oracle_nuisances(df, ph, ph, ph, ph)

    print(f"\n{'='*70}\np_I={p_I}, ph={ph}, n={N:,}\n{'='*70}")

    for name, ee_fn in [
        ("Three-step-g", ee_three_step_g),
        ("Three-step-ipw", ee_three_step_ipw),
        ("Robins'", ee_robins),
    ]:
        theta0 = np.zeros(5)
        eq_sum = ee_fn(theta0, df)  # length-5 array of SUMS over n obs

        # Need per-observation contributions for an SE; recompute by calling
        # the ee function is only the summed version, so instead perturb: use
        # the fact that eq_sum / N is the sample mean of the (unobserved)
        # per-obs score. For the SE we re-derive per-obs values by calling
        # the ee function on singleton-weighted subsets is wasteful; instead
        # just bootstrap via splitting the sample into chunks.
        n_chunks = 50
        chunk_means = []
        idx_split = np.array_split(np.arange(N), n_chunks)
        for idx in idx_split:
            sub = df.iloc[idx]
            chunk_means.append(ee_fn(theta0, sub) / len(idx))
        chunk_means = np.array(chunk_means)  # (n_chunks, 5)
        se = chunk_means.std(axis=0, ddof=1) / np.sqrt(n_chunks)
        mean = eq_sum / N

        z = mean / se
        print(f"\n{name}: mean score at theta=0 (should be ~0), "
              f"+/- MC SE, z-score")
        labels = ["psi05", "psi15", "psi25", "psi35", "psi45"]
        for lab, m, s, zz in zip(labels, mean, se, z):
            flag = "  <-- SIGNIFICANT" if abs(zz) > 3 else ""
            print(f"  {lab}: {m:+.6f} +/- {s:.6f}   z={zz:+.2f}{flag}")

        root = m_estimate(ee_fn, df)
        print(f"  fsolve root: {np.round(root, 5)}")


if __name__ == "__main__":
    for p_I in [0.2, 1.0]:
        check_cell(p_I=p_I, ph=0.3, seed=12345)
