"""Verify oracle_nuisances.py's closed-form mu05..mu45 against the DGP.

Analogous to ../../../psi0_python/n72000/check_ipw_ee_oracle.py, but for
psi1's non-null mechanism (psi05..psi45 = 1.0), where mu05..mu45 are not
constant and had to be derived (see oracle_nuisances.py's docstring). Plug
the derived nuisances into all three estimating equations at the true theta
= (1,1,1,1,1) and check the per-observation mean score is ~0 on a large-n
draw -- if the derivation is wrong, this will show a large, many-sigma
nonzero mean somewhere.
"""
import numpy as np

from generate_data import generate_data
from oracle_nuisances import oracle_nuisances
from estimators import ee_three_step_g, ee_three_step_ipw, ee_robins, m_estimate

N = 5_000_000
TRUE_THETA = (1.0, 1.0, 1.0, 1.0, 1.0)


def check_cell(p_I, ph, seed):
    rng = np.random.default_rng(seed)
    df = generate_data(
        N, psi05=1.0, psi15=1.0, psi25=1.0, psi35=1.0, psi45=1.0,
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
        theta = np.array(TRUE_THETA)
        eq_sum = ee_fn(theta, df)

        n_chunks = 50
        chunk_means = []
        idx_split = np.array_split(np.arange(N), n_chunks)
        for idx in idx_split:
            sub = df.iloc[idx]
            chunk_means.append(ee_fn(theta, sub) / len(idx))
        chunk_means = np.array(chunk_means)
        se = chunk_means.std(axis=0, ddof=1) / np.sqrt(n_chunks)
        mean = eq_sum / N

        z = mean / se
        print(f"\n{name}: mean score at theta={TRUE_THETA} (should be ~0), "
              f"+/- MC SE, z-score")
        labels = ["psi05", "psi15", "psi25", "psi35", "psi45"]
        for lab, m, s, zz in zip(labels, mean, se, z):
            flag = "  <-- SIGNIFICANT" if abs(zz) > 3 else ""
            print(f"  {lab}: {m:+.6f} +/- {s:.6f}   z={zz:+.2f}{flag}")

        root = m_estimate(ee_fn, df, start=TRUE_THETA)
        print(f"  fsolve root (start at truth): {np.round(root, 5)}")


if __name__ == "__main__":
    for p_I in [0.2, 1.0]:
        check_cell(p_I=p_I, ph=0.3, seed=12345)
