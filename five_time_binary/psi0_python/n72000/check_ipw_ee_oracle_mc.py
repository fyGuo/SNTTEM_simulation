"""Step 1 (replicated version): run the actual 300-iteration MC design at
n=72000, ph=0.3, but with EXACT oracle nuisances (closed-form from the known
DGP) instead of ML-fitted ones -- no cross-fitting, no working_model() call.

If the estimating equations are correctly specified, the median psi-hat
across replicates should land close to 0 for all three methods here, even
though the earlier ML-nuisance run showed Three-step-ipw's psi05 median
sitting at +0.11 to +0.17. This isolates equation-correctness from
nuisance-estimation error.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from joblib import Parallel, delayed

from generate_data import generate_data
from estimators import ee_three_step_g, ee_three_step_ipw, ee_robins, m_estimate

N = 72000
N_ITER = 300
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


def _one_iter(i, p_I, ph, seed):
    rng = np.random.default_rng([seed, i])
    df = generate_data(
        N, psi05=0.0, psi15=0.0, psi25=0.0, psi35=0.0, psi45=0.0,
        p_I1=p_I, p_I2=p_I, p_I3=p_I, p_I4=p_I,
        ph1=ph, ph2=ph, ph3=ph, ph4=ph, rng=rng,
    )
    df = oracle_nuisances(df, ph, ph, ph, ph)

    out = {"id": i}
    for name, ee_fn in [
        ("Three-step-g", ee_three_step_g),
        ("Three-step-ipw", ee_three_step_ipw),
        ("Robins'", ee_robins),
    ]:
        try:
            theta = m_estimate(ee_fn, df)
        except Exception:
            theta = [np.nan] * 5
        out[name] = theta
    return out


def run_cell(p_I, ph, seed):
    results = Parallel(n_jobs=-1)(
        delayed(_one_iter)(i, p_I, ph, seed) for i in range(1, N_ITER + 1)
    )
    rows = []
    for r in results:
        for method in ["Three-step-g", "Three-step-ipw", "Robins'"]:
            psi05, psi15, psi25, psi35, psi45 = r[method]
            rows.append(dict(
                id=r["id"], method=method,
                est_psi05=psi05, est_psi15=psi15, est_psi25=psi25,
                est_psi35=psi35, est_psi45=psi45,
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for p_I in [0.2, 1.0]:
        df = run_cell(p_I=p_I, ph=0.3, seed=3411)
        print(f"\n{'='*70}\np_I={p_I}, ph=0.3, n={N}, oracle nuisances, "
              f"{N_ITER} iterations\n{'='*70}")
        for method in ["Three-step-g", "Three-step-ipw", "Robins'"]:
            sub = df[df["method"] == method]
            print(f"\n{method}:")
            for p in ["psi05", "psi15", "psi25", "psi35", "psi45"]:
                x = sub[f"est_{p}"]
                failrate = (x.abs() > 10).mean()
                med = np.nan if failrate > 0.05 else x.median()
                lo, hi = x.quantile(0.025), x.quantile(0.975)
                var = ((hi - lo) / 3.96) ** 2
                print(f"  {p}: median={med:+.4f}  var={var:.4f}  "
                      f"failrate={failrate:.2%}")
        df.to_csv(f"oracle_mc_pI{p_I}_ph03.csv", index=False)
