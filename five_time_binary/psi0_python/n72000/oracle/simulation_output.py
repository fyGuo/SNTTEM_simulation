"""Run one simulation scenario, ORACLE variant.

All three estimators (Three-step-g, Three-step-ipw, Robins') are fitted using
the EXACT closed-form nuisance functions implied by the known DGP
(oracle_nuisances.py) instead of any fitted ML or parametric working model --
no cross-fitting, no nuisance-estimation noise at all. This isolates each
estimating equation's own finite-sample bias/variance from nuisance-estimation
error entirely.

The main function run_simulation() is called by run_simulation.py for each
(ph, p_I) combination, mirroring ../simulation_output.py's structure and
return format so ../check_results.py-style reporting works unchanged.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from generate_data import generate_data
from oracle_nuisances import oracle_nuisances
from estimators import ee_three_step_g, ee_three_step_ipw, ee_robins, m_estimate


def _one_iter(
    i,
    n,
    psi05,
    psi15,
    psi25,
    psi35,
    psi45,
    p_I1,
    p_I2,
    p_I3,
    p_I4,
    ph1,
    ph2,
    ph3,
    ph4,
    seed,
):
    """Run a single simulation iteration. Returns a dict of estimates.

    Protective: if anything in the iteration fails (data generation or
    root-finding), the estimates are returned as NaN and the error message is
    captured in the "error" field instead of aborting the run.
    """
    rng = np.random.default_rng([seed, i])

    try:
        df = generate_data(n, psi05=psi05, psi15=psi15, psi25=psi25,
                           psi35=psi35, psi45=psi45,
                           p_I1=p_I1, p_I2=p_I2, p_I3=p_I3, p_I4=p_I4,
                           ph1=ph1, ph2=ph2, ph3=ph3, ph4=ph4, rng=rng)

        df = oracle_nuisances(df, ph1, ph2, ph3, ph4)

        est_g = m_estimate(ee_three_step_g, df, start=(0.0, 0.0, 0.0, 0.0, 0.0))
        est_ipw = m_estimate(ee_three_step_ipw, df, start=(0.0, 0.0, 0.0, 0.0, 0.0))
        est_rbs = m_estimate(ee_robins, df, start=(0.0, 0.0, 0.0, 0.0, 0.0))
    except Exception as exc:  # noqa: BLE001 - record any failure as NaN
        nan = float("nan")
        return {
            "id": i,
            "est_psi05_g": nan, "est_psi15_g": nan, "est_psi25_g": nan,
            "est_psi35_g": nan, "est_psi45_g": nan,
            "est_psi05_ipw": nan, "est_psi15_ipw": nan, "est_psi25_ipw": nan,
            "est_psi35_ipw": nan, "est_psi45_ipw": nan,
            "est_psi05_rbs": nan, "est_psi15_rbs": nan, "est_psi25_rbs": nan,
            "est_psi35_rbs": nan, "est_psi45_rbs": nan,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "id": i,
        "est_psi05_g": est_g[0],
        "est_psi15_g": est_g[1],
        "est_psi25_g": est_g[2],
        "est_psi35_g": est_g[3],
        "est_psi45_g": est_g[4],
        "est_psi05_ipw": est_ipw[0],
        "est_psi15_ipw": est_ipw[1],
        "est_psi25_ipw": est_ipw[2],
        "est_psi35_ipw": est_ipw[3],
        "est_psi45_ipw": est_ipw[4],
        "est_psi05_rbs": est_rbs[0],
        "est_psi15_rbs": est_rbs[1],
        "est_psi25_rbs": est_rbs[2],
        "est_psi35_rbs": est_rbs[3],
        "est_psi45_rbs": est_rbs[4],
        "error": None,
    }


def run_simulation(
    n=6000,
    n_iter=100,
    psi05=0.0,
    psi15=0.0,
    psi25=0.0,
    psi35=0.0,
    psi45=0.0,
    p_I1=0.25,
    p_I2=0.25,
    p_I3=0.25,
    p_I4=0.25,
    ph1=0.30,
    ph2=0.30,
    ph3=0.30,
    ph4=0.30,
    seed=3411,
    n_jobs=-1,
):
    """Run the full ORACLE simulation and return a tidy DataFrame of estimates.

    Reports the Three-step-g, Three-step-ipw, and Robins' estimators, all
    fitted with exact closed-form (oracle) nuisances; no GMM combination.
    """
    common = dict(
        n=n, psi05=psi05, psi15=psi15, psi25=psi25, psi35=psi35, psi45=psi45,
        p_I1=p_I1, p_I2=p_I2, p_I3=p_I3, p_I4=p_I4,
        ph1=ph1, ph2=ph2, ph3=ph3, ph4=ph4,
    )

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_iter)(i=i, seed=seed, **common) for i in range(1, n_iter + 1)
    )

    rows = pd.DataFrame(results)
    ids = np.arange(1, n_iter + 1)

    # --- Protective reporting: which iterations failed (returned NaN) ---
    failed = rows[rows["error"].notna()]
    n_failed = len(failed)
    if n_failed:
        print(f"  [WARN] {n_failed}/{n_iter} iterations failed "
              f"(p_I1={p_I1}, p_I2={p_I2}, p_I3={p_I3}, p_I4={p_I4}, "
              f"ph1={ph1}, ph2={ph2}, ph3={ph3}, ph4={ph4}); "
              f"their estimates are NaN.")
        for _, frow in failed.iterrows():
            print(f"    iter {int(frow['id'])}: {frow['error']}")

    df_g = pd.DataFrame(
        {"id": ids,
         "est_psi05": rows["est_psi05_g"].values,
         "est_psi15": rows["est_psi15_g"].values,
         "est_psi25": rows["est_psi25_g"].values,
         "est_psi35": rows["est_psi35_g"].values,
         "est_psi45": rows["est_psi45_g"].values,
         "method": "Three-step-g"}
    )
    df_ipw = pd.DataFrame(
        {"id": ids,
         "est_psi05": rows["est_psi05_ipw"].values,
         "est_psi15": rows["est_psi15_ipw"].values,
         "est_psi25": rows["est_psi25_ipw"].values,
         "est_psi35": rows["est_psi35_ipw"].values,
         "est_psi45": rows["est_psi45_ipw"].values,
         "method": "Three-step-ipw"}
    )
    df_robins = pd.DataFrame(
        {"id": ids,
         "est_psi05": rows["est_psi05_rbs"].values,
         "est_psi15": rows["est_psi15_rbs"].values,
         "est_psi25": rows["est_psi25_rbs"].values,
         "est_psi35": rows["est_psi35_rbs"].values,
         "est_psi45": rows["est_psi45_rbs"].values,
         "method": "Robins' estimator"}
    )

    est_final = pd.concat([df_g, df_ipw, df_robins], ignore_index=True)
    est_final["p_I1"] = p_I1
    est_final["p_I2"] = p_I2
    est_final["p_I3"] = p_I3
    est_final["p_I4"] = p_I4
    est_final["ph1"] = ph1
    est_final["ph2"] = ph2
    est_final["ph3"] = ph3
    est_final["ph4"] = ph4
    est_final["seed"] = seed
    est_final["psi05"] = psi05
    est_final["psi15"] = psi15
    est_final["psi25"] = psi25
    est_final["psi35"] = psi35
    est_final["psi45"] = psi45

    return est_final


if __name__ == "__main__":
    import time
    t0 = time.time()
    results = run_simulation(n=6000, n_iter=100,
                             psi05=0, psi15=0, psi25=0, psi35=0, psi45=0,
                             p_I1=0.25, p_I2=0.25, p_I3=0.25, p_I4=0.25,
                             ph1=0.30, ph2=0.30, ph3=0.30, ph4=0.30, seed=3411)
    print(f"Elapsed: {time.time() - t0:.1f}s")
    summary = (
        results.groupby("method")[
            ["est_psi05", "est_psi15", "est_psi25", "est_psi35", "est_psi45"]
        ].agg(["mean", "var"])
    )
    print(summary)
