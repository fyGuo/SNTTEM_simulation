"""Outer loop: run simulation over a grid of (ph, p_I) values with Ray.

Each (ph, p_I, iteration) triple is dispatched as a separate Ray task,
giving 12 * 100 = 1200 tasks that saturate all available CPUs.
Results are saved to simulation_results.pkl and simulation_results.csv.
"""
import os
import time
import numpy as np
import pandas as pd
import ray

from simulation_output import _one_iter

# True parameter values
PSI03 = 1.0
PSI13 = 1.0
PSI23 = 1.0
SIGMA = 20

# Simulation settings
N = 4000
N_ITER = 100
N_BOOT = 200
SEED = 3411

# Grid matching run_Rscript.R
PH_VALUES = [0.4, 0.6, 0.8]
P_I_VALUES = [0.25, 0.5, 0.75, 1]

output_path = os.path.join(os.path.dirname(__file__), "simulation_results.pkl")

COMMON = dict(
    n=N, psi03=PSI03, psi13=PSI13, psi23=PSI23,
    n_boot=N_BOOT, sigma=SIGMA,
    working_ps0=True, working_ps1=True, working_ps2=True,
    working_mu03=True, working_mu13=True, working_mu23=True,
)


@ray.remote
def _ray_iter(i, ph, p_I):
    return _one_iter(i=i, ph=ph, p_I=p_I, seed=SEED, **COMMON)


def _build_df(rows, ph, p_I):
    rows_df = pd.DataFrame(rows)
    ids = rows_df["id"].values
    shared = dict(p_I=p_I, ph=ph, seed=SEED,
                  psi03=PSI03, psi13=PSI13, psi23=PSI23, sigma=SIGMA)
    df_three = pd.DataFrame({
        "id": ids,
        "est_psi03": rows_df["est_psi03_old"].values,
        "est_psi13": rows_df["est_psi13_old"].values,
        "est_psi23": rows_df["est_psi23_old"].values,
        "method": "Three-step", **shared,
    })
    df_robins = pd.DataFrame({
        "id": ids,
        "est_psi03": rows_df["est_psi03_rbs"].values,
        "est_psi13": rows_df["est_psi13_rbs"].values,
        "est_psi23": rows_df["est_psi23_rbs"].values,
        "method": "Robins' estimator", **shared,
    })
    df_gmm = pd.DataFrame({
        "id": ids,
        "est_psi03": rows_df["est_psi03_gmm"].values,
        "est_psi13": rows_df["est_psi13_gmm"].values,
        "est_psi23": rows_df["est_psi23_gmm"].values,
        "method": "GMM estimator", **shared,
    })
    return pd.concat([df_three, df_robins, df_gmm], ignore_index=True)


if __name__ == "__main__":
    ray.init(address=None, runtime_env={"working_dir": "."})

    t_start = time.time()

    futures = {
        (ph, p_I, i): _ray_iter.remote(i, ph, p_I)
        for ph in PH_VALUES
        for p_I in P_I_VALUES
        for i in range(1, N_ITER + 1)
    }

    print(f"Submitted {len(futures)} tasks to Ray.")

    keys = list(futures.keys())
    all_results = ray.get([futures[k] for k in keys])

    results_by_scenario = {(ph, p_I): [] for ph in PH_VALUES for p_I in P_I_VALUES}
    for (ph, p_I, i), result in zip(keys, all_results):
        results_by_scenario[(ph, p_I)].append(result)

    ray.shutdown()

    all_dfs = [
        _build_df(rows, ph, p_I)
        for (ph, p_I), rows in results_by_scenario.items()
    ]
    results = pd.concat(all_dfs, ignore_index=True)
    results.to_pickle(output_path)
    results.to_csv(output_path.replace(".pkl", ".csv"), index=False)

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")
    print(f"Saved to {output_path}")
