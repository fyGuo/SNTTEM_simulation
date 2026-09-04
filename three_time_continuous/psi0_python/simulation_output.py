"""Run one simulation scenario and return per-iteration estimates.

Equivalent to simulation_output.R. The main function run_simulation() is
called by run_simulation.py for each (ph, p_I) combination.

The simulation:
  1. Generates n=6000 observations per iteration.
  2. Randomly splits data into two half-samples (cross-fitting).
  3. Fits ML nuisance models (SuperLearner-like ensemble) on each half.
  4. Solves estimating equations for psi03, psi13, psi23.
  5. Bootstraps the fitted dataset to estimate the cross-estimator covariance.
  6. Combines Three-step and Robins' estimates via inverse-covariance GMM weighting.
  7. Returns estimates from Three-step, Robins', and GMM methods.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from generate_data import generate_data
from working_models_ml import working_model
from estimators import ee_three_step, ee_robins, m_estimate


def _bootstrap_gmm(df, actual_old, actual_rbs, n_boot, rng):
    """Estimate covariance via bootstrap and return the GMM-combined estimate.

    Resamples rows of df (which already has fitted nuisance columns) to avoid
    refitting expensive ML models. The bootstrap covariance is used to compute
    optimal inverse-covariance weights, which are then applied to the actual
    point estimates.
    """
    n = len(df)
    boot = np.empty((n_boot, 6))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        df_b = df.iloc[idx].reset_index(drop=True)
        est_old_b = m_estimate(ee_three_step, df_b)
        est_rbs_b = m_estimate(ee_robins, df_b)
        boot[b] = np.concatenate([est_old_b, est_rbs_b])

    cov_mat = np.cov(boot, rowvar=False)          # (6, 6)
    inv_mat = np.linalg.pinv(cov_mat)
    A = np.vstack([np.eye(3), np.eye(3)])          # (6, 3)
    mat2 = A.T @ inv_mat @ A                       # (3, 3)
    weight = np.linalg.pinv(mat2) @ A.T @ inv_mat  # (3, 6)

    actual_stacked = np.concatenate([actual_old, actual_rbs])  # (6,)
    return weight @ actual_stacked                              # (3,)


def _one_iter(
    i,
    n,
    psi03,
    psi13,
    psi23,
    p_I,
    ph,
    seed,
    n_boot,
    working_ps0,
    working_ps1,
    working_ps2,
    working_mu03,
    working_mu13,
    working_mu23,
):
    """Run a single simulation iteration. Returns a dict of estimates."""
    rng = np.random.default_rng([seed, i])

    df = generate_data(n, psi03=psi03, psi13=psi13, psi23=psi23, p_I=p_I, ph=ph, rng=rng)

    idx = rng.permutation(n)
    id_1 = idx[: n // 2]
    id_2 = idx[n // 2 :]

    df = working_model(
        df, id_1, id_2,
        ps0=working_ps0, ps1=working_ps1, ps2=working_ps2,
        mu03=working_mu03, mu13=working_mu13, mu23=working_mu23,
    )

    est_old = m_estimate(ee_three_step, df, start=(0.0, 0.0, 0.0))
    est_rbs = m_estimate(ee_robins, df, start=(0.0, 0.0, 0.0))

    gmm_est = _bootstrap_gmm(df, est_old, est_rbs, n_boot=n_boot, rng=rng)

    return {
        "id": i,
        "est_psi03_old": est_old[0],
        "est_psi13_old": est_old[1],
        "est_psi23_old": est_old[2],
        "est_psi03_rbs": est_rbs[0],
        "est_psi13_rbs": est_rbs[1],
        "est_psi23_rbs": est_rbs[2],
        "est_psi03_gmm": gmm_est[0],
        "est_psi13_gmm": gmm_est[1],
        "est_psi23_gmm": gmm_est[2],
    }


def run_simulation(
    n=6000,
    n_iter=100,
    psi03=0.0,
    psi13=0.0,
    psi23=0.0,
    p_I=0.25,
    ph=0.30,
    seed=3411,
    n_jobs=-1,
    n_boot=200,
    working_ps0=True,
    working_ps1=True,
    working_ps2=True,
    working_mu03=True,
    working_mu13=True,
    working_mu23=True,
):
    """Run the full simulation and return a tidy DataFrame of estimates.

    Each iteration runs a bootstrap (n_boot resamples of the nuisance-fitted
    data) to estimate the cross-estimator covariance, then combines Three-step
    and Robins' estimates via optimal inverse-covariance GMM weighting.
    """
    common = dict(
        n=n, psi03=psi03, psi13=psi13, psi23=psi23, p_I=p_I, ph=ph,
        n_boot=n_boot,
        working_ps0=working_ps0, working_ps1=working_ps1, working_ps2=working_ps2,
        working_mu03=working_mu03, working_mu13=working_mu13, working_mu23=working_mu23,
    )

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_iter)(i=i, seed=seed, **common) for i in range(1, n_iter + 1)
    )

    rows = pd.DataFrame(results)
    ids = np.arange(1, n_iter + 1)

    df_three = pd.DataFrame(
        {"id": ids,
         "est_psi03": rows["est_psi03_old"].values,
         "est_psi13": rows["est_psi13_old"].values,
         "est_psi23": rows["est_psi23_old"].values,
         "method": "Three-step"}
    )
    df_robins = pd.DataFrame(
        {"id": ids,
         "est_psi03": rows["est_psi03_rbs"].values,
         "est_psi13": rows["est_psi13_rbs"].values,
         "est_psi23": rows["est_psi23_rbs"].values,
         "method": "Robins' estimator"}
    )
    df_gmm = pd.DataFrame(
        {"id": ids,
         "est_psi03": rows["est_psi03_gmm"].values,
         "est_psi13": rows["est_psi13_gmm"].values,
         "est_psi23": rows["est_psi23_gmm"].values,
         "method": "GMM estimator"}
    )

    est_final = pd.concat([df_three, df_robins, df_gmm], ignore_index=True)
    est_final["p_I"] = p_I
    est_final["ph"] = ph
    est_final["seed"] = seed
    est_final["psi03"] = psi03
    est_final["psi13"] = psi13
    est_final["psi23"] = psi23

    return est_final


if __name__ == "__main__":
    import time
    t0 = time.time()
    results = run_simulation(n=6000, n_iter=100, psi03=0, psi13=0, psi23=0,
                             p_I=0.25, ph=0.30, seed=3411)
    print(f"Elapsed: {time.time() - t0:.1f}s")
    summary = (
        results.groupby("method")[["est_psi03", "est_psi13", "est_psi23"]]
        .agg(["mean", "var"])
    )
    print(summary)
