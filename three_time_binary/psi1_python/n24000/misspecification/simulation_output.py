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
from estimators import (
    ee_three_step, ee_robins, m_estimate,
    gamma13_true, gamma13_wrong, gamma23_true, gamma23_wrong,
)


def _bootstrap_gmm(df, actual_old, actual_rbs, n_boot, rng, g13, g23):
    """Estimate covariance via bootstrap and return the GMM-combined estimate.

    Resamples rows of df (which already has fitted nuisance columns) to avoid
    refitting expensive ML models. The bootstrap covariance is used to compute
    optimal inverse-covariance weights, which are then applied to the actual
    point estimates. g13, g23 select the (possibly misspecified) blip forms.
    """
    n = len(df)
    boot = np.empty((n_boot, 6))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        df_b = df.iloc[idx].reset_index(drop=True)
        est_old_b = m_estimate(ee_three_step, df_b, g13=g13, g23=g23)
        est_rbs_b = m_estimate(ee_robins, df_b, g13=g13, g23=g23)
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
    p_I1,
    p_I2,
    ph1,
    ph2,
    seed,
    n_boot,
    working_ps0,
    working_ps1,
    working_ps2,
    working_mu03,
    working_mu13,
    working_mu23,
    g13,
    g23,
):
    """Run a single simulation iteration. Returns a dict of estimates.

    Protective: if anything in the iteration fails (data generation, nuisance
    fitting, root-finding, bootstrap), the estimates are returned as NaN and the
    error message is captured in the "error" field instead of aborting the run.
    """
    rng = np.random.default_rng([seed, i])

    try:
        df = generate_data(n, psi03=psi03, psi13=psi13, psi23=psi23,
                           p_I1=p_I1, p_I2=p_I2, ph1=ph1, ph2=ph2, rng=rng)

        idx = rng.permutation(n)
        id_1 = idx[: n // 2]
        id_2 = idx[n // 2 :]

        df = working_model(
            df, id_1, id_2,
            ps0=working_ps0, ps1=working_ps1, ps2=working_ps2,
            mu03=working_mu03, mu13=working_mu13, mu23=working_mu23,
        )

        est_old = m_estimate(ee_three_step, df, start=(0.0, 0.0, 0.0),
                             g13=g13, g23=g23)
        est_rbs = m_estimate(ee_robins, df, start=(0.0, 0.0, 0.0),
                             g13=g13, g23=g23)

        gmm_est = _bootstrap_gmm(df, est_old, est_rbs, n_boot=n_boot, rng=rng,
                                 g13=g13, g23=g23)
    except Exception as exc:  # noqa: BLE001 - record any failure as NaN
        nan = float("nan")
        return {
            "id": i,
            "est_psi03_old": nan, "est_psi13_old": nan, "est_psi23_old": nan,
            "est_psi03_rbs": nan, "est_psi13_rbs": nan, "est_psi23_rbs": nan,
            "est_psi03_gmm": nan, "est_psi13_gmm": nan, "est_psi23_gmm": nan,
            "error": f"{type(exc).__name__}: {exc}",
        }

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
        "error": None,
    }


def run_simulation(
    n=6000,
    n_iter=100,
    psi03=0.0,
    psi13=0.0,
    psi23=0.0,
    p_I1=0.25,
    p_I2=0.25,
    ph1=0.30,
    ph2=0.30,
    seed=3411,
    n_jobs=-1,
    n_boot=200,
    working_ps0=True,
    working_ps1=True,
    working_ps2=True,
    working_mu03=True,
    working_mu13=True,
    working_mu23=True,
    gamma13_misspec=False,
    gamma23_misspec=False,
):
    """Run the full simulation and return a tidy DataFrame of estimates.

    Each iteration runs a bootstrap (n_boot resamples of the nuisance-fitted
    data) to estimate the cross-estimator covariance, then combines Three-step
    and Robins' estimates via optimal inverse-covariance GMM weighting.

    gamma13_misspec / gamma23_misspec: if True, fit with the WRONG blip form at
    time 1 / time 2 (the DGP is always the true one).
    """
    g13 = gamma13_wrong if gamma13_misspec else gamma13_true
    g23 = gamma23_wrong if gamma23_misspec else gamma23_true

    common = dict(
        n=n, psi03=psi03, psi13=psi13, psi23=psi23,
        p_I1=p_I1, p_I2=p_I2, ph1=ph1, ph2=ph2,
        n_boot=n_boot,
        working_ps0=working_ps0, working_ps1=working_ps1, working_ps2=working_ps2,
        working_mu03=working_mu03, working_mu13=working_mu13, working_mu23=working_mu23,
        g13=g13, g23=g23,
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
              f"(p_I1={p_I1}, p_I2={p_I2}, ph1={ph1}, ph2={ph2}); "
              f"their estimates are NaN.")
        for _, frow in failed.iterrows():
            print(f"    iter {int(frow['id'])}: {frow['error']}")

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
    est_final["p_I1"] = p_I1
    est_final["p_I2"] = p_I2
    est_final["ph1"] = ph1
    est_final["ph2"] = ph2
    est_final["seed"] = seed
    est_final["psi03"] = psi03
    est_final["psi13"] = psi13
    est_final["psi23"] = psi23
    est_final["gamma13"] = "wrong" if gamma13_misspec else "correct"
    est_final["gamma23"] = "wrong" if gamma23_misspec else "correct"

    return est_final


if __name__ == "__main__":
    import time
    t0 = time.time()
    results = run_simulation(n=6000, n_iter=100, psi03=0, psi13=0, psi23=0,
                             p_I1=0.25, p_I2=0.25, ph1=0.30, ph2=0.30, seed=3411)
    print(f"Elapsed: {time.time() - t0:.1f}s")
    summary = (
        results.groupby("method")[["est_psi03", "est_psi13", "est_psi23"]]
        .agg(["mean", "var"])
    )
    print(summary)
