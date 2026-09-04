"""
Simulation for: Overlapped vs. Non-overlapped Population Comparison.

Implements the design in simulation_design.md:
  - DGM of section 2 with default parameters of section 5
  - centered g-estimation of the blip parameters psi_2, psi_1 (section 1)
      * q2 fit by OLS on the A2 == 0 stratum
      * q1 fit by OLS of qhat2 on the A1 == 0 stratum
  - proposed estimator tau_hat (IPW non-overlapped term - hybrid overlapped term)
  - benchmark fully-IPW estimator tau_hat^IPW
  - oracle check using the true blips

True marginal contrast E(Y^{0,1,0} - Y^{1,1,0}) = -(beta2 + beta3*lam12 + psi11*lam12) = -1.75
"""

import time
import numpy as np

# ----------------------------------------------------------------------------
# Default parameters (section 5)
# ----------------------------------------------------------------------------
P = dict(
    # pi0
    a00=0.0, a01=0.5,
    # L1
    lam10=0.0, lam11=0.6, lam12=0.5, sL1=1.0,
    # pi1
    a10=0.0, a11=0.3, a12=0.4, a13=0.5,
    # L2
    lam20=0.0, lam21=0.6, lam22=0.5, lam23=0.3, sL2=1.0,
    # pi2
    a20=0.0, a21=0.3, a22=0.4, a23=0.5,
    # treatment-free mean h
    b0=1.0, b1=0.5, b2=1.0, b3=0.8,
    # primitive blips
    psi10=0.5, psi11=0.7,   # delta1 = psi10 + psi11*L1
    psi20=0.4, psi21=0.0,   # delta2 = psi20
    # outcome noise
    sY=1.0,
)

TRUE_TAU = -(P["b2"] + P["b3"] * P["lam12"] + P["psi11"] * P["lam12"])  # -1.75
TRUE_PSI2 = np.array([P["psi20"], 0.0])                    # gamma2 = psi20 (+ 0 * L2)
TRUE_PSI1 = np.array([P["psi10"] + P["psi20"],             # theta0 = 0.9
                      P["psi11"],                          # theta1 = 0.7
                      -P["psi20"]])                        # theta2 = -0.4


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------
# Data-generating mechanism (section 2)
# ----------------------------------------------------------------------------
def simulate(n, rng):
    L0 = rng.normal(0.0, 1.0, n)
    pi0 = expit(P["a00"] + P["a01"] * L0)
    A0 = rng.binomial(1, pi0)

    L1 = rng.normal(P["lam10"] + P["lam11"] * L0 + P["lam12"] * A0, P["sL1"])
    pi1 = expit(P["a10"] + P["a11"] * L0 + P["a12"] * A0 + P["a13"] * L1)
    A1 = rng.binomial(1, pi1)

    L2 = rng.normal(P["lam20"] + P["lam21"] * L1 + P["lam22"] * A1 + P["lam23"] * A0, P["sL2"])
    pi2 = expit(P["a20"] + P["a21"] * L1 + P["a22"] * A1 + P["a23"] * L2)
    A2 = rng.binomial(1, pi2)

    Y = (P["b0"] + P["b1"] * L0 + P["b2"] * A0 + P["b3"] * L1
         + A1 * (P["psi10"] + P["psi11"] * L1)
         + A2 * P["psi20"]
         + rng.normal(0.0, P["sY"], n))

    return dict(L0=L0, A0=A0, L1=L1, A1=A1, L2=L2, A2=A2, Y=Y)


def true_props(d):
    """Exact DGM logistic propensities used as the working propensity models."""
    pi0 = expit(P["a00"] + P["a01"] * d["L0"])
    pi1 = expit(P["a10"] + P["a11"] * d["L0"] + P["a12"] * d["A0"] + P["a13"] * d["L1"])
    pi2 = expit(P["a20"] + P["a21"] * d["L1"] + P["a22"] * d["A1"] + P["a23"] * d["L2"])
    return pi0, pi1, pi2


def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ----------------------------------------------------------------------------
# Centered g-estimation of psi2, psi1 (section 1)
# ----------------------------------------------------------------------------
def gest_blips(d, pi1, pi2):
    n = d["Y"].size
    one = np.ones(n)
    A0, A1, A2, L0, L1, L2, Y = (d["A0"], d["A1"], d["A2"],
                                 d["L0"], d["L1"], d["L2"], d["Y"])

    # ---- q2 = E[Y | S2, A2 = 0], working model linear in (1,L0,A0,L1,A1,A1*L1)
    Xq2 = np.column_stack([one, L0, A0, L1, A1, A1 * L1])
    m0 = A2 == 0
    xi2 = ols_fit(Xq2[m0], Y[m0])
    q2hat = Xq2 @ xi2

    # ---- psi2:  Pn[ m2 (A2 - pi2) (Y - A2*gamma2 - q2) ] = 0
    m2 = np.column_stack([one, L2])                       # (n, 2)
    w2 = (A2 - pi2)
    Amat2 = (m2 * (w2 * A2)[:, None]).T @ m2 / n          # Pn[m2 (A2-pi2) A2 m2^T]
    bvec2 = (m2 * (w2 * (Y - q2hat))[:, None]).sum(0) / n
    psi2 = np.linalg.solve(Amat2, bvec2)
    gamma2 = m2 @ psi2

    # ---- q1 = E[ qhat2 | S1, A1 = 0 ], working model linear in (1,L0,A0,L1)
    Xq1 = np.column_stack([one, L0, A0, L1])
    m1mask = A1 == 0
    xi1 = ols_fit(Xq1[m1mask], q2hat[m1mask])
    q1hat = Xq1 @ xi1

    # ---- psi1:  Pn[ m1 (A1 - pi1) (R1 - A1*gamma1 - q1) ] = 0
    #      R1 = Y - {A2 - (1-A0)A1} gamma2
    R1 = Y - (A2 - (1 - A0) * A1) * gamma2
    m1 = np.column_stack([one, L1, A0])                   # (n, 3)
    w1 = (A1 - pi1)
    Amat1 = (m1 * (w1 * A1)[:, None]).T @ m1 / n
    bvec1 = (m1 * (w1 * (R1 - q1hat))[:, None]).sum(0) / n
    psi1 = np.linalg.solve(Amat1, bvec1)
    gamma1 = m1 @ psi1

    return psi2, psi1, gamma2, gamma1


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def proposed_tau(d, pi0, pi1, gamma2, gamma1):
    A0, A1, A2, Y = d["A0"], d["A1"], d["A2"], d["Y"]
    phi_non = (1 - A0) / (1 - pi0) * A1 / pi1 * (Y - A2 * gamma2)
    phi_ovl = A0 / pi0 * (Y - A2 * gamma2 - (A1 - 1) * gamma1)
    contrib = phi_non - phi_ovl
    tau = contrib.mean()
    se = np.sqrt(np.mean((contrib - tau) ** 2) / contrib.size)
    return tau, se


def partial_ipw_tau(d, pi0, pi1, gamma2):
    """Section 3.3: IPW for A0,A1 in both arms; gamma2 blip for A2 (no pi2, no gamma1)."""
    A0, A1, A2, Y = d["A0"], d["A1"], d["A2"], d["Y"]
    H2 = Y - A2 * gamma2
    rho_non = (1 - A0) / (1 - pi0) * A1 / pi1 * H2
    rho_ovl = A0 / pi0 * A1 / pi1 * H2
    contrib = rho_non - rho_ovl
    tau = contrib.mean()
    se = np.sqrt(np.mean((contrib - tau) ** 2) / contrib.size)
    return tau, se


def ipw_tau(d, pi0, pi1, pi2):
    A0, A1, A2, Y = d["A0"], d["A1"], d["A2"], d["Y"]
    base = A1 / pi1 * (1 - A2) / (1 - pi2) * Y
    psi_non = (1 - A0) / (1 - pi0) * base
    psi_ovl = A0 / pi0 * base
    contrib = psi_non - psi_ovl
    tau = contrib.mean()
    se = np.sqrt(np.mean((contrib - tau) ** 2) / contrib.size)
    return tau, se


def oracle_gammas(d):
    gamma2 = np.full(d["Y"].size, TRUE_PSI2[0])           # = 0.4
    gamma1 = TRUE_PSI1[0] + TRUE_PSI1[1] * d["L1"] + TRUE_PSI1[2] * d["A0"]
    return gamma2, gamma1


def estimate_all(d):
    """All four target-contrast point estimates on dataset d (true propensities)."""
    pi0, pi1, pi2 = true_props(d)
    psi2, psi1, gamma2, gamma1 = gest_blips(d, pi1, pi2)
    og2, og1 = oracle_gammas(d)
    tau = dict(
        proposed=proposed_tau(d, pi0, pi1, gamma2, gamma1)[0],
        partial_ipw=partial_ipw_tau(d, pi0, pi1, gamma2)[0],
        ipw=ipw_tau(d, pi0, pi1, pi2)[0],
        oracle=proposed_tau(d, pi0, pi1, og2, og1)[0],
    )
    return tau, psi2, psi1


# ----------------------------------------------------------------------------
# Bootstrap (one Monte Carlo rep + its nonparametric bootstrap)
# ----------------------------------------------------------------------------
KEYS = ["proposed", "partial_ipw", "ipw", "oracle"]


def bootstrap_rep(args):
    """One MC replication with a nonparametric bootstrap (picklable for parallel map).

    args = (seed, n, B). Resamples subjects with replacement B times, refitting the
    g-estimation blips on each resample, and forms a percentile 95% CI per estimator.
    Returns point estimates, coverage indicators, CI widths, and blip estimates.
    """
    seed, n, B = args
    rng = np.random.default_rng(seed)
    d = simulate(n, rng)

    point, psi2, psi1 = estimate_all(d)
    boot = {k: np.empty(B) for k in KEYS}
    for b in range(B):
        idx = rng.integers(0, n, n)
        db = {key: d[key][idx] for key in d}
        tb, _, _ = estimate_all(db)
        for k in KEYS:
            boot[k][b] = tb[k]

    cover, width = {}, {}
    for k in KEYS:
        lo, hi = np.quantile(boot[k], [0.025, 0.975])
        cover[k] = bool(lo <= TRUE_TAU <= hi)
        width[k] = float(hi - lo)
    return dict(point=point, cover=cover, width=width, psi2=psi2, psi1=psi1)


# ----------------------------------------------------------------------------
# Monte Carlo driver
# ----------------------------------------------------------------------------
def run(R=100, n=2000, seed=20260622):
    rng = np.random.default_rng(seed)
    keys = ["proposed", "partial_ipw", "ipw", "oracle"]
    est = {k: np.empty(R) for k in keys}
    se = {k: np.empty(R) for k in keys}
    psi2_all = np.empty((R, 2))
    psi1_all = np.empty((R, 3))

    t0 = time.perf_counter()
    for r in range(R):
        d = simulate(n, rng)
        pi0, pi1, pi2 = true_props(d)

        psi2, psi1, gamma2, gamma1 = gest_blips(d, pi1, pi2)
        psi2_all[r] = psi2
        psi1_all[r] = psi1

        est["proposed"][r], se["proposed"][r] = proposed_tau(d, pi0, pi1, gamma2, gamma1)
        est["partial_ipw"][r], se["partial_ipw"][r] = partial_ipw_tau(d, pi0, pi1, gamma2)
        est["ipw"][r], se["ipw"][r] = ipw_tau(d, pi0, pi1, pi2)

        og2, og1 = oracle_gammas(d)
        est["oracle"][r], se["oracle"][r] = proposed_tau(d, pi0, pi1, og2, og1)

    elapsed = time.perf_counter() - t0

    # ---- report
    print(f"R={R} reps, n={n}, true tau = {TRUE_TAU:.4f}")
    print(f"elapsed: {elapsed:.2f}s  ({1000*elapsed/R:.1f} ms / iterate)\n")

    z = 1.959963985
    header = (f"{'estimator':<12} {'mean':>9} {'bias':>9} {'emp.Var':>9} "
              f"{'cover95':>8} {'RMSE':>9}")
    print(header)
    print("-" * len(header))
    for k in keys:
        e = est[k]
        bias = e.mean() - TRUE_TAU
        emp_var = e.var(ddof=1)
        cover = np.mean(np.abs(e - TRUE_TAU) <= z * se[k])   # IF-based Wald CI, no bootstrap
        rmse = np.sqrt(np.mean((e - TRUE_TAU) ** 2))
        print(f"{k:<12} {e.mean():9.4f} {bias:9.4f} {emp_var:9.4f} "
              f"{cover:8.2f} {rmse:9.4f}")

    print("\nblip estimates (mean across reps):")
    print(f"  psi2  est = {psi2_all.mean(0)}   true = {TRUE_PSI2}")
    print(f"  psi1  est = {psi1_all.mean(0)}   true = {TRUE_PSI1}")

    return dict(R=R, n=n, keys=keys, est=est, se=se,
                psi2=psi2_all, psi1=psi1_all, elapsed=elapsed, true_tau=TRUE_TAU)


def summarize(res):
    """Per-estimator rows: (name, mean, bias, emp_var, cover95, rmse).

    cover95 is the fraction of reps whose influence-function Wald CI
    (tau_hat +/- 1.96 * IF-SE) covers the truth -- no bootstrap.
    """
    z = 1.959963985
    rows = []
    for k in res["keys"]:
        e = res["est"][k]
        s = res["se"][k]
        rows.append((
            k,
            e.mean(),
            e.mean() - res["true_tau"],
            e.var(ddof=1),
            float(np.mean(np.abs(e - res["true_tau"]) <= z * s)),
            float(np.sqrt(np.mean((e - res["true_tau"]) ** 2))),
        ))
    return rows


if __name__ == "__main__":
    run(R=100, n=2000)
