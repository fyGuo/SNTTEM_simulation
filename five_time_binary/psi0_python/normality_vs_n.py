"""Does asymptotic normality return at ph1=ph2=0.2 as n grows?

At ph=0.2 the estimating equations up-weight the few "stayers" (1/0.2 = 5) at
both times, so the effective sample size is far below n -- which is why the
estimator is skewed/heavy-tailed at n=6000 even though it is asymptotically
normal. This sweeps n and tracks skewness/kurtosis and the QQ shape for the
Three-step and Simple g-estimators (bootstrap/GMM skipped -- not needed here).

Outputs: normality_vs_n.pdf  (QQ grid: rows=method, cols=n) and a printed table.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import skew, kurtosis, norm
from joblib import Parallel, delayed

from generate_data import generate_data
from working_models_ml import working_model
from estimators import ee_three_step, ee_robins, m_estimate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "normality_vs_n.pdf")

N_VALUES = [6000, 12000, 24000, 48000]
N_ITER = 300
SEED = 3411
# Fixed scenario: high switching, full eligibility (isolates the ph=0.2 effect).
P_I1 = P_I2 = 1.0
PH1 = PH2 = 0.2
CAP = 10.0  # exclude |est| > CAP (numerical failures) for the shape analysis

COLORS = {"Three-step": "#3B4992", "Simple g": "#EE0000"}


def _one(i, n):
    rng = np.random.default_rng([SEED, n, i])
    df = generate_data(n, psi03=0, psi13=0, psi23=0,
                       p_I1=P_I1, p_I2=P_I2, ph1=PH1, ph2=PH2, rng=rng)
    idx = rng.permutation(n)
    id1, id2 = idx[: n // 2], idx[n // 2:]
    df = working_model(df, id1, id2)
    old = m_estimate(ee_three_step, df)
    rbs = m_estimate(ee_robins, df)
    return old[0], rbs[0]   # psi03 only


def main():
    results = {}
    for n in N_VALUES:
        ests = Parallel(n_jobs=-1, verbose=1)(
            delayed(_one)(i, n) for i in range(1, N_ITER + 1)
        )
        arr = np.array(ests)
        results[n] = {"Three-step": arr[:, 0], "Simple g": arr[:, 1]}
        print(f"n={n} done")

    # ---- Table ----
    print(f"\nShape of psi03 at ph1=ph2={PH1}, p_I1=p_I2={P_I1} "
          f"(|est|<={CAP:.0f}), {N_ITER} reps:")
    print(f"{'n':>7s} {'method':12s} {'kept':>5s} {'skew':>7s} {'exKurt':>7s} "
          f"{'var':>8s}")
    for n in N_VALUES:
        for m in ("Three-step", "Simple g"):
            x = results[n][m]
            x = x[np.abs(x) <= CAP]
            print(f"{n:7d} {m:12s} {len(x):5d} {skew(x):7.2f} "
                  f"{kurtosis(x):7.2f} {np.var(x):8.3f}")

    # ---- QQ grid: rows = method, cols = n ----
    methods = ["Three-step", "Simple g"]
    fig, axes = plt.subplots(len(methods), len(N_VALUES),
                             figsize=(3.4 * len(N_VALUES), 3.2 * len(methods)),
                             squeeze=False)
    for r, m in enumerate(methods):
        for c, n in enumerate(N_VALUES):
            ax = axes[r, c]
            x = results[n][m]
            x = x[np.abs(x) <= CAP]
            z = (x - np.median(x)) / ((np.quantile(x, .75)
                                       - np.quantile(x, .25)) / 1.349)
            z = np.sort(z)
            theo = norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
            ax.plot(theo, z, ".", ms=3, color=COLORS[m], alpha=0.6)
            ax.plot([-3.2, 3.2], [-3.2, 3.2], "k--", lw=0.7)
            ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
            sk = skew(x)
            if r == 0:
                ax.set_title(f"n = {n}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{m}\nsample (robust z)", fontsize=9)
            if r == len(methods) - 1:
                ax.set_xlabel("theoretical quantiles", fontsize=8)
            ax.text(0.05, 0.92, f"skew={sk:.2f}", transform=ax.transAxes,
                    fontsize=8, va="top")
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)
    fig.suptitle(rf"QQ of $\psi_{{03}}$ vs $n$ at ph1=ph2={PH1}, "
                 rf"p_I1=p_I2={P_I1}  ($|\hat\psi|\leq{CAP:.0f}$)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    with PdfPages(OUT) as pdf:
        pdf.savefig(fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
