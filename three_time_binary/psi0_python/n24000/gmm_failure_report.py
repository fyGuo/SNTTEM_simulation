"""Characterize the GMM estimator's distribution in its failure cells.

The GMM combination inverts a bootstrap covariance that becomes near-singular in
low-information scenarios (low ph / low eligibility); the resulting weights
explode and produce divergent psi03 estimates. This script reports, for every
cell where the GMM failure rate (|est_psi03| > FAIL_THRESH) exceeds FAIL_RATE_MAX:

  * a text summary (failure rate, sign split, magnitude, clean-core IQR), and
  * a symlog histogram (so both the near-zero core and the extreme tails show).

Outputs: gmm_failure_distribution.pdf  (and prints the table).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAIL_THRESH = 10.0
FAIL_RATE_MAX = 0.05
PARAM = "est_psi03"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "simulation_results.pkl")
OUT = os.path.join(HERE, "gmm_failure_distribution.pdf")


def main():
    df = pd.read_pickle(RESULTS)
    gmm = df[df["method"] == "GMM estimator"]

    cells = []
    for key, sub in gmm.groupby(["p_I1", "p_I2", "ph1", "ph2"]):
        x = sub[PARAM].dropna()
        rate = (x.abs() > FAIL_THRESH).mean()
        if rate > FAIL_RATE_MAX:
            cells.append((key, x))

    # ---- Text report ----
    print(f"GMM failure cells (|{PARAM}| > {FAIL_THRESH:.0f} in "
          f">{FAIL_RATE_MAX:.0%} of reps):")
    header = (f"{'p_I1,p_I2,ph1,ph2':22s} {'n':>4s} {'fail%':>6s} "
              f"{'neg/pos':>9s} {'min':>11s} {'max':>11s} {'core P25..P75':>16s}")
    print(header)
    for key, x in cells:
        fails = x[x.abs() > FAIL_THRESH]
        core = x[x.abs() <= FAIL_THRESH]
        rate = len(fails) / len(x)
        print(f"{str(key):22s} {len(x):4d} {rate*100:5.0f}% "
              f"{f'{(fails<0).sum()}/{(fails>0).sum()}':>9s} "
              f"{x.min():11.1f} {x.max():11.1f} "
              f"  [{core.quantile(.25):.2f},{core.quantile(.75):.2f}]")

    if not cells:
        print("No GMM failure cells.")
        return

    # ---- Symlog histograms (one panel per failure cell) ----
    n = len(cells)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    # symmetric-log bin edges: log-spaced tails + linear core, so the near-zero
    # cluster and the extreme divergences are both resolved (not one block).
    pos = np.logspace(np.log10(FAIL_THRESH), 7, 22)   # 10 .. 1e7
    core = np.linspace(-FAIL_THRESH, FAIL_THRESH, 21)
    bins = np.unique(np.concatenate([-pos[::-1], core, pos]))
    for ax, (key, x) in zip(axes, cells):
        pI1, pI2, ph1, ph2 = key
        ax.hist(x.values, bins=bins, color="#008B45", alpha=0.85)
        ax.set_xscale("symlog", linthresh=FAIL_THRESH)
        ax.axvspan(-FAIL_THRESH, FAIL_THRESH, color="0.85", zorder=0)
        rate = (x.abs() > FAIL_THRESH).mean()
        ax.set_title(f"p_I1={pI1}, p_I2={pI2}, ph1={ph1}, ph2={ph2}\n"
                     f"fail rate {rate:.0%}  (shaded = |est|<=10 core)",
                     fontsize=9)
        ax.set_xlabel(r"$\hat\psi_{03}$ (symlog)", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("GMM estimator distribution in failure cells "
                 "(bimodal: clean core near 0 + divergent tails)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
