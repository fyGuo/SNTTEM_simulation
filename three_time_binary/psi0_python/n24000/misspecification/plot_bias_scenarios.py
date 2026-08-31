"""Bias of the three estimators across the four blip-misspecification scenarios.

Under the null (true psi = 0) empirical bias = mean(estimate) - 0. Dots show the
Monte Carlo mean bias; error bars span the 2.5th-97.5th Monte Carlo percentiles.

Layout: rows = parameter (psi03, psi13, psi23), columns = the two settings
(p_I1 = p_I2 in {0.2, 1}); x-axis = the 4 scenarios (gamma13, gamma23 each
correct/wrong); colour = estimator.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "simulation_results.pkl")
OUT = os.path.join(HERE, "misspecification_bias.pdf")

# Estimator display names + colours (match the n24000 report figures).
METHOD_MAP = {
    "Three-step": "Three-step",
    "Robins' estimator": "Simple $g$",
    "GMM estimator": "GMM",
}
METHODS = ["Three-step", "Simple $g$", "GMM"]
COLORS = {"Three-step": "#3B4992", "Simple $g$": "#EE0000", "GMM": "#008B45"}

PARAMS = [("est_psi03", r"$\widehat\psi_{03}$"),
          ("est_psi13", r"$\widehat\psi_{13}$"),
          ("est_psi23", r"$\widehat\psi_{23}$")]

# Scenario order and two-line x labels (gamma13 / gamma23, C=correct, W=wrong).
SCEN = [("correct", "correct"), ("correct", "wrong"),
        ("wrong", "correct"), ("wrong", "wrong")]
def _slabel(g13, g23):
    c = {"correct": "C", "wrong": "W"}
    return f"$\\gamma_{{13}}$:{c[g13]}\n$\\gamma_{{23}}$:{c[g23]}"
SCEN_LABELS = [_slabel(*s) for s in SCEN]

TRUE_PSI = 0.0


def main():
    df = pd.read_pickle(RESULTS)
    df = df.copy()
    df["method"] = df["method"].map(METHOD_MAP)

    pI_vals = sorted(df["p_I1"].unique())
    x = np.arange(len(SCEN))
    dx = 0.22  # horizontal offset between estimators

    fig, axes = plt.subplots(len(PARAMS), len(pI_vals),
                             figsize=(4.2 * len(pI_vals), 3.0 * len(PARAMS)),
                             sharex=True, sharey="row", squeeze=False)

    for r, (col, plabel) in enumerate(PARAMS):
        for c, pI in enumerate(pI_vals):
            ax = axes[r, c]
            ax.axhline(0, color="0.5", lw=0.8, zorder=1)
            for mi, m in enumerate(METHODS):
                means, lo_len, hi_len = [], [], []
                for (g13, g23) in SCEN:
                    vals = df[(df.p_I1 == pI) & (df.method == m)
                              & (df.gamma13 == g13) & (df.gamma23 == g23)][col]
                    v = vals.dropna().values
                    center = v.mean() - TRUE_PSI
                    # Error bars: 2.5th-97.5th Monte Carlo percentiles (bias scale).
                    lo = np.percentile(v, 2.5) - TRUE_PSI
                    hi = np.percentile(v, 97.5) - TRUE_PSI
                    means.append(center)
                    lo_len.append(max(center - lo, 0.0))
                    hi_len.append(max(hi - center, 0.0))
                off = (mi - (len(METHODS) - 1) / 2) * dx
                ax.errorbar(x + off, means, yerr=[lo_len, hi_len], fmt="o", ms=4,
                            color=COLORS[m], capsize=2, lw=1, label=m, zorder=3)
            if r == 0:
                ax.set_title(f"$p_{{I1}}=p_{{I2}}={pI:g}$", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"Bias of {plabel}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(SCEN_LABELS, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, axis="y", lw=0.4, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Empirical bias by blip-misspecification scenario "
                 "(true $\\psi=0$, $n=24000$, 300 reps)",
                 fontsize=12, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
