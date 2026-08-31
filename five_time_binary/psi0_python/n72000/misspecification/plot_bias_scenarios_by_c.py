"""One bias subfigure per additive constant c in {0, 1.5, 3}.

For each c, reads simulation_results_add{c}.pkl and plots the four blip scenarios
cc / cw / wc / ww (correct/wrong for gamma13, gamma23). Dots are the Monte Carlo
mean bias; error bars span the 2.5th-97.5th Monte Carlo percentiles. Rows =
parameter (psi03, psi13, psi23), columns = setting (p_I1 = p_I2 in {0.2, 1}),
colour = estimator. Writes misspecification_bias_c{c}.pdf per c.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ADDS = [0.0, 3.0]
TRUE_PSI = 0.0

# Per-figure title (additive constant delta). delta = 0 is the absorbable case.
SUBTITLE = {0.0: r"$\delta = 0$ (not misspecified)",
            1.5: r"$\delta = 1.5$",
            3.0: r"$\delta = 3$"}

# psi03 keeps a free (autoscaled) y-axis -- Simple g diverges there at delta=3.
# psi13/psi23 use a FIXED y-range, common across both settings and both figures,
# so the bias is directly comparable. (They never diverge.)
YLIM = {"est_psi03": None, "est_psi13": (-1.2, 1.2), "est_psi23": (-1.15, 0.45)}

METHOD_MAP = {"Three-step": "Three-step",
              "Robins' estimator": "Simple $g$",
              "GMM estimator": "GMM"}
METHODS = ["Three-step", "Simple $g$", "GMM"]
COLORS = {"Three-step": "#3B4992", "Simple $g$": "#EE0000", "GMM": "#008B45"}

PARAMS = [("est_psi03", r"$\widehat\psi_{03}$"),
          ("est_psi13", r"$\widehat\psi_{13}$"),
          ("est_psi23", r"$\widehat\psi_{23}$")]

# Scenario order and two-line (gamma13 / gamma23) labels, matching
# misspecification_bias.pdf: C = correct, W = wrong.
SCEN = [("correct", "correct"), ("correct", "wrong"),
        ("wrong", "correct"), ("wrong", "wrong")]
_C = {"correct": "C", "wrong": "W"}
SCEN_LABELS = [f"$\\gamma_{{13}}$:{_C[a]}\n$\\gamma_{{23}}$:{_C[b]}"
               for (a, b) in SCEN]


def make_figure(c, df, out):
    df = df.copy()
    df["method"] = df["method"].map(METHOD_MAP)
    pI_vals = sorted(df["p_I1"].unique())
    x = np.arange(len(SCEN))
    dx = 0.22

    # Independent y-axes so every panel shows its own full range (the p_I=1
    # panels are much tighter than p_I=0.2 and would be compressed if shared).
    fig, axes = plt.subplots(len(PARAMS), len(pI_vals),
                             figsize=(4.2 * len(pI_vals), 3.0 * len(PARAMS)),
                             sharex=True, sharey=False, squeeze=False)

    for r, (col, plabel) in enumerate(PARAMS):
        for ci, pI in enumerate(pI_vals):
            ax = axes[r, ci]
            ax.axhline(0, color="0.5", lw=0.8, zorder=1)
            for mi, m in enumerate(METHODS):
                means, lo_len, hi_len = [], [], []
                for (g13, g23) in SCEN:
                    v = df[(df.p_I1 == pI) & (df.method == m)
                           & (df.gamma13 == g13) & (df.gamma23 == g23)][col]
                    v = v.dropna().values
                    center = v.mean() - TRUE_PSI
                    lo = np.percentile(v, 2.5) - TRUE_PSI
                    hi = np.percentile(v, 97.5) - TRUE_PSI
                    means.append(center)
                    lo_len.append(max(center - lo, 0.0))
                    hi_len.append(max(hi - center, 0.0))
                off = (mi - (len(METHODS) - 1) / 2) * dx
                ax.errorbar(x + off, means, yerr=[lo_len, hi_len], fmt="o", ms=4,
                            color=COLORS[m], capsize=2, lw=1, label=m, zorder=3)
            if YLIM[col] is not None:
                ax.set_ylim(*YLIM[col])   # fixed range for psi13/psi23
            if r == 0:
                ax.set_title(f"$p_{{I1}}=p_{{I2}}={pI:g}$", fontsize=11)
            if ci == 0:  # y-label only on the left (p_I=0.2) column
                ax.set_ylabel(f"Bias of {plabel}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(SCEN_LABELS, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, axis="y", lw=0.4, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # delta title on top, shared legend on the bottom.
    fig.suptitle(SUBTITLE.get(c, f"$\\delta = {c:g}$"),
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    for c in ADDS:
        df = pd.read_pickle(os.path.join(HERE, f"simulation_results_add{c}.pkl"))
        make_figure(c, df, os.path.join(HERE, f"misspecification_bias_c{c}.pdf"))
