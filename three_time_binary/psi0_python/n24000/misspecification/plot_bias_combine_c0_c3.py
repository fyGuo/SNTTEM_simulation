"""Combine the c=0 and c=3 scenario figures into one large stacked figure.

Two subfigures (A: c=0, B: c=3), each the 3x2 grid (rows = psi03/psi13/psi23,
cols = p_I in {0.2, 1}) with the four blip scenarios cc/cw/wc/ww on the x-axis.
psi03 keeps a free y-axis; psi13/psi23 use fixed, common ranges.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = [("A", 0.0, r"$\delta = 0$ (not misspecified)"),
          ("B", 3.0, r"$\delta = 3$")]
OUT = os.path.join(HERE, "misspecification_bias_c0_c3.pdf")
TRUE_PSI = 0.0

METHOD_MAP = {"Three-step": "Three-step",
              "Robins' estimator": "Simple $g$",
              "GMM estimator": "GMM"}
METHODS = ["Three-step", "Simple $g$", "GMM"]
COLORS = {"Three-step": "#3B4992", "Simple $g$": "#EE0000", "GMM": "#008B45"}

PARAMS = [("est_psi03", r"$\widehat\psi_{03}$"),
          ("est_psi13", r"$\widehat\psi_{13}$"),
          ("est_psi23", r"$\widehat\psi_{23}$")]
YLIM = {"est_psi03": None, "est_psi13": (-1.2, 1.2), "est_psi23": (-1.15, 0.45)}

SCEN = [("correct", "correct"), ("correct", "wrong"),
        ("wrong", "correct"), ("wrong", "wrong")]
_C = {"correct": "C", "wrong": "W"}
SCEN_LABELS = [f"$\\gamma_{{13}}$:{_C[a]}\n$\\gamma_{{23}}$:{_C[b]}"
               for (a, b) in SCEN]


def draw_panel(subfig, letter, c, subtitle, df):
    """Draw the 3x2 grid for one additive constant c into `subfig`."""
    df = df.copy()
    df["method"] = df["method"].map(METHOD_MAP)
    pI_vals = sorted(df["p_I1"].unique())
    x = np.arange(len(SCEN))
    dx = 0.22

    axes = subfig.subplots(len(PARAMS), len(pI_vals), sharex=True, squeeze=False)
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
                ax.set_ylim(*YLIM[col])
            if r == 0:
                ax.set_title(f"$p_{{I1}}=p_{{I2}}={pI:g}$", fontsize=10)
            if ci == 0:  # y-label only on the left (p_I=0.2) column
                ax.set_ylabel(f"Bias of {plabel}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(SCEN_LABELS, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, axis="y", lw=0.4, alpha=0.5)

    subfig.suptitle(subtitle, fontsize=13, fontweight="bold", y=1.0)
    subfig.text(0.005, 1.0, letter, fontsize=20, fontweight="bold",
                ha="left", va="top")
    return axes[0, 0].get_legend_handles_labels()


def main():
    # A and B side by side (B to the right of A), with a shared legend on the
    # bottom and no overall figure title.
    fig = plt.figure(figsize=(8.4 * len(PANELS), 3.0 * len(PARAMS) + 0.8))
    body_sf, legend_sf = fig.subfigures(
        2, 1, height_ratios=[3.0 * len(PARAMS), 0.3], hspace=0.02)
    panel_sfs = body_sf.subfigures(1, len(PANELS), wspace=0.04)

    handles = labels = None
    for sf, (letter, c, subtitle) in zip(panel_sfs, PANELS):
        df = pd.read_pickle(os.path.join(HERE, f"simulation_results_add{c}.pkl"))
        handles, labels = draw_panel(sf, letter, c, subtitle, df)

    legend_sf.legend(handles, labels, loc="center", ncol=len(METHODS),
                     fontsize=11, frameon=False)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
