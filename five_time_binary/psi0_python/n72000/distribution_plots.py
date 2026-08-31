"""QQ plots and histograms of the estimator distributions, per cell.

Same facet grid as the variance/bias reports: rows = (p_I1..p_I4), cols = the
ph setting, one page per parameter. Within each panel the methods are overlaid.
Divergent estimates (|est| > FAIL_THRESH) are excluded for display -- they are
numerical failures, summarized separately in the failure-proportion report;
here we want the shape of the working estimator.

Outputs (multipage, one page per parameter):
  simulation_results_plots_qq.pdf
  simulation_results_plots_histogram.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from make_report_pdfs import (
    load_results, rename_methods, METHODS, COLORS, PARAMS, FAIL_THRESH,
    P_I_COLS, PH_COLS, fmt_vec, cell_combos,
)

HERE = os.path.dirname(os.path.abspath(__file__))
QQ_OUT = os.path.join(HERE, "simulation_results_plots_qq.pdf")
HIST_OUT = os.path.join(HERE, "simulation_results_plots_histogram.pdf")
# Estimates with |est| > DISPLAY_THRESH are excluded from the display.
# Set to float("inf") to plot ALL values (no restriction).
DISPLAY_THRESH = FAIL_THRESH

COLOR_MAP = {m: COLORS[i] for i, m in enumerate(METHODS)}
PARAM_LABEL = {"est_psi05": r"\psi_{05}", "est_psi15": r"\psi_{15}",
               "est_psi25": r"\psi_{25}", "est_psi35": r"\psi_{35}",
               "est_psi45": r"\psi_{45}"}


def _robust_z(x):
    """Standardize by median and IQR-based SD (robust to residual heavy tails)."""
    med = np.median(x)
    sd = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    return (x - med) / sd if sd > 0 else x - med


def _cell_mask(df, cols, vec):
    mask = np.ones(len(df), dtype=bool)
    for col, val in zip(cols, vec):
        mask &= (df[col] == val).values
    return mask


def _grid(df, param, kind, pdf):
    pi = cell_combos(df, P_I_COLS)
    ph = cell_combos(df, PH_COLS)
    fig, axes = plt.subplots(len(pi), len(ph),
                             figsize=(4 * len(ph), 3.3 * len(pi)),
                             constrained_layout=True, squeeze=False)
    for r, pi_vec in enumerate(pi):
        for c, ph_vec in enumerate(ph):
            ax = axes[r, c]
            cell = df[_cell_mask(df, P_I_COLS, pi_vec)
                      & _cell_mask(df, PH_COLS, ph_vec)]
            for m in METHODS:
                x = cell.loc[cell.method == m, param].dropna().values
                x = x[np.abs(x) <= DISPLAY_THRESH]   # |est|>thresh excluded (inf = all)
                if len(x) < 5:
                    continue
                if kind == "qq":
                    z = np.sort(_robust_z(x))
                    theo = norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
                    ax.plot(theo, z, ".", ms=3, color=COLOR_MAP[m], alpha=0.6,
                            label=m)
                else:  # histogram
                    ax.hist(x, bins=25, density=True, histtype="stepfilled",
                            color=COLOR_MAP[m], alpha=0.35, label=m)
                    ax.hist(x, bins=25, density=True, histtype="step",
                            color=COLOR_MAP[m], lw=1.2)
            if kind == "qq":
                lim = 3.2
                ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                if r == len(pi) - 1:
                    ax.set_xlabel("theoretical quantiles", fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"{fmt_vec(pi_vec, 'p_I')}\nsample (robust z)",
                                  fontsize=8)
            else:
                ax.axvline(0.0, color="black", ls="--", lw=0.7)
                if r == len(pi) - 1:
                    ax.set_xlabel("estimate", fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"{fmt_vec(pi_vec, 'p_I')}\ndensity", fontsize=8)
            if r == 0:
                ax.set_title(fmt_vec(ph_vec, "ph"), fontsize=10)
            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.3, alpha=0.5)

    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
                   fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=False)
    name = "QQ plot" if kind == "qq" else "Histogram"
    note = (rf"; $|\hat\psi|\leq {DISPLAY_THRESH:g}$"
            if np.isfinite(DISPLAY_THRESH) else "; all values")
    fig.suptitle(rf"{name} of ${PARAM_LABEL[param]}$  ({note.lstrip('; ')})",
                 fontsize=14, fontweight="bold", y=1.06)
    pdf.savefig(fig, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_results()
    df = rename_methods(df)
    df = df[df["method"].isin(METHODS)]

    for out, kind in [(QQ_OUT, "qq"), (HIST_OUT, "hist")]:
        with PdfPages(out) as pdf:
            for param in PARAMS:
                _grid(df, param, kind, pdf)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
