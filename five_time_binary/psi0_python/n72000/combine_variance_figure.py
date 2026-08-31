"""Combine the five per-psi variance panels (2.5/97.5 method) into one large
figure with sub-titled subfigures A, B, C, D, E.

Reuses the summary logic from make_report_pdfs.py. Layout of each subfigure
matches the standalone panels: grouped bars (x = the ph setting, color =
method), rows = (p_I1..p_I4) combos.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from make_report_pdfs import (
    load_results, rename_methods, summarize, METHODS, COLORS,
    P_I_COLS, PH_COLS, fmt_vec, cell_combos, _mask_for, bar_value_labels,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Match the n24k filename stem used by the standalone panels.
OUTPUT_PATH = os.path.join(HERE, "simulation_results_n24k_plots.pdf")

PANELS = [
    ("A", "var_psi05", r"Empirical variance of $\psi_{05}$"),
    ("B", "var_psi15", r"Empirical variance of $\psi_{15}$"),
    ("C", "var_psi25", r"Empirical variance of $\psi_{25}$"),
    ("D", "var_psi35", r"Empirical variance of $\psi_{35}$"),
    ("E", "var_psi45", r"Empirical variance of $\psi_{45}$"),
]


def build_combined(summary, out):
    pi_combos = cell_combos(summary, P_I_COLS)
    ph_combos = cell_combos(summary, PH_COLS)
    n_rows = len(pi_combos)
    color_map = {m: COLORS[i] for i, m in enumerate(METHODS)}
    x = np.arange(len(ph_combos))
    bar_width = 0.8 / len(METHODS)

    fig = plt.figure(figsize=(1.9 * len(ph_combos) + 3,
                              3.2 * n_rows * len(PANELS) + 1.2))
    # Thin top strip for the shared legend, then one subfigure per panel.
    legend_sf, *panel_sfs = fig.subfigures(
        len(PANELS) + 1, 1, hspace=0.05,
        height_ratios=[0.35] + [3.2 * n_rows] * len(PANELS),
    )

    legend_handles = None
    for sf, (letter, metric, title) in zip(panel_sfs, PANELS):
        axes = sf.subplots(n_rows, 1, sharex=True, sharey=True, squeeze=False)
        for row_i, pi_vec in enumerate(pi_combos):
            ax = axes[row_i, 0]
            sub = summary[_mask_for(summary, P_I_COLS, pi_vec)]
            for m_i, m in enumerate(METHODS):
                msub = sub[sub["method"] == m]
                heights = []
                for ph_vec in ph_combos:
                    v = msub.loc[_mask_for(msub, PH_COLS, ph_vec), metric].values
                    heights.append(v[0] if len(v) else np.nan)
                offsets = x + (m_i - (len(METHODS) - 1) / 2) * bar_width
                bars = ax.bar(offsets, heights, width=bar_width,
                              label=m, color=color_map[m], alpha=0.85)
                bar_value_labels(ax, bars, heights)
            ax.set_ylabel(fmt_vec(pi_vec, "p_I"), fontsize=9)
            if row_i == n_rows - 1:
                ax.set_xlabel("ph", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([fmt_vec(v) for v in ph_combos], fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

        if legend_handles is None:
            legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()

        # Sub-title with the panel letter (A/B/C/D/E) at the left.
        sf.suptitle(f"{title}", fontsize=14, fontweight="bold", y=1.0)
        sf.text(0.005, 1.0, letter, fontsize=20, fontweight="bold",
                ha="left", va="top")

    legend_sf.legend(legend_handles, legend_labels, loc="center",
                     ncol=len(METHODS), fontsize=11, frameon=False)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined figure to {out}")


if __name__ == "__main__":
    df = load_results()
    df = rename_methods(df)
    df = df[df["method"].isin(METHODS)]
    summary = summarize(df, robust=True, var_method="pct")
    out = OUTPUT_PATH.replace(".pdf", "_variance_2.5-97.5_ABC.pdf")
    build_combined(summary, out)
