"""Combine the three per-psi variance panels (2.5/97.5 method) into one large
figure with sub-titled subfigures A, B, C.

Reuses the summary logic from make_report_pdfs.py. Layout of each subfigure
matches the standalone panels: grouped bars (x = ph1, color = method),
rows = (p_I1, p_I2) combos, cols = ph2.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from make_report_pdfs import (
    load_results, rename_methods, summarize, METHODS, COLORS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Match the n24k filename stem used by the standalone panels.
OUTPUT_PATH = os.path.join(HERE, "simulation_results_n24k_plots.pdf")

PANELS = [
    ("A", "var_psi03", r"Empirical variance of $\psi_{03}$"),
    ("B", "var_psi13", r"Empirical variance of $\psi_{13}$"),
    ("C", "var_psi23", r"Empirical variance of $\psi_{23}$"),
]


def build_combined(summary, out):
    pi_combos = sorted(
        summary[["p_I1", "p_I2"]].drop_duplicates().itertuples(index=False, name=None)
    )
    ph2_vals = sorted(summary["ph2"].unique())
    ph1_vals = sorted(summary["ph1"].unique())
    n_rows, n_cols = len(pi_combos), len(ph2_vals)
    color_map = {m: COLORS[i] for i, m in enumerate(METHODS)}
    x = np.arange(len(ph1_vals))
    bar_width = 0.8 / len(METHODS)

    fig = plt.figure(figsize=(4 * n_cols, 3.2 * n_rows * len(PANELS) + 1.2))
    # Thin top strip for the shared legend, then one subfigure per panel.
    legend_sf, *panel_sfs = fig.subfigures(
        len(PANELS) + 1, 1, hspace=0.05,
        height_ratios=[0.35] + [3.2 * n_rows] * len(PANELS),
    )

    legend_handles = None
    for sf, (letter, metric, title) in zip(panel_sfs, PANELS):
        axes = sf.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
        for row_i, (pI1, pI2) in enumerate(pi_combos):
            for col_i, ph2 in enumerate(ph2_vals):
                ax = axes[row_i, col_i]
                sub = summary[
                    (summary["p_I1"] == pI1) & (summary["p_I2"] == pI2)
                    & (summary["ph2"] == ph2)
                ]
                for m_i, m in enumerate(METHODS):
                    msub = (sub[sub["method"] == m]
                            .set_index("ph1").reindex(ph1_vals))
                    offsets = x + (m_i - (len(METHODS) - 1) / 2) * bar_width
                    heights = msub[metric].values
                    bars = ax.bar(offsets, heights, width=bar_width,
                                  label=m, color=color_map[m], alpha=0.85)
                    vlabels = ["" if not np.isfinite(h) else f"{h:.3g}"
                               for h in heights]
                    ax.bar_label(bars, labels=vlabels, fontsize=5,
                                 rotation=0, padding=1.5)
                if col_i == 0:
                    ax.set_ylabel(f"p_I1={pI1}, p_I2={pI2}", fontsize=9)
                if row_i == 0:
                    ax.set_title(f"ph2 = {ph2}", fontsize=10)
                if row_i == n_rows - 1:
                    ax.set_xlabel("ph1", fontsize=9)
                ax.set_xticks(x)
                ax.set_xticklabels([str(v) for v in ph1_vals], fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

        if legend_handles is None:
            legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()

        # Sub-title with the panel letter (A/B/C) at the left.
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
