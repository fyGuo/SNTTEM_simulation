"""Compare psi05 across estimators AND nuisance-fitting approach.

Loads both simulation_results.pkl files (oracle/: exact closed-form
nuisances; ML_nuisance/: RF/GBM/poly ensemble) and produces, per (p_I, ph)
cell, grouped points of Three-step-g / Weighted regression / Robins', each
shown twice (Oracle vs Super-learner nuisances):

- psi05_oracle_vs_ml.pdf -- one page, point (median) + 95% CI error bar
  (median +/- 1.96 * robust SD) against the TRUE_PSI reference line. Shows
  bias and variance together in one read.
- psi05_oracle_vs_ml_table.tex -- a standalone, pdflatex-compilable table
  of the same cells' median/variance/MSE (booktabs), for a paper/report.

ML_nuisance/simulation_results.pkl is the post-ee_three_step_ipw-fix rerun
(completed 2026-09-03; see CLAUDE.md) -- Three-step-ipw here reflects the
corrected equation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(HERE, "psi05_oracle_vs_ml.pdf")
TRUE_PSI = 0.0
PSI_PARAM = "psi05"

P_I_COLS = ["p_I1", "p_I2", "p_I3", "p_I4"]
PH_COLS = ["ph1", "ph2", "ph3", "ph4"]

METHOD_MAP = {
    "Three-step-g": "Three-step-g estimator",
    "Three-step-ipw": "Weighted regression estimator",
    "Robins' estimator": "Simple g-estimator",
}
METHODS = [
    "Simple g-estimator",
    "Three-step-g estimator",
    "Weighted regression estimator",
]

FAIL_THRESH = 10.0
FAIL_RATE_MAX = 0.05


def load(source_label, path):
    df = pd.read_pickle(path)
    df = df.copy()
    df["method"] = df["method"].map(METHOD_MAP).fillna(df["method"])
    df["source"] = source_label
    return df


def _robust_sd(x):
    return (x.quantile(0.975) - x.quantile(0.025)) / 3.96


def summarize(df, psi_param, true_psi=TRUE_PSI):
    est_col = f"est_{psi_param}"

    def dropped(x):
        return (x.abs() > FAIL_THRESH).mean() > FAIL_RATE_MAX

    def center(x):
        return np.nan if dropped(x) else x.median()

    def var(x):
        return np.nan if dropped(x) else _robust_sd(x) ** 2

    def mse(x):
        return np.nan if dropped(x) else (x.median() - true_psi) ** 2 + _robust_sd(x) ** 2

    def coverage(x):
        if dropped(x):
            return np.nan
        sd = _robust_sd(x)
        return np.mean((x - 1.96 * sd < true_psi) & (x + 1.96 * sd > true_psi))

    def sd(x):
        return np.nan if dropped(x) else _robust_sd(x)

    return (
        df.groupby(P_I_COLS + PH_COLS + ["method", "source"])
        .agg(mean=(est_col, center), var=(est_col, var), sd=(est_col, sd),
             mse=(est_col, mse), coverage=(est_col, coverage))
        .reset_index()
    )


def _fmt(values, name):
    vals = list(values)
    if all(v == vals[0] for v in vals):
        return f"{name}={vals[0]:g}"
    return f"{name}=({', '.join(f'{v:g}' for v in vals)})"


# Method colors (matching check_results.py); Oracle = solid circle, Super-learner = open triangle.
COLORS = {"Simple g-estimator": "#3B4992",
          "Three-step-g estimator": "#EE0000",
          "Weighted regression estimator": "#008B45"}
SOURCES = ["Oracle", "Super-learner"]
ALPHA = {"Oracle": 0.9, "Super-learner": 0.55}
MARKER = {"Oracle": "o", "Super-learner": "^"}


def plot_point_ci(summary, pdf, pi_combos, ph_combos):
    n_rows, n_cols = len(pi_combos), len(ph_combos)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.4 * n_rows),
        sharey=True, constrained_layout=True, squeeze=False,
    )
    x = np.arange(len(METHODS))
    offsets = {"Oracle": -0.12, "Super-learner": 0.12}

    for row_i, pi_vec in enumerate(pi_combos):
        for col_i, ph_vec in enumerate(ph_combos):
            ax = axes[row_i, col_i]
            mask = np.ones(len(summary), dtype=bool)
            for col, val in zip(P_I_COLS, pi_vec):
                mask &= (summary[col] == val).values
            for col, val in zip(PH_COLS, ph_vec):
                mask &= (summary[col] == val).values
            sub = summary[mask]

            ax.axhline(TRUE_PSI, linestyle="--", color="gray", linewidth=1.0, zorder=1)

            for s_i, source in enumerate(SOURCES):
                for m_i, m in enumerate(METHODS):
                    row = sub[(sub["method"] == m) & (sub["source"] == source)]
                    if row.empty or not np.isfinite(row["mean"].values[0]):
                        continue
                    median = row["mean"].values[0]
                    ci = 1.96 * row["sd"].values[0]
                    ax.errorbar(
                        x[m_i] + offsets[source], median, yerr=ci,
                        fmt=MARKER[source], color=COLORS[m],
                        markersize=9, markeredgecolor="black", markeredgewidth=0.7,
                        elinewidth=1.8, capsize=5, capthick=1.8,
                        alpha=ALPHA[source], zorder=3,
                    )

            if col_i == 0:
                ax.set_ylabel(_fmt(pi_vec, "pI"), fontsize=15)
            if row_i == 0:
                ax.set_title(_fmt(ph_vec, "ph"), fontsize=16)
            ax.set_xticks(x)
            ax.set_xlim(-0.5, len(METHODS) - 0.5)
            ax.set_xticklabels(METHODS, fontsize=12, rotation=20, ha="right")
            ax.tick_params(labelsize=13)
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=MARKER[s], color="black", linestyle="",
               markersize=9, alpha=ALPHA[s], label=s)
        for s in SOURCES
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.995, 0.93), fontsize=13, framealpha=0.9)
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(summary, path):
    """Write a standalone, pdflatex-compilable table of median/variance/MSE."""
    rows = summary.sort_values(
        by=P_I_COLS + PH_COLS + ["method", "source"],
        key=lambda s: s if s.name != "source" else s.map({"Oracle": 0, "Super-learner": 1}),
    )

    g = r"\psi_{05}"
    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Empirical median, variance, and MSE of ${g}$ --- Oracle vs Super-learner nuisances}}",
        r"\begin{tabular}{ll l l r r r}",
        r"\toprule",
        r"pI & $ph$ & Method & Source & Median & Variance & MSE \\",
        r"\midrule",
    ]
    for _, r in rows.iterrows():
        p_i = r[P_I_COLS[0]]
        ph = r[PH_COLS[0]]
        median = "" if not np.isfinite(r["mean"]) else f"{r['mean']:.3f}"
        var = "" if not np.isfinite(r["var"]) else f"{r['var']:.3f}"
        mse = "" if not np.isfinite(r["mse"]) else f"{r['mse']:.3f}"
        lines.append(
            rf"{p_i:g} & {ph:g} & {r['method']} & {r['source']} & "
            rf"{median} & {var} & {mse} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
        r"\end{document}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    df_oracle = load("Oracle", os.path.join(HERE, "oracle", "simulation_results.pkl"))
    df_ml = load("Super-learner", os.path.join(HERE, "ML_nuisance", "simulation_results.pkl"))
    df = pd.concat([df_oracle, df_ml], ignore_index=True)
    df = df[df["method"].isin(METHODS)]

    summary = summarize(df, PSI_PARAM)
    print(f"===== {PSI_PARAM}: Oracle vs Super-learner nuisances =====")
    print(summary.to_string(index=False))

    pi_combos = sorted(df[P_I_COLS].drop_duplicates().itertuples(index=False, name=None))
    ph_combos = sorted(df[PH_COLS].drop_duplicates().itertuples(index=False, name=None))

    with PdfPages(OUTPUT_PATH) as pdf:
        plot_point_ci(summary, pdf, pi_combos, ph_combos)
    print(f"\nSaved plot to {OUTPUT_PATH}")

    table_path = os.path.join(HERE, "psi05_oracle_vs_ml_table.tex")
    write_latex_table(summary, table_path)
    print(f"Saved LaTeX table to {table_path}")


if __name__ == "__main__":
    main()
