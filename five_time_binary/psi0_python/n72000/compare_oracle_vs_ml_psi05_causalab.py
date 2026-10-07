"""Compare psi05 across estimators AND nuisance-fitting approach (CausaLab cut).

Same as compare_oracle_vs_ml_psi05.py, but drops the Three-step-g estimator
for a CausaLab audience -- only Simple g-/Robins' and Weighted regression
(Three-step-ipw) are shown.

Loads both simulation_results.pkl files (oracle/: exact closed-form
nuisances; ML_nuisance/: RF/GBM/poly ensemble) and produces, per (p_I, ph)
cell, grouped points of Simple g- / Weighted regression, each shown twice
(Oracle vs Super-learner nuisances):

- psi05_oracle_vs_ml_causalab.pdf -- one page, point (median) + 95% CI
  error bar (median +/- 1.96 * robust SD) against the TRUE_PSI reference
  line. Shows bias and variance together in one read.
- psi05_oracle_vs_ml_table_causalab.tex -- a standalone, pdflatex-compilable
  table of the same cells' median/variance/MSE (booktabs), for a
  paper/report.

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
OUTPUT_PATH = os.path.join(HERE, "psi05_oracle_vs_ml_causalab.pdf")
OUTPUT_PATH_STEP1 = os.path.join(HERE, "psi05_oracle_vs_ml_causalab_step1.pdf")
OUTPUT_PATH_STEP2 = os.path.join(HERE, "psi05_oracle_vs_ml_causalab_step2.pdf")
OUTPUT_PATH_STEP3 = os.path.join(HERE, "psi05_oracle_vs_ml_causalab_step3.pdf")
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
    "Weighted regression estimator",
]

PLOT_DROP_PH = [0.5]

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


# Method colors: SNTTEM = blue, MSM-IPW = red. Oracle = solid circle, Super-learner = open triangle.
COLORS = {"Simple g-estimator": "#1F77B4",
          "Weighted regression estimator": "#D62728"}
DISPLAY_LABELS = {"Simple g-estimator": "SNTTEM",
                   "Weighted regression estimator": "MSM-IPW"}
SOURCES = ["Oracle", "Super-learner"]
ALPHA = {"Oracle": 0.9, "Super-learner": 0.9}
MARKER = {"Oracle": "o", "Super-learner": "^"}


FADE_ALPHA = 0.10
FADE_COLOR = "#CFCFCF"


def plot_point_ci(summary, pdf, pi_combos, ph_combos, highlight=None):
    """highlight: set of (row_i, col_i) panels drawn at full strength; every
    other panel is faded out (presentation build). None = all panels full."""
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
            faded = highlight is not None and (row_i, col_i) not in highlight
            text_color = FADE_COLOR if faded else "black"

            ax.axhline(TRUE_PSI, linestyle="--", color=FADE_COLOR if faded else "gray",
                       linewidth=1.0, zorder=1)

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
                        markersize=11, markeredgecolor="black", markeredgewidth=1.0,
                        elinewidth=3.2, capsize=7, capthick=3.2,
                        alpha=FADE_ALPHA if faded else ALPHA[source], zorder=3,
                    )

            if col_i == 0:
                ax.set_ylabel(_fmt(pi_vec, "pI"), fontsize=15, color=text_color)
            if row_i == 0:
                ax.set_title(_fmt(ph_vec, "ph"), fontsize=16, color=text_color)
            ax.set_xticks(x)
            ax.set_xlim(-0.5, len(METHODS) - 0.5)
            ax.set_xticklabels([DISPLAY_LABELS[m] for m in METHODS],
                               fontsize=12, rotation=0, ha="center")
            ax.tick_params(labelsize=13, colors=text_color)
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.15 if faded else 0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)

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
            rf"{p_i:g} & {ph:g} & {DISPLAY_LABELS[r['method']]} & {r['source']} & "
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
    # Figure only: drop the middle ph column to keep it readable (table keeps all cells).
    ph_combos = [ph_vec for ph_vec in ph_combos if ph_vec[0] not in PLOT_DROP_PH]

    with PdfPages(OUTPUT_PATH) as pdf:
        plot_point_ci(summary, pdf, pi_combos, ph_combos)
    print(f"\nSaved plot to {OUTPUT_PATH}")

    # Presentation build, step 1: only the top-left panel at full strength.
    with PdfPages(OUTPUT_PATH_STEP1) as pdf:
        plot_point_ci(summary, pdf, pi_combos, ph_combos, highlight={(0, 0)})
    print(f"Saved plot to {OUTPUT_PATH_STEP1}")

    # Presentation build, step 2: only the bottom-left panel at full strength.
    with PdfPages(OUTPUT_PATH_STEP2) as pdf:
        plot_point_ci(summary, pdf, pi_combos, ph_combos, highlight={(1, 0)})
    print(f"Saved plot to {OUTPUT_PATH_STEP2}")

    # Presentation build, step 3: the two right-hand panels at full strength.
    with PdfPages(OUTPUT_PATH_STEP3) as pdf:
        plot_point_ci(summary, pdf, pi_combos, ph_combos, highlight={(0, 1), (1, 1)})
    print(f"Saved plot to {OUTPUT_PATH_STEP3}")

    table_path = os.path.join(HERE, "psi05_oracle_vs_ml_table_causalab.tex")
    write_latex_table(summary, table_path)
    print(f"Saved LaTeX table to {table_path}")


if __name__ == "__main__":
    main()
