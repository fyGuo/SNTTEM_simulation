"""Compare psi05 across estimators AND nuisance-fitting approach.

Loads both simulation_results.pkl files (oracle/: exact closed-form
nuisances; ML_nuisance/: RF/GBM/poly ensemble) and plots, per (p_I, ph) cell,
grouped bars of Three-step-g / Three-step-ipw / Robins', each shown twice
(Oracle vs ML nuisances) -- four pages (variance, median, MSE, coverage), one
psi05-only PDF.

Mirrors ../../psi0_python/n72000/compare_oracle_vs_ml_psi05.py with
TRUE_PSI=1.0 for this (non-null) directory.

ML_nuisance/simulation_results.pkl is the 2026-09-02 production rerun: both
the ee_three_step_ipw equation fix and the full-data covariate-plug-in
working_model() change (mu05..mu45 fit on the full cross-fitting fold with
A_t as an explicit covariate, rather than only the A_t==A_{t-1} subset --
see CLAUDE.md) are reflected in these numbers.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(HERE, "psi05_oracle_vs_ml.pdf")
TRUE_PSI = 1.0
PSI_PARAM = "psi05"

P_I_COLS = ["p_I1", "p_I2", "p_I3", "p_I4"]
PH_COLS = ["ph1", "ph2", "ph3", "ph4"]

METHOD_MAP = {
    "Three-step-g": "Three-step-g estimator",
    "Three-step-ipw": "Three-step-ipw estimator",
    "Robins' estimator": "Simple g-estimator",
}
METHODS = [
    "Simple g-estimator",
    "Three-step-g estimator",
    "Three-step-ipw estimator",
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

    return (
        df.groupby(P_I_COLS + PH_COLS + ["method", "source"])
        .agg(mean=(est_col, center), var=(est_col, var),
             mse=(est_col, mse), coverage=(est_col, coverage))
        .reset_index()
    )


def _fmt(values, name):
    vals = list(values)
    if all(v == vals[0] for v in vals):
        return f"{name}={vals[0]:g}"
    return f"{name}=({', '.join(f'{v:g}' for v in vals)})"


# Method colors (matching check_results.py); Oracle = solid, ML = hatched.
COLORS = {"Simple g-estimator": "#3B4992",
          "Three-step-g estimator": "#EE0000",
          "Three-step-ipw estimator": "#008B45"}
SOURCES = ["Oracle", "ML"]
HATCH = {"Oracle": "", "ML": "///"}
ALPHA = {"Oracle": 0.9, "ML": 0.55}


def plot_one_metric(summary, metric, title, pdf, pi_combos, ph_combos):
    n_rows, n_cols = len(pi_combos), len(ph_combos)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.4 * n_rows),
        sharey=True, constrained_layout=True, squeeze=False,
    )
    bar_w = 0.38
    x = np.arange(len(METHODS))

    for row_i, pi_vec in enumerate(pi_combos):
        for col_i, ph_vec in enumerate(ph_combos):
            ax = axes[row_i, col_i]
            mask = np.ones(len(summary), dtype=bool)
            for col, val in zip(P_I_COLS, pi_vec):
                mask &= (summary[col] == val).values
            for col, val in zip(PH_COLS, ph_vec):
                mask &= (summary[col] == val).values
            sub = summary[mask]

            for s_i, source in enumerate(SOURCES):
                offset = (s_i - 0.5) * bar_w
                heights = [
                    sub.loc[(sub["method"] == m) & (sub["source"] == source), metric].values
                    for m in METHODS
                ]
                heights = [h[0] if len(h) else np.nan for h in heights]
                bars = ax.bar(
                    x + offset, heights, width=bar_w,
                    color=[COLORS[m] for m in METHODS],
                    alpha=ALPHA[source], hatch=HATCH[source],
                    edgecolor="black", linewidth=0.4,
                    label=source,
                )
                vlabels = ["" if not np.isfinite(h) else f"{h:.3g}" for h in heights]
                ax.bar_label(bars, labels=vlabels, fontsize=5, padding=1.5, rotation=90)

            if metric == "coverage":
                ax.axhline(0.95, linestyle="--", color="black", linewidth=0.8)
            if col_i == 0:
                ax.set_ylabel(_fmt(pi_vec, "p_I"), fontsize=8)
            if row_i == 0:
                ax.set_title(_fmt(ph_vec, "ph"), fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(METHODS, fontsize=6, rotation=20, ha="right")
            ax.tick_params(labelsize=8)
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    # One shared legend for Oracle (solid) vs ML (hatched).
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", alpha=ALPHA[s], hatch=HATCH[s], label=s)
        for s in SOURCES
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(title, fontsize=13)
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df_oracle = load("Oracle", os.path.join(HERE, "oracle", "simulation_results.pkl"))
    df_ml = load("ML", os.path.join(HERE, "ML_nuisance", "simulation_results.pkl"))
    df = pd.concat([df_oracle, df_ml], ignore_index=True)
    df = df[df["method"].isin(METHODS)]

    summary = summarize(df, PSI_PARAM)
    print(f"===== {PSI_PARAM}: Oracle vs ML nuisances =====")
    print(summary.to_string(index=False))

    pi_combos = sorted(df[P_I_COLS].drop_duplicates().itertuples(index=False, name=None))
    ph_combos = sorted(df[PH_COLS].drop_duplicates().itertuples(index=False, name=None))

    g = r"\psi_{05}"
    metrics = [
        ("var",      rf"Empirical variance of ${g}$ — Oracle vs ML nuisances"),
        ("mean",     rf"Empirical median of ${g}$ — Oracle vs ML nuisances"),
        ("mse",      rf"Empirical MSE of ${g}$ — Oracle vs ML nuisances"),
        ("coverage", rf"Empirical coverage of ${g}$ — Oracle vs ML nuisances"),
    ]
    with PdfPages(OUTPUT_PATH) as pdf:
        for metric, title in metrics:
            plot_one_metric(summary, metric, title, pdf, pi_combos, ph_combos)

        # Caveat page.
        fig = plt.figure(figsize=(8.5, 2.4))
        fig.text(
            0.05, 0.5,
            "Both runs are post-fix (2026-09-02): ee_three_step_ipw equation fix\n"
            "and the full-data covariate-plug-in working_model() change (mu05..mu45\n"
            "fit on the full cross-fitting fold with A_t as a covariate, rather than\n"
            "only the A_t==A_{t-1} subset) are both reflected here -- see CLAUDE.md.\n"
            "Known residual gap: at ph=0.3, Simple g/Three-step-g's ML variance still\n"
            "runs ~3-4x above Oracle (Three-step-ipw's plain-parametric nuisances track\n"
            "Oracle closely everywhere). A matched n=144,000 check (40 iters, not yet\n"
            "at production 300-iter precision) did not show this gap closing with n.",
            fontsize=10, va="center",
        )
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
