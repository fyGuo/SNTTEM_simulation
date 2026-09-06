"""Compare psi05 at the hard cell (p_I=0.2, ph=0.3) across sample sizes.

Loads simulation_results.pkl from n72000/, n144000/, and n288000/ (both
oracle/ and ML_nuisance/ subfolders), restricts to the hard cell in each,
and plots grouped bars of Simple g-/Three-step-g/Three-step-ipw, each shown
twice (Oracle vs ML nuisances), with columns for n=72000/144000/288000 --
four pages (variance, median, MSE, coverage), one psi05-only PDF.

n72000's pkls hold the full 6-cell production grid, so those two are
filtered down to the hard cell; n144000's and n288000's pkls already hold
only that one cell.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(HERE, "psi05_hardcell_n_progression.pdf")
TRUE_PSI = 1.0
PSI_PARAM = "psi05"

HARD_CELL = {"p_I1": 0.2, "p_I2": 0.2, "p_I3": 0.2, "p_I4": 0.2,
             "ph1": 0.3, "ph2": 0.3, "ph3": 0.3, "ph4": 0.3}

N_VALUES = [72000, 144000, 288000]

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


def load(source_label, n_value, path):
    df = pd.read_pickle(path)
    for col, val in HARD_CELL.items():
        df = df[df[col] == val]
    df = df.copy()
    df["method"] = df["method"].map(METHOD_MAP).fillna(df["method"])
    df["source"] = source_label
    df["n"] = n_value
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
        df.groupby(["n", "method", "source"])
        .agg(mean=(est_col, center), var=(est_col, var),
             mse=(est_col, mse), coverage=(est_col, coverage))
        .reset_index()
    )


# Method colors (matching check_results.py); Oracle = solid, ML = hatched.
COLORS = {"Simple g-estimator": "#3B4992",
          "Three-step-g estimator": "#EE0000",
          "Three-step-ipw estimator": "#008B45"}
SOURCES = ["Oracle", "ML"]
HATCH = {"Oracle": "", "ML": "///"}
ALPHA = {"Oracle": 0.9, "ML": 0.55}


def plot_one_metric(summary, metric, title, pdf):
    n_cols = len(N_VALUES)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(4.5 * n_cols, 3.8),
        sharey=True, constrained_layout=True, squeeze=False,
    )
    bar_w = 0.38
    x = np.arange(len(METHODS))

    for col_i, n_val in enumerate(N_VALUES):
        ax = axes[0, col_i]
        sub = summary[summary["n"] == n_val]

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
            ax.bar_label(bars, labels=vlabels, fontsize=6, padding=1.5, rotation=0)

        if metric == "coverage":
            ax.axhline(0.95, linestyle="--", color="black", linewidth=0.8)
        ax.set_title(f"n={n_val:,}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=6, rotation=20, ha="right")
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

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
    frames = []
    for n_val in N_VALUES:
        folder = f"n{n_val}"
        frames.append(load("Oracle", n_val, os.path.join(HERE, folder, "oracle", "simulation_results.pkl")))
        frames.append(load("ML", n_val, os.path.join(HERE, folder, "ML_nuisance", "simulation_results.pkl")))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["method"].isin(METHODS)]

    summary = summarize(df, PSI_PARAM)
    print(f"===== {PSI_PARAM} at the hard cell (p_I=0.2, ph=0.3): n progression =====")
    print(summary.sort_values(["n", "method", "source"]).to_string(index=False))

    g = r"\psi_{05}"
    metrics = [
        ("var",      rf"Empirical variance of ${g}$ at the hard cell — Oracle vs ML nuisances"),
        ("mean",     rf"Empirical median of ${g}$ at the hard cell — Oracle vs ML nuisances"),
        ("mse",      rf"Empirical MSE of ${g}$ at the hard cell — Oracle vs ML nuisances"),
        ("coverage", rf"Empirical coverage of ${g}$ at the hard cell — Oracle vs ML nuisances"),
    ]
    with PdfPages(OUTPUT_PATH) as pdf:
        for metric, title in metrics:
            plot_one_metric(summary, metric, title, pdf)

        fig = plt.figure(figsize=(8.5, 2.4))
        fig.text(
            0.05, 0.5,
            "Hard cell: p_I=0.2, ph=0.3 (the worst cell in the production grid --\n"
            "only ~0.81% of subjects follow a constant-treatment path). n=72000's\n"
            "numbers are the hard-cell row of the full 6-cell production grid;\n"
            "n=144000 and n=288000 rerun only this cell, at 2x and 4x n. All runs\n"
            "post-ee_three_step_ipw-fix, 300 iterations, seed 3411. See CLAUDE.md\n"
            "\"Status: n=144000 hard-cell check\".",
            fontsize=10, va="center",
        )
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
