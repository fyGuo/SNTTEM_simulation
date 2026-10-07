"""psi05 when the later blip models are misspecified (CausaLab cut).

Same point + 95% CI layout as ../compare_oracle_vs_ml_psi05_causalab.py, but
showing only this folder's run: gamma15..gamma45 fitted as psi*(-L-0.5)
while the DGP keeps psi*(1+L) (gamma05 is correct), with exact closed-form
(oracle) nuisances.

Only Simple g-/Robins' (SNTTEM) and Three-step-ipw (labelled "Three-step") are shown.
Writes psi05_misspecification_causalab.pdf next to itself: point (median) +
95% CI error bar (median +/- 1.96 * robust SD) against the TRUE_PSI line.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(HERE, "psi05_misspecification_causalab.pdf")
TRUE_PSI = 1.0
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
PLOT_DROP_P_I = [0.2]

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


# Method colors: SNTTEM = blue, Three-step = red.
COLORS = {"Simple g-estimator": "#1F77B4",
          "Weighted regression estimator": "#D62728"}
DISPLAY_LABELS = {"Simple g-estimator": "SNTTEM",
                   "Weighted regression estimator": "Three-step"}
SOURCES = ["Misspecified blips"]
ALPHA = {"Misspecified blips": 0.9}
MARKER = {"Misspecified blips": "o"}


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
    offsets = {"Misspecified blips": 0.0}

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

    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load("Misspecified blips", os.path.join(HERE, "simulation_results.pkl"))
    df = df[df["method"].isin(METHODS)]

    summary = summarize(df, PSI_PARAM)
    print(f"===== {PSI_PARAM}: misspecified blips (oracle nuisances) =====")
    print(summary.to_string(index=False))

    pi_combos = sorted(df[P_I_COLS].drop_duplicates().itertuples(index=False, name=None))
    ph_combos = sorted(df[PH_COLS].drop_duplicates().itertuples(index=False, name=None))
    # Figure only: ph columns listed in PLOT_DROP_PH are left out.
    ph_combos = [ph_vec for ph_vec in ph_combos if ph_vec[0] not in PLOT_DROP_PH]
    # Figure only: p_I rows listed in PLOT_DROP_P_I are left out.
    pi_combos = [pi_vec for pi_vec in pi_combos if pi_vec[0] not in PLOT_DROP_P_I]

    with PdfPages(OUTPUT_PATH) as pdf:
        plot_point_ci(summary, pdf, pi_combos, ph_combos)
    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
