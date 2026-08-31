"""Visualize simulation results.

Produces four panel plots for each of the five blip parameters
(psi05, psi15, psi25, psi35, psi45):
  - Empirical variance
  - Empirical median (bias check)
  - Mean squared error
  - Coverage (95% CI using empirical SD)

Run after run_simulation.py has produced simulation_results.pkl.

NOTE: the GMM estimator is not part of this simulation. Only the Three-step
and Robins' (simple g-) estimators are summarized.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "simulation_results.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "simulation_results_plots.pdf")
TRUE_PSI = 0.0  # true value matches the simulated data (psi_j5 = 0 for all j)

# All five blip parameters this simulation estimates (see CLAUDE.md).
PSI_PARAMS = ["psi05", "psi15", "psi25", "psi35", "psi45"]

# Cell identifiers: five time points means four eligibility and four
# persistence probabilities per cell.
P_I_COLS = ["p_I1", "p_I2", "p_I3", "p_I4"]
PH_COLS = ["ph1", "ph2", "ph3", "ph4"]


def load_results(path=RESULTS_PATH):
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    return pd.read_csv(path)


def rename_methods(df):
    method_map = {
        "Three-step-g": "Three-step-g estimator",
        "Three-step-ipw": "Three-step-ipw estimator",
        "Robins' estimator": "Simple g-estimator",
        "eff": "Semiparametric efficient estimator",
    }
    df = df.copy()
    df["method"] = df["method"].map(method_map).fillna(df["method"])
    return df


def _fmt(values, name):
    """Compact label for a 4-tuple: 'p_I=0.2' if constant, else the full tuple."""
    vals = list(values)
    if all(v == vals[0] for v in vals):
        return f"{name}={vals[0]:g}"
    return f"{name}=({', '.join(f'{v:g}' for v in vals)})"


# Drop a method in a cell where more than this fraction of its psi05 estimates
# diverge (|est| > FAIL_THRESH); above ~5% the 2.5/97.5 range is unreliable.
FAIL_THRESH = 10.0
FAIL_RATE_MAX = 0.05


def _robust_sd(x):
    """Robust SD via the 2.5/97.5 percentile range: (P97.5 - P2.5) / 3.96."""
    return (x.quantile(0.975) - x.quantile(0.025)) / 3.96


def summarize(df, psi_param, true_psi=TRUE_PSI):
    """Per-cell robust summary statistics for one blip parameter (e.g. 'psi05').

    Center = median, spread = 2.5/97.5 percentile range. A method whose failure
    rate (|est| > 10) exceeds 5% in a cell is dropped (all metrics NaN), so
    divergent cells are simply not reported. MSE uses bias^2 + var; the
    coverage CI uses the robust SD as the standard-error plug-in.
    """
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

    summary = (
        df.groupby(P_I_COLS + PH_COLS + ["method"])
        .agg(
            mean=(est_col, center),
            var=(est_col, var),
            mse=(est_col, mse),
            coverage=(est_col, coverage),
        )
        .reset_index()
    )
    return summary


COLORS = ["#3B4992", "#EE0000", "#008B45", "#631879"]
METHODS = [
    "Simple g-estimator",
    "Three-step-g estimator",
    "Three-step-ipw estimator",
]


def _greek(psi_param):
    """'psi05' -> r'\\psi_{05}' for plot titles."""
    return r"\psi_{" + psi_param[3:] + "}"


def plot_one_psi(summary, psi_param, pdf, pi_combos, ph_combos):
    """Append the four metric pages for one blip parameter to an open PdfPages."""
    n_rows, n_cols = len(pi_combos), len(ph_combos)
    g = _greek(psi_param)

    metrics = [
        ("var",      rf"Empirical variance of ${g}$"),
        ("mean",     rf"Empirical median of ${g}$"),
        ("mse",      rf"Empirical MSE of ${g}$"),
        ("coverage", rf"Empirical coverage proportion of ${g}$"),
    ]

    color_map = {m: COLORS[i] for i, m in enumerate(METHODS)}
    x = np.arange(len(METHODS))

    for metric, title in metrics:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
            sharey=True, constrained_layout=True, squeeze=False,
        )
        for row_i, pi_vec in enumerate(pi_combos):
            for col_i, ph_vec in enumerate(ph_combos):
                ax = axes[row_i, col_i]
                mask = np.ones(len(summary), dtype=bool)
                for col, val in zip(P_I_COLS, pi_vec):
                    mask &= (summary[col] == val).values
                for col, val in zip(PH_COLS, ph_vec):
                    mask &= (summary[col] == val).values
                sub = summary[mask]
                vals = [
                    sub.loc[sub["method"] == m, metric].values
                    for m in METHODS
                ]
                heights = [v[0] if len(v) else np.nan for v in vals]
                bars = ax.bar(x, heights,
                              color=[color_map[m] for m in METHODS],
                              alpha=0.85)
                # Exact value above each bar (blank for dropped/NaN bars).
                vlabels = ["" if not np.isfinite(h) else f"{h:.3g}"
                           for h in heights]
                ax.bar_label(bars, labels=vlabels, fontsize=5,
                             rotation=0, padding=1.5)
                if metric == "coverage":
                    ax.axhline(0.95, linestyle="--", color="black",
                               linewidth=0.8)
                if col_i == 0:
                    ax.set_ylabel(_fmt(pi_vec, "p_I"), fontsize=8)
                if row_i == 0:
                    ax.set_title(_fmt(ph_vec, "ph"), fontsize=9)
                ax.set_xticks(x)
                ax.set_xticklabels(METHODS, fontsize=6, rotation=20,
                                   ha="right")
                ax.tick_params(labelsize=8)
                ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        fig.suptitle(title, fontsize=13)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_results(df, output_path=OUTPUT_PATH):
    """One PDF, 4 pages per blip parameter (var, median, mse, coverage).

    Rows = (p_I1..p_I4) combos, cols = (ph1..ph4) combos. Within each panel the
    metric is shown as bars over the methods (the only remaining variation).
    """
    pi_combos = sorted(
        df[P_I_COLS].drop_duplicates().itertuples(index=False, name=None)
    )
    ph_combos = sorted(
        df[PH_COLS].drop_duplicates().itertuples(index=False, name=None)
    )
    with PdfPages(output_path) as pdf:
        for psi_param in PSI_PARAMS:
            summary = summarize(df, psi_param)
            print(f"\n===== {psi_param} =====")
            print(summary.to_string(index=False))
            plot_one_psi(summary, psi_param, pdf, pi_combos, ph_combos)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    df = load_results()
    df = rename_methods(df)
    df = df[df["method"].isin(METHODS)]
    plot_results(df)
