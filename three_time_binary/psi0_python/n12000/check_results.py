"""Visualize simulation results. Equivalent to check_results.R.

Produces four panel plots for psi03:
  - Empirical variance
  - Empirical mean (bias check)
  - Mean squared error
  - Coverage (95% CI using empirical SD)

Run after run_simulation.py has produced simulation_results.pkl.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "simulation_results.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "simulation_results_plots.pdf")
TRUE_PSI03 = 0.0  # true value matches the simulated data (psi03 = 0)


def load_results(path=RESULTS_PATH):
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    return pd.read_csv(path)


def rename_methods(df):
    method_map = {
        "Three-step": "Three-step estimator",
        "Robins' estimator": "Simple g-estimator",
        "GMM estimator": "GMM estimator",
        "eff": "Semiparametric efficient estimator",
    }
    df = df.copy()
    df["method"] = df["method"].map(method_map).fillna(df["method"])
    return df


# Drop a method in a cell where more than this fraction of its psi03 estimates
# diverge (|est| > FAIL_THRESH); above ~5% the 2.5/97.5 range is unreliable.
FAIL_THRESH = 10.0
FAIL_RATE_MAX = 0.05


def _robust_sd(x):
    """Robust SD via the 2.5/97.5 percentile range: (P97.5 - P2.5) / 3.96."""
    return (x.quantile(0.975) - x.quantile(0.025)) / 3.96


def summarize(df, true_psi03=TRUE_PSI03):
    """Per-cell robust summary statistics for psi03.

    Center = median, spread = 2.5/97.5 percentile range. A method whose failure
    rate (|est| > 10) exceeds 5% in a cell is dropped (all metrics NaN), so the
    divergent GMM cells are simply not reported. MSE uses bias^2 + var; the
    coverage CI uses the robust SD as the standard-error plug-in.
    """
    def dropped(x):
        return (x.abs() > FAIL_THRESH).mean() > FAIL_RATE_MAX

    def center(x):
        return np.nan if dropped(x) else x.median()

    def var(x):
        return np.nan if dropped(x) else _robust_sd(x) ** 2

    def mse(x):
        return np.nan if dropped(x) else (x.median() - true_psi03) ** 2 + _robust_sd(x) ** 2

    def coverage(x):
        if dropped(x):
            return np.nan
        sd = _robust_sd(x)
        return np.mean((x - 1.96 * sd < true_psi03) & (x + 1.96 * sd > true_psi03))

    summary = (
        df.groupby(["p_I1", "p_I2", "ph1", "ph2", "method"])
        .agg(
            mean_psi03=("est_psi03", center),
            var_psi03=("est_psi03", var),
            mse_psi03=("est_psi03", mse),
            coverage_psi03=("est_psi03", coverage),
        )
        .reset_index()
    )
    return summary


COLORS = ["#3B4992", "#EE0000", "#008B45", "#631879"]
METHODS = [
    "Three-step estimator",
    "Simple g-estimator",
    "GMM estimator",
]


def plot_results(summary, output_path=OUTPUT_PATH):
    """One PDF page per metric; each page a 4x4 facet.

    Rows = (p_I1, p_I2) combos, cols = (ph1, ph2) combos. Within each panel the
    metric is shown as bars over the methods (the only remaining variation).
    """
    pi_combos = sorted(
        summary[["p_I1", "p_I2"]].drop_duplicates().itertuples(index=False, name=None)
    )
    ph_combos = sorted(
        summary[["ph1", "ph2"]].drop_duplicates().itertuples(index=False, name=None)
    )
    n_rows, n_cols = len(pi_combos), len(ph_combos)

    metrics = [
        ("var_psi03",      r"Empirical variance of $\psi_{03}$"),
        ("mean_psi03",     r"Empirical median of $\psi_{03}$"),
        ("mse_psi03",      r"Empirical MSE of $\psi_{03}$"),
        ("coverage_psi03", r"Empirical coverage proportion of $\psi_{03}$"),
    ]

    color_map = {m: COLORS[i] for i, m in enumerate(METHODS)}
    x = np.arange(len(METHODS))

    with PdfPages(output_path) as pdf:
        for metric, title in metrics:
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                sharey=True, constrained_layout=True, squeeze=False,
            )
            for row_i, (pI1, pI2) in enumerate(pi_combos):
                for col_i, (ph1, ph2) in enumerate(ph_combos):
                    ax = axes[row_i, col_i]
                    sub = summary[
                        (summary["p_I1"] == pI1) & (summary["p_I2"] == pI2)
                        & (summary["ph1"] == ph1) & (summary["ph2"] == ph2)
                    ]
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
                    if metric == "coverage_psi03":
                        ax.axhline(0.95, linestyle="--", color="black",
                                   linewidth=0.8)
                    if col_i == 0:
                        ax.set_ylabel(f"p_I1={pI1}, p_I2={pI2}", fontsize=8)
                    if row_i == 0:
                        ax.set_title(f"ph1={ph1}, ph2={ph2}", fontsize=9)
                    ax.set_xticks(x)
                    ax.set_xticklabels(METHODS, fontsize=6, rotation=20,
                                       ha="right")
                    ax.tick_params(labelsize=8)
                    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
            fig.suptitle(title, fontsize=13)
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    df = load_results()
    df = rename_methods(df)
    df = df[df["method"].isin(METHODS)]
    # Robust summary (median + IQR-based spread); no outlier removal needed.
    summary = summarize(df)
    print(summary.to_string(index=False))
    plot_results(summary)
