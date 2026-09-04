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
import matplotlib.gridspec as gridspec

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "simulation_results.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "simulation_results_plots.pdf")
TRUE_PSI03 = 1.0  # true value used in check_results.R


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


def summarize(df, true_psi03=TRUE_PSI03):
    """Compute per-(p_I, ph, method) summary statistics for psi03."""
    def coverage(x):
        sd = np.std(x, ddof=1)
        return np.mean((x - 1.96 * sd < true_psi03) & (x + 1.96 * sd > true_psi03))

    summary = (
        df.groupby(["p_I", "ph", "method"])
        .agg(
            mean_psi03=("est_psi03", "mean"),
            var_psi03=("est_psi03", "var"),
            mse_psi03=("est_psi03", lambda x: np.mean((x - true_psi03) ** 2)),
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
    p_I_values = sorted(summary["p_I"].unique())
    n_cols = len(p_I_values)

    metrics = [
        ("var_psi03",      r"Empirical variance of $\psi_{03}$"),
        ("mean_psi03",     r"Empirical mean of $\psi_{03}$"),
        ("mse_psi03",      r"Empirical MSE of $\psi_{03}$"),
        ("coverage_psi03", r"Empirical coverage proportion of $\psi_{03}$"),
    ]

    fig, axes = plt.subplots(
        4, n_cols, figsize=(4 * n_cols, 16), sharey="row", constrained_layout=True
    )
    if n_cols == 1:
        axes = axes.reshape(4, 1)

    color_map = {m: COLORS[i] for i, m in enumerate(METHODS)}

    for col_i, p_I in enumerate(p_I_values):
        sub = summary[summary["p_I"] == p_I]
        for row_i, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_i, col_i]
            for method in METHODS:
                msub = sub[sub["method"] == method].sort_values("ph")
                if msub.empty:
                    continue
                ax.plot(
                    msub["ph"], msub[metric],
                    marker="o", label=method, color=color_map[method], alpha=0.85
                )
            if row_i == 3:
                ax.axhline(0.95, linestyle="--", color="black", linewidth=0.8)
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_i == 0:
                ax.set_title(
                    f"Eligibility proportion\n$I_1 = {p_I}$", fontsize=9
                )
            ax.set_xlabel(r"Proportion $A_1 = A_0$ (ph)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(METHODS),
        fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=False
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    df = load_results()
    df = rename_methods(df)
    df = df[df["method"].isin(METHODS)]
    summary = summarize(df)
    print(summary.to_string(index=False))
    plot_results(summary)
