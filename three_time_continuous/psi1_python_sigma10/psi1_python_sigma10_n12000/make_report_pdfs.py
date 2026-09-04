"""Regenerate the variance and bias report PDFs (continuous-outcome psi1 DGP).

Mirrors the binary make_report_pdfs.py but uses true psi = 1 (matching the
continuous simulated data) for bias.

Outputs:
  simulation_results_plots_variance_iqr.pdf
  simulation_results_plots_variance_2.5-97.5.pdf
  simulation_results_plots_bias.pdf
  simulation_results_plots_failure_proportion.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "simulation_results.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "simulation_results_plots.pdf")

TRUE_PSI03 = 1
TRUE_PSI13 = 1
TRUE_PSI23 = 1

COLORS = ["#3B4992", "#EE0000", "#008B45", "#631879"]
METHODS = [
    "Three-step estimator",
    "Simple g-estimator",
    "GMM estimator",
]


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


# A method is not reported in a cell where more than this fraction of its
# estimates diverge (|estimate| > FAIL_THRESH). Above ~5% failures the
# 2.5/97.5 percentile range can no longer absorb the outliers.
FAIL_THRESH = 10.0
FAIL_RATE_MAX = 0.05
PARAMS = ["est_psi03", "est_psi13", "est_psi23"]


def _failrate(x):
    return (x.abs() > FAIL_THRESH).mean()


def _robust_var(x, method="iqr"):
    """Robust variance estimate, NaN when the cell has too many failures.

    method="iqr": ((P75 - P25) / 1.349) ** 2 -- middle 50% only, immune to heavy
        tails, reflects the estimators' core efficiency.
    method="pct": ((P97.5 - P2.5) / 3.96) ** 2 -- 95% percentile range, captures
        fuller variability including the tails.
    """
    if _failrate(x) > FAIL_RATE_MAX:
        return float("nan")
    if method == "iqr":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        return ((q3 - q1) / 1.349) ** 2
    lo, hi = x.quantile(0.025), x.quantile(0.975)
    return ((hi - lo) / 3.96) ** 2


def _robust_center(x, true):
    """Median minus the truth, or NaN if the cell has too many failures."""
    if _failrate(x) > FAIL_RATE_MAX:
        return float("nan")
    return x.median() - true


def summarize(df, true_psi03=TRUE_PSI03, true_psi13=TRUE_PSI13,
              true_psi23=TRUE_PSI23, robust=False, var_method="iqr"):
    """Per-cell summary. If robust=True, center = median and spread = the robust
    variance (var_method "iqr" or "pct"); divergent estimates need not be removed.
    """
    if robust:
        rv = lambda x: _robust_var(x, var_method)
        agg = {
            "bias_psi03": ("est_psi03", lambda x: _robust_center(x, true_psi03)),
            "var_psi03": ("est_psi03", rv),
            "bias_psi13": ("est_psi13", lambda x: _robust_center(x, true_psi13)),
            "var_psi13": ("est_psi13", rv),
            "bias_psi23": ("est_psi23", lambda x: _robust_center(x, true_psi23)),
            "var_psi23": ("est_psi23", rv),
        }
    else:
        agg = {
            "bias_psi03": ("est_psi03", lambda x: x.mean() - true_psi03),
            "var_psi03": ("est_psi03", "var"),
            "bias_psi13": ("est_psi13", lambda x: x.mean() - true_psi13),
            "var_psi13": ("est_psi13", "var"),
            "bias_psi23": ("est_psi23", lambda x: x.mean() - true_psi23),
            "var_psi23": ("est_psi23", "var"),
        }
    summary = (
        df.groupby(["p_I1", "p_I2", "ph1", "ph2", "method"])
        .agg(**agg)
        .reset_index()
    )
    return summary


def _bar_panel(summary, psis, output_suffix, label, logy=False):
    """Multipage PDF, one page per psi metric.

    Layout per page: grouped bars with x = ph1 and color = method.
    Facet: rows = (p_I1, p_I2) combos, cols = ph2.
    """
    pi_combos = sorted(
        summary[["p_I1", "p_I2"]].drop_duplicates().itertuples(index=False, name=None)
    )
    ph2_vals = sorted(summary["ph2"].unique())
    ph1_vals = sorted(summary["ph1"].unique())
    n_rows, n_cols = len(pi_combos), len(ph2_vals)
    color_map = {m: COLORS[i] for i, m in enumerate(METHODS)}
    x = np.arange(len(ph1_vals))
    bar_width = 0.8 / len(METHODS)

    out = OUTPUT_PATH.replace(".pdf", output_suffix)
    with PdfPages(out) as pdf:
        for metric, title in psis:
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                sharex=True, sharey=True, constrained_layout=True, squeeze=False,
            )
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
                        # Exact value above each bar (blank for dropped/NaN bars).
                        vlabels = ["" if not np.isfinite(h) else f"{h:.3g}"
                                   for h in heights]
                        ax.bar_label(bars, labels=vlabels, fontsize=5,
                                     rotation=0, padding=1.5)
                    if logy:
                        ax.set_yscale("log")
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
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
                       fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=False)
            fig.suptitle(title, fontsize=15, fontweight="bold", y=1.07)
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {label} plot to {out}")


def plot_variance(summary, suffix="_variance.pdf", label="variance"):
    psis = [
        ("var_psi03", r"Empirical variance of $\psi_{03}$"),
        ("var_psi13", r"Empirical variance of $\psi_{13}$"),
        ("var_psi23", r"Empirical variance of $\psi_{23}$"),
    ]
    _bar_panel(summary, psis, suffix, label, logy=False)


def plot_bias(summary):
    psis = [
        ("bias_psi03", r"Empirical bias of $\psi_{03}$"),
        ("bias_psi13", r"Empirical bias of $\psi_{13}$"),
        ("bias_psi23", r"Empirical bias of $\psi_{23}$"),
    ]
    _bar_panel(summary, psis, "_bias.pdf", "bias", logy=False)


def failrate_summary(df):
    """Per-cell, per-method proportion of divergent estimates (|est| > 10)."""
    return (
        df.groupby(["p_I1", "p_I2", "ph1", "ph2", "method"])
        .agg(
            fail_psi03=("est_psi03", _failrate),
            fail_psi13=("est_psi13", _failrate),
            fail_psi23=("est_psi23", _failrate),
        )
        .reset_index()
    )


def plot_failrate(summary):
    psis = [
        ("fail_psi03", r"Failure proportion of $\psi_{03}$ ($|\hat\psi|>10$)"),
        ("fail_psi13", r"Failure proportion of $\psi_{13}$ ($|\hat\psi|>10$)"),
        ("fail_psi23", r"Failure proportion of $\psi_{23}$ ($|\hat\psi|>10$)"),
    ]
    _bar_panel(summary, psis, "_failure_proportion.pdf", "failure proportion",
               logy=False)


if __name__ == "__main__":
    df = load_results()
    df = rename_methods(df)
    df = df[df["method"].isin(METHODS)]

    # Robust summary: median center, and the variance via either the IQR or the
    # 2.5/97.5 percentile range (saved as separate figures). A method is dropped
    # (not reported) in any cell where >5% of its estimates diverge.
    g = df.groupby(["p_I1", "p_I2", "ph1", "ph2", "method"])
    for p in PARAMS:
        bad = g[p].apply(_failrate)
        bad = bad[bad > FAIL_RATE_MAX]
        if len(bad):
            print(f"Dropped (failure rate > {FAIL_RATE_MAX:.0%}) for {p}:")
            for idx, rate in bad.items():
                print(f"    {dict(zip(['p_I1','p_I2','ph1','ph2','method'], idx))}"
                      f"  failrate={rate:.0%}")

    # Variance: one figure per method.
    plot_variance(summarize(df, robust=True, var_method="iqr"),
                  "_variance_iqr.pdf", "variance (IQR)")
    plot_variance(summarize(df, robust=True, var_method="pct"),
                  "_variance_2.5-97.5.pdf", "variance (2.5/97.5)")
    # Bias is the same either way (median - truth).
    plot_bias(summarize(df, robust=True))
    # Failure proportion per cell/method (kept for reference).
    plot_failrate(failrate_summary(df))
