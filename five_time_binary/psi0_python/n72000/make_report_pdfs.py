"""Regenerate the variance and bias report PDFs.

Rebuilds the report PDFs headlessly after a new simulation run. Uses true
psi = 0 (matching the simulated data) for bias.

NOTE: the GMM estimator is not part of this simulation. Only the Three-step
and Robins' (simple g-) estimators are reported.

Outputs:
  simulation_results_plots_variance.pdf
  simulation_results_plots_bias.pdf
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

TRUE_PSI05 = 0
TRUE_PSI15 = 0
TRUE_PSI25 = 0
TRUE_PSI35 = 0
TRUE_PSI45 = 0

COLORS = ["#3B4992", "#EE0000", "#008B45", "#631879"]
METHODS = [
    "Three-step estimator",
    "Simple g-estimator",
]

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
        "Three-step": "Three-step estimator",
        "Robins' estimator": "Simple g-estimator",
        "eff": "Semiparametric efficient estimator",
    }
    df = df.copy()
    df["method"] = df["method"].map(method_map).fillna(df["method"])
    return df


def fmt_vec(values, name=None):
    """Compact label for a 4-tuple.

    Constant across time -> "0.2" (or "ph=0.2" when `name` is given);
    otherwise the full tuple "(1, 0.8, 0.5, 0.2)".
    """
    vals = list(values)
    if all(v == vals[0] for v in vals):
        body = f"{vals[0]:g}"
    else:
        body = "(" + ", ".join(f"{v:g}" for v in vals) + ")"
    return body if name is None else f"{name}={body}"


def cell_combos(summary, cols):
    """Sorted unique tuples of the given cell-identifier columns."""
    return sorted(summary[cols].drop_duplicates().itertuples(index=False, name=None))


def _mask_for(summary, cols, vec):
    mask = np.ones(len(summary), dtype=bool)
    for col, val in zip(cols, vec):
        mask &= (summary[col] == val).values
    return mask


def bar_value_labels(ax, bars, heights, fontsize=5):
    """Write the exact value above each bar; blank for dropped/NaN bars.

    Uses ax.text rather than ax.bar_label: bar_label creates Annotation objects,
    which fail inside a SubFigure on matplotlib < 3.6 ('SubFigure' object has no
    attribute 'get_dpi'). combine_variance_figure.py draws into subfigures.
    """
    for bar, h in zip(bars, heights):
        if not np.isfinite(h):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3g}",
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=fontsize)


# A method is not reported in a cell where more than this fraction of its
# estimates diverge (|estimate| > FAIL_THRESH). Above ~5% failures the
# 2.5/97.5 percentile range can no longer absorb the outliers.
FAIL_THRESH = 10.0
FAIL_RATE_MAX = 0.05
PARAMS = ["est_psi05", "est_psi15", "est_psi25", "est_psi35", "est_psi45"]


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


def summarize(df, true_psi05=TRUE_PSI05, true_psi15=TRUE_PSI15,
              true_psi25=TRUE_PSI25, true_psi35=TRUE_PSI35,
              true_psi45=TRUE_PSI45, robust=False, var_method="iqr"):
    """Per-cell summary. If robust=True, center = median and spread = the robust
    variance (var_method "iqr" or "pct"); divergent estimates need not be removed.
    """
    truths = {
        "est_psi05": true_psi05,
        "est_psi15": true_psi15,
        "est_psi25": true_psi25,
        "est_psi35": true_psi35,
        "est_psi45": true_psi45,
    }
    agg = {}
    for param in PARAMS:
        suffix = param.replace("est_", "")          # e.g. "psi05"
        true = truths[param]
        if robust:
            agg[f"bias_{suffix}"] = (
                param, lambda x, t=true: _robust_center(x, t))
            agg[f"var_{suffix}"] = (
                param, lambda x, vm=var_method: _robust_var(x, vm))
        else:
            agg[f"bias_{suffix}"] = (param, lambda x, t=true: x.mean() - t)
            agg[f"var_{suffix}"] = (param, "var")

    summary = (
        df.groupby(P_I_COLS + PH_COLS + ["method"])
        .agg(**agg)
        .reset_index()
    )
    return summary


def _bar_panel(summary, psis, output_suffix, label, logy=False):
    """Multipage PDF, one page per psi metric.

    Layout per page: grouped bars with x = the ph setting and color = method.
    Facet: rows = (p_I1..p_I4) combos, one column.
    """
    pi_combos = cell_combos(summary, P_I_COLS)
    ph_combos = cell_combos(summary, PH_COLS)
    n_rows, n_cols = len(pi_combos), 1
    color_map = {m: COLORS[i] for i, m in enumerate(METHODS)}
    x = np.arange(len(ph_combos))
    bar_width = 0.8 / len(METHODS)

    out = OUTPUT_PATH.replace(".pdf", output_suffix)
    with PdfPages(out) as pdf:
        for metric, title in psis:
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(1.9 * len(ph_combos) + 3, 3.2 * n_rows),
                sharex=True, sharey=True, constrained_layout=True, squeeze=False,
            )
            for row_i, pi_vec in enumerate(pi_combos):
                ax = axes[row_i, 0]
                sub = summary[_mask_for(summary, P_I_COLS, pi_vec)]
                for m_i, m in enumerate(METHODS):
                    msub = sub[sub["method"] == m]
                    heights = []
                    for ph_vec in ph_combos:
                        v = msub.loc[_mask_for(msub, PH_COLS, ph_vec), metric].values
                        heights.append(v[0] if len(v) else np.nan)
                    offsets = x + (m_i - (len(METHODS) - 1) / 2) * bar_width
                    bars = ax.bar(offsets, heights, width=bar_width,
                                  label=m, color=color_map[m], alpha=0.85)
                    # Exact value above each bar (blank for dropped/NaN bars).
                    vlabels = ["" if not np.isfinite(h) else f"{h:.3g}"
                               for h in heights]
                    ax.bar_label(bars, labels=vlabels, fontsize=5,
                                 rotation=0, padding=1.5)
                if logy:
                    ax.set_yscale("log")
                ax.set_ylabel(fmt_vec(pi_vec, "p_I"), fontsize=9)
                if row_i == n_rows - 1:
                    ax.set_xlabel("ph", fontsize=9)
                ax.set_xticks(x)
                ax.set_xticklabels([fmt_vec(v) for v in ph_combos], fontsize=8)
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
        ("var_psi05", r"Empirical variance of $\psi_{05}$"),
        ("var_psi15", r"Empirical variance of $\psi_{15}$"),
        ("var_psi25", r"Empirical variance of $\psi_{25}$"),
        ("var_psi35", r"Empirical variance of $\psi_{35}$"),
        ("var_psi45", r"Empirical variance of $\psi_{45}$"),
    ]
    _bar_panel(summary, psis, suffix, label, logy=False)


def plot_bias(summary):
    psis = [
        ("bias_psi05", r"Empirical bias of $\psi_{05}$"),
        ("bias_psi15", r"Empirical bias of $\psi_{15}$"),
        ("bias_psi25", r"Empirical bias of $\psi_{25}$"),
        ("bias_psi35", r"Empirical bias of $\psi_{35}$"),
        ("bias_psi45", r"Empirical bias of $\psi_{45}$"),
    ]
    _bar_panel(summary, psis, "_bias.pdf", "bias", logy=False)


def failrate_summary(df):
    """Per-cell, per-method proportion of divergent estimates (|est| > 10)."""
    agg = {f"fail_{p.replace('est_', '')}": (p, _failrate) for p in PARAMS}
    return (
        df.groupby(P_I_COLS + PH_COLS + ["method"])
        .agg(**agg)
        .reset_index()
    )


def plot_failrate(summary):
    psis = [
        ("fail_psi05", r"Failure proportion of $\psi_{05}$ ($|\hat\psi|>10$)"),
        ("fail_psi15", r"Failure proportion of $\psi_{15}$ ($|\hat\psi|>10$)"),
        ("fail_psi25", r"Failure proportion of $\psi_{25}$ ($|\hat\psi|>10$)"),
        ("fail_psi35", r"Failure proportion of $\psi_{35}$ ($|\hat\psi|>10$)"),
        ("fail_psi45", r"Failure proportion of $\psi_{45}$ ($|\hat\psi|>10$)"),
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
    g = df.groupby(P_I_COLS + PH_COLS + ["method"])
    for p in PARAMS:
        bad = g[p].apply(_failrate)
        bad = bad[bad > FAIL_RATE_MAX]
        if len(bad):
            print(f"Dropped (failure rate > {FAIL_RATE_MAX:.0%}) for {p}:")
            for idx, rate in bad.items():
                print(f"    {dict(zip(P_I_COLS + PH_COLS + ['method'], idx))}"
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
