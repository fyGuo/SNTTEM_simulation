"""Histogram + kernel density of estimated parameters by method.

Reads simulation_results.csv and produces, for each estimated parameter
(est_psi05 ... est_psi45), a histogram overlaid with a Gaussian KDE for each
estimation method. True parameter value (0) marked.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

df = pd.read_csv("simulation_results.csv")

params = ["est_psi05", "est_psi15", "est_psi25", "est_psi35", "est_psi45"]
true_vals = {p: 0.0 for p in params}
methods = list(df["method"].unique())
colors = dict(zip(methods, plt.cm.tab10.colors))

# Cell identifiers: five time points means four eligibility and four
# persistence probabilities per cell.
P_I_COLS = ["p_I1", "p_I2", "p_I3", "p_I4"]
PH_COLS = ["ph1", "ph2", "ph3", "ph4"]

# Facet: rows = (p_I1..p_I4) combos, cols = (ph1..ph4) combos
pi_combos = sorted(df[P_I_COLS].drop_duplicates().itertuples(index=False, name=None))
ph_combos = sorted(df[PH_COLS].drop_duplicates().itertuples(index=False, name=None))


def fmt_vec(values, name=None):
    """Compact label: '0.2' if constant across time, else '(1, 0.8, 0.5, 0.2)'."""
    vals = list(values)
    if all(v == vals[0] for v in vals):
        body = f"{vals[0]:g}"
    else:
        body = "(" + ", ".join(f"{v:g}" for v in vals) + ")"
    return body if name is None else f"{name}={body}"


def cell_mask(frame, cols, vec):
    mask = np.ones(len(frame), dtype=bool)
    for col, val in zip(cols, vec):
        mask &= (frame[col] == val).values
    return mask


def panel(ax, sub, p):
    """Draw histogram + KDE per method on one axis for subset `sub`."""
    lo, hi = sub[p].quantile(0.001), sub[p].quantile(0.999)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = sub[p].min() - 0.1, sub[p].max() + 0.1
    bins = np.linspace(lo, hi, 26)
    xs = np.linspace(lo, hi, 300)
    for m in methods:
        vals = sub.loc[sub["method"] == m, p].dropna().values
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, density=True, alpha=0.30, color=colors[m])
        if vals.std() > 0:
            ax.plot(xs, gaussian_kde(vals)(xs), color=colors[m], lw=1.8,
                    label=m)
    ax.axvline(true_vals[p], color="black", ls="--", lw=1.0)


# One figure per parameter: rows = (p_I1..p_I4), cols = (ph1..ph4)
for p in params:
    fig, axes = plt.subplots(len(pi_combos), len(ph_combos),
                             figsize=(4 * len(ph_combos), 3 * len(pi_combos)),
                             sharex=True, squeeze=False)
    for i, pi_vec in enumerate(pi_combos):
        for j, ph_vec in enumerate(ph_combos):
            ax = axes[i, j]
            sub = df[cell_mask(df, P_I_COLS, pi_vec)
                     & cell_mask(df, PH_COLS, ph_vec)]
            panel(ax, sub, p)
            if i == 0:
                ax.set_title(fmt_vec(ph_vec, "ph"))
            if j == 0:
                ax.set_ylabel(f"{fmt_vec(pi_vec, 'p_I')}\ndensity")
            if i == len(pi_combos) - 1:
                ax.set_xlabel("estimate")
    # single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="black", ls="--", lw=1.0))
    labels.append("true value")
    fig.legend(handles, labels, loc="upper center", ncol=len(methods) + 1,
               fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"{p}: distribution by setting (rows p_I, cols ph)",
                 y=1.0, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"estimates_density_{p}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

# Per-method summary (bias / SD)
summary = (df.groupby("method")[params]
             .agg(["mean", "std"])
             .round(4))
print(summary)
