"""Full report suite for the psi = 1, n = 24000 run (12-cell ph x p_I grid).

Ports the psi0/n24000 reports to psi1's single-time-point grid and true psi = 1:
  * Bar plots (value labels): 4-metric summary, variance (IQR + 2.5/97.5),
    bias, failure proportion.
  * QQ + histogram, using ALL values (no |est|<=10 restriction).
  * Bias z-score CSVs: bias / sqrt(var/300) per cell for psi03/13/23.
  * GMM failure table.

Run after run_simulation.py:  python make_reports.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "simulation_results.pkl")
PREFIX = os.path.join(HERE, "simulation_results_psi1_n24k_plots")

# --- config (matches the psi0/n24000 conventions) ---
N_ITER = 300
PARAMS = ["est_psi03", "est_psi13", "est_psi23"]
TRUE = {"est_psi03": 1.0, "est_psi13": 1.0, "est_psi23": 1.0}   # psi = 1 mechanism
FAIL_THRESH = 10.0       # |est| > this == divergent
FAIL_RATE_MAX = 0.05     # drop a method in a cell with > this divergence rate
DISPLAY_THRESH = float("inf")   # QQ/hist: inf = plot all values

COLORS = ["#3B4992", "#EE0000", "#008B45"]
METHODS = ["Three-step estimator", "Simple g-estimator", "GMM estimator"]
CMAP = {m: COLORS[i] for i, m in enumerate(METHODS)}
PLABEL = {"est_psi03": r"\psi_{03}", "est_psi13": r"\psi_{13}",
          "est_psi23": r"\psi_{23}"}


def load():
    df = pd.read_pickle(RESULTS)
    method_map = {"Three-step": "Three-step estimator",
                  "Robins' estimator": "Simple g-estimator",
                  "GMM estimator": "GMM estimator"}
    df = df.copy()
    df["method"] = df["method"].map(method_map).fillna(df["method"])
    return df[df["method"].isin(METHODS)]


# ----------------------------- statistics --------------------------------
def _failrate(x):
    return (x.abs() > FAIL_THRESH).mean()


def _robust_var(x, method):
    if _failrate(x) > FAIL_RATE_MAX:
        return np.nan
    if method == "iqr":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        return ((q3 - q1) / 1.349) ** 2
    lo, hi = x.quantile(0.025), x.quantile(0.975)
    return ((hi - lo) / 3.96) ** 2


def _robust_center(x, true):
    if _failrate(x) > FAIL_RATE_MAX:
        return np.nan
    return x.median() - true


def robust_summary(df, var_method):
    rows = []
    for (pI, ph, m), c in df.groupby(["p_I", "ph", "method"]):
        d = dict(p_I=pI, ph=ph, method=m)
        for p in PARAMS:
            x = c[p].dropna()
            d["bias_" + p] = _robust_center(x, TRUE[p])
            d["var_" + p] = _robust_var(x, var_method)
        rows.append(d)
    return pd.DataFrame(rows)


def failrate_summary(df):
    rows = []
    for (pI, ph, m), c in df.groupby(["p_I", "ph", "method"]):
        d = dict(p_I=pI, ph=ph, method=m)
        for p in PARAMS:
            d["fail_" + p] = _failrate(c[p].dropna())
        rows.append(d)
    return pd.DataFrame(rows)


def four_metric(df, param="est_psi03"):
    true = TRUE[param]
    rows = []
    for (pI, ph, m), c in df.groupby(["p_I", "ph", "method"]):
        x = c[param].dropna()
        if _failrate(x) > FAIL_RATE_MAX:
            med = var = mse = cov = np.nan
        else:
            med = x.median()
            sd = (x.quantile(0.975) - x.quantile(0.025)) / 3.96
            var = sd ** 2
            mse = (med - true) ** 2 + var
            cov = np.mean((x - 1.96 * sd < true) & (x + 1.96 * sd > true))
        rows.append(dict(p_I=pI, ph=ph, method=m,
                         median=med, var=var, mse=mse, coverage=cov))
    return pd.DataFrame(rows)


# ----------------------------- bar plots ---------------------------------
def _bar_page(summary, metric, title, pdf, hline=None):
    p_I_vals = sorted(summary["p_I"].unique())
    ph_vals = sorted(summary["ph"].unique())
    x = np.arange(len(ph_vals))
    bw = 0.8 / len(METHODS)
    fig, axes = plt.subplots(1, len(p_I_vals), figsize=(4 * len(p_I_vals), 4),
                             sharey=True, squeeze=False)
    for ci, pI in enumerate(p_I_vals):
        ax = axes[0][ci]
        sub = summary[summary["p_I"] == pI]
        for mi, m in enumerate(METHODS):
            ms = sub[sub["method"] == m].set_index("ph").reindex(ph_vals)
            off = x + (mi - (len(METHODS) - 1) / 2) * bw
            h = ms[metric].values
            bars = ax.bar(off, h, width=bw, color=CMAP[m], label=m, alpha=0.85)
            ax.bar_label(bars, labels=["" if not np.isfinite(v) else f"{v:.3g}"
                                       for v in h],
                         fontsize=6, rotation=0, padding=1.5)
        if hline is not None:
            ax.axhline(hline, ls="--", color="black", lw=0.8)
        ax.set_title(rf"$p_I = {pI}$", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in ph_vals], fontsize=8)
        ax.set_xlabel("ph", fontsize=8)
        ax.grid(True, axis="y", lw=0.4, alpha=0.5)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
               fontsize=9, bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.10)
    fig.tight_layout()
    pdf.savefig(fig, dpi=200, bbox_inches="tight")
    plt.close(fig)


def bar_reports(df):
    # 4-metric summary (psi03)
    fm = four_metric(df, "est_psi03")
    with PdfPages(PREFIX + ".pdf") as pdf:
        _bar_page(fm, "var", r"Empirical variance of $\psi_{03}$", pdf)
        _bar_page(fm, "median", r"Empirical median of $\psi_{03}$ (true $=1$)", pdf,
                  hline=1.0)
        _bar_page(fm, "mse", r"Empirical MSE of $\psi_{03}$", pdf)
        _bar_page(fm, "coverage", r"Empirical coverage of $\psi_{03}$", pdf,
                  hline=0.95)
    print("Saved", PREFIX + ".pdf")

    # variance (IQR + 2.5/97.5) and bias, all three params
    for vm, suf, lab in [("iqr", "_variance_iqr.pdf", "variance (IQR)"),
                         ("pct", "_variance_2.5-97.5.pdf", "variance (2.5/97.5)")]:
        s = robust_summary(df, vm)
        with PdfPages(PREFIX + suf) as pdf:
            for p in PARAMS:
                _bar_page(s, "var_" + p,
                          rf"Empirical {lab} of ${PLABEL[p]}$", pdf)
        print("Saved", PREFIX + suf)

    sb = robust_summary(df, "iqr")
    with PdfPages(PREFIX + "_bias.pdf") as pdf:
        for p in PARAMS:
            _bar_page(sb, "bias_" + p, rf"Empirical bias of ${PLABEL[p]}$", pdf,
                      hline=0.0)
    print("Saved", PREFIX + "_bias.pdf")

    fr = failrate_summary(df)
    with PdfPages(PREFIX + "_failure_proportion.pdf") as pdf:
        for p in PARAMS:
            _bar_page(fr, "fail_" + p,
                      rf"Failure proportion of ${PLABEL[p]}$ ($|\hat\psi|>10$)", pdf)
    print("Saved", PREFIX + "_failure_proportion.pdf")


# --------------------------- QQ / histogram ------------------------------
def _robust_z(x):
    med = np.median(x)
    sd = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    return (x - med) / sd if sd > 0 else x - med


def _dist_page(df, param, kind, pdf):
    p_I_vals = sorted(df["p_I"].unique())
    ph_vals = sorted(df["ph"].unique())
    fig, axes = plt.subplots(len(p_I_vals), len(ph_vals),
                             figsize=(4 * len(ph_vals), 3.0 * len(p_I_vals)),
                             squeeze=False, constrained_layout=True)
    for r, pI in enumerate(p_I_vals):
        for cc, ph in enumerate(ph_vals):
            ax = axes[r][cc]
            cell = df[(df.p_I == pI) & (df.ph == ph)]
            for m in METHODS:
                x = cell.loc[cell.method == m, param].dropna().values
                x = x[np.abs(x) <= DISPLAY_THRESH]
                if len(x) < 5:
                    continue
                if kind == "qq":
                    z = np.sort(_robust_z(x))
                    theo = norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
                    ax.plot(theo, z, ".", ms=3, color=CMAP[m], alpha=0.6, label=m)
                else:
                    ax.hist(x, bins=25, density=True, histtype="stepfilled",
                            color=CMAP[m], alpha=0.35, label=m)
                    ax.hist(x, bins=25, density=True, histtype="step",
                            color=CMAP[m], lw=1.1)
            if kind == "qq":
                lim = 3.2
                ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
            else:
                ax.axvline(TRUE[param], color="black", ls="--", lw=0.7)
            if r == 0:
                ax.set_title(f"ph = {ph}", fontsize=9)
            if cc == 0:
                ax.set_ylabel(rf"$p_I = {pI}$", fontsize=8)
            ax.grid(True, lw=0.3, alpha=0.5)
            ax.tick_params(labelsize=7)
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=False)
    note = "all values" if not np.isfinite(DISPLAY_THRESH) else rf"$|\hat\psi|\leq {DISPLAY_THRESH:g}$"
    name = "QQ plot" if kind == "qq" else "Histogram"
    fig.suptitle(rf"{name} of ${PLABEL[param]}$  ({note})",
                 fontsize=14, fontweight="bold", y=1.04)
    pdf.savefig(fig, dpi=200, bbox_inches="tight")
    plt.close(fig)


def dist_reports(df):
    for out, kind in [("_qq.pdf", "qq"), ("_histogram.pdf", "hist")]:
        with PdfPages(PREFIX + out) as pdf:
            for p in PARAMS:
                _dist_page(df, p, kind, pdf)
        print("Saved", PREFIX + out)


# --------------------------- bias z-scores -------------------------------
def bias_zscores(df):
    for param in PARAMS:
        rows = []
        for (pI, ph, m), c in df.groupby(["p_I", "ph", "method"]):
            x = c[param].dropna()
            xc = x[x.abs() <= FAIL_THRESH]          # clip divergent
            bias = xc.mean() - TRUE[param]
            var = _robust_var(xc, "iqr")
            se = np.sqrt(var / N_ITER)
            z = bias / se
            rows.append(dict(p_I=pI, ph=ph, method=m, n_kept=len(xc),
                             bias=bias, var=var, se=se, z=z))
        r = pd.DataFrame(rows)
        r["sig"] = r["z"].abs() > 1.96
        tag = param.replace("est_", "")
        out = os.path.join(HERE, f"bias_se_zscores_psi1_n24k_{tag}_mean_clipped.csv")
        r.to_csv(out, index=False)
        print(f"{tag}: significant |z|>1.96 = {r['sig'].sum()}/{len(r)}  -> {os.path.basename(out)}")


# --------------------------- GMM failure ---------------------------------
def gmm_failure(df):
    g = df[df.method == "GMM estimator"]
    bad = []
    for (pI, ph), c in g.groupby(["p_I", "ph"]):
        fr = _failrate(c["est_psi03"])
        if fr > FAIL_RATE_MAX:
            bad.append((pI, ph, fr))
    print("\nGMM failure cells (|est_psi03|>10 in >5% of reps):")
    if not bad:
        print("  None -- GMM well-behaved in all 12 cells at n=24000.")
    else:
        for pI, ph, fr in bad:
            print(f"  p_I={pI}, ph={ph}: {fr*100:.0f}%")


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} rows, "
          f"{df[['p_I','ph']].drop_duplicates().shape[0]} cells, "
          f"methods={sorted(df['method'].unique())}\n")
    bar_reports(df)
    dist_reports(df)
    print()
    bias_zscores(df)
    gmm_failure(df)
