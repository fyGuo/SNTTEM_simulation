"""Monte Carlo bias/variance summary table for psi_03 across sample sizes.

Robust statistics matching the variance figures:
  - Bias     = median(estimate) - true psi   (true psi = 0)
  - Variance = ((P97.5 - P2.5) / 3.96) ** 2
  - A method is blanked ("--") in any cell where > 5% of its estimates diverge
    (|estimate| > 10).

Layout: rows = the 36 (p_I1, p_I2, ph1, ph2) settings, stacked in one block per
sample size (n = 6000, 12000, 24000); columns = Bias / Var for each estimator.

Outputs (written next to this script):
  mc_bias_variance_psi03.csv
  mc_bias_variance_psi03.tex
"""
import os
import numpy as np
import pandas as pd

import make_report_pdfs as mr
from make_report_pdfs import (
    load_results, rename_methods, summarize, METHODS,
)

# Report the robust bias/variance for every cell, including GMM cells with a
# high divergence rate (their values can be extreme but are shown exactly rather
# than blanked). Disables the >5% failure-rate NaN gate in make_report_pdfs.
mr.FAIL_RATE_MAX = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))

# (sample size n, path to that run's pickle)
DATASETS = [
    (6000, os.path.join(HERE, "simulation_results.pkl")),
    (12000, os.path.join(HERE, "n12000", "simulation_results.pkl")),
    (24000, os.path.join(HERE, "n24000", "simulation_results.pkl")),
]

SETTING_COLS = ["p_I1", "p_I2", "ph1", "ph2"]
# Short column tags per estimator, in the canonical METHODS order.
METHOD_TAG = {
    "Three-step estimator": "threestep",
    "Simple g-estimator": "gest",
    "GMM estimator": "gmm",
}
# Target parameters reported as side-by-side panels (left -> right).
PARAMS = ["psi03", "psi13"]
PARAM_PRETTY = {"psi03": "$\\psi_{03}$", "psi13": "$\\psi_{13}$",
                "psi23": "$\\psi_{23}$"}


def build_long():
    """Return a tidy DataFrame: one row per (n, setting), with bias/var columns
    for every (estimator, parameter) pair."""
    rows = []
    for n, path in DATASETS:
        df = rename_methods(load_results(path))
        df = df[df["method"].isin(METHODS)]
        summ = summarize(df, robust=True, var_method="pct")
        piv = summ.pivot_table(
            index=SETTING_COLS, columns="method",
            values=[f"bias_{p}" for p in PARAMS] + [f"var_{p}" for p in PARAMS],
            dropna=False,
        )
        for setting, _ in piv.iterrows():
            rec = {"n": n}
            rec.update(dict(zip(SETTING_COLS, setting)))
            for p in PARAMS:
                for m in METHODS:
                    tag = METHOD_TAG[m]
                    rec[f"bias_{tag}_{p}"] = piv.loc[setting, (f"bias_{p}", m)]
                    rec[f"var_{tag}_{p}"] = piv.loc[setting, (f"var_{p}", m)]
            rows.append(rec)
    value_cols = []
    for p in PARAMS:
        for m in METHODS:
            tag = METHOD_TAG[m]
            value_cols += [f"bias_{tag}_{p}", f"var_{tag}_{p}"]
    out = pd.DataFrame(rows, columns=["n"] + SETTING_COLS + value_cols)
    return out.sort_values(["n"] + SETTING_COLS).reset_index(drop=True)


def fmt(x, nd=2):
    """Fixed 2 decimal places; NaN (dropped method) shown as an en dash."""
    if not np.isfinite(x):
        return "--"
    return f"{x:.{nd}f}"


def write_csv(long, path):
    long.to_csv(path, index=False)
    print(f"Wrote {path}")


def write_latex(long, path):
    tag_order = [METHOD_TAG[m] for m in METHODS]
    pretty = {"threestep": "Three-step", "gest": "Simple $g$", "gmm": "GMM"}
    n_est = len(METHODS)
    per_param = 2 * n_est                 # Bias/Var for each estimator
    ncols = 2 + len(PARAMS) * per_param   # ph1, ph2, then one panel per param

    # Compact table: fix p_I1 = p_I2 = 0.2 (stated in the caption) so the two
    # constant columns can be dropped, leaving ph1, ph2 + the estimator panels.
    # 14 columns is too wide for portrait, so the float is rotated (landscape).
    tab = long[(long["p_I1"] == 0.2) & (long["p_I2"] == 0.2)]
    param_list = ", ".join(PARAM_PRETTY[p] for p in PARAMS)

    lines = []
    lines.append("% Requires \\usepackage{booktabs}")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{4pt}\\small")
    lines.append("\\caption{Monte Carlo bias and variance of "
                 f"$\\widehat\\psi_{{03}}$ (left) and $\\widehat\\psi_{{13}}$ "
                 "(right) by sample size, with $p_{I1}=p_{I2}=0.2$ fixed and true "
                 "$\\psi=0$. Bias is median $-\\,\\psi$; variance is the robust "
                 "$((P_{97.5}-P_{2.5})/3.96)^2$ estimate. GMM entries are reported "
                 "exactly even in cells where many of its replicates diverge "
                 "($|\\hat\\psi|>10$), so some GMM variances are extremely large "
                 "and not meaningful.}")
    lines.append("\\label{tab:mc-bias-var-psi03}")
    lines.append("\\begin{tabular}{cc " + ("cc " * n_est) * len(PARAMS) + "}")
    lines.append("\\toprule")

    # Row 1: parameter-panel group headers.
    param_hdr = "& "
    param_cmids = []
    for i, p in enumerate(PARAMS):
        start = 3 + i * per_param
        param_hdr += f"& \\multicolumn{{{per_param}}}{{c}}{{{PARAM_PRETTY[p]}}} "
        param_cmids.append(f"\\cmidrule(lr){{{start}-{start + per_param - 1}}}")
    lines.append(param_hdr + "\\\\")
    lines.append(" ".join(param_cmids))

    # Row 2: estimator group headers (repeated within each panel).
    est_hdr = "& "
    est_cmids = []
    for i in range(len(PARAMS) * n_est):
        start = 3 + 2 * i
        est_cmids.append(f"\\cmidrule(lr){{{start}-{start + 1}}}")
    for _ in PARAMS:
        for t in tag_order:
            est_hdr += f"& \\multicolumn{{2}}{{c}}{{{pretty[t]}}} "
    lines.append(est_hdr + "\\\\")
    lines.append(" ".join(est_cmids))

    # Row 3: Bias/Var sub-headers.
    sub_hdr = "$p_{h1}$ & $p_{h2}$" + " & Bias & Var" * (len(PARAMS) * n_est)
    lines.append(sub_hdr + " \\\\")
    lines.append("\\midrule")

    n_vals = sorted(tab["n"].unique())
    for bi, n in enumerate(n_vals):
        block = tab[tab["n"] == n]
        lines.append(f"\\multicolumn{{{ncols}}}{{l}}"
                     f"{{\\textit{{$n = {n}$}}}} \\\\")
        for _, r in block.iterrows():
            cells = [f"{r.ph1:g}", f"{r.ph2:g}"]
            for p in PARAMS:
                for t in tag_order:
                    cells.append(fmt(r[f"bias_{t}_{p}"]))
                    cells.append(fmt(r[f"var_{t}_{p}"]))
            lines.append(" & ".join(cells) + " \\\\")
        # Separator between blocks, but not after the last one.
        lines.append("\\midrule" if bi < len(n_vals) - 1 else "\\bottomrule")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    long = build_long()
    write_csv(long, os.path.join(HERE, "mc_bias_variance_psi03.csv"))
    write_latex(long, os.path.join(HERE, "mc_bias_variance_psi03.tex"))
