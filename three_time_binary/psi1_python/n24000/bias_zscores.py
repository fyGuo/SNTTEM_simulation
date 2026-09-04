"""Bias z-score tables: z = bias / sqrt(var/N_ITER) per cell, flag |z| > 1.96.

bias = mean(|est| <= FAIL_THRESH) - true   (clipped mean)
var  = IQR-based robust variance
true = 1 (psi = 1 mechanism)

Writes one CSV per parameter (psi03/13/23). Run after run_simulation.py.
"""
import os
import numpy as np
import pandas as pd
import make_report_pdfs as mr   # reuse load/rename/METHODS/_robust_var/FAIL_THRESH

HERE = os.path.dirname(os.path.abspath(__file__))
N_ITER = 300
TRUE = 1.0
PARAMS = ["est_psi03", "est_psi13", "est_psi23"]
CELL = ["p_I1", "p_I2", "ph1", "ph2"]

if __name__ == "__main__":
    df = mr.rename_methods(mr.load_results())
    df = df[df["method"].isin(mr.METHODS)].copy()

    for param in PARAMS:
        rows = []
        for keys, c in df.groupby(CELL + ["method"]):
            x = c[param].dropna()
            xc = x[x.abs() <= mr.FAIL_THRESH]        # clip divergent
            bias = xc.mean() - TRUE
            var = mr._robust_var(xc, "iqr")
            se = np.sqrt(var / N_ITER)
            z = bias / se
            d = dict(zip(CELL + ["method"], keys))
            d.update(n_kept=len(xc), bias=bias, var=var, se=se, z=z)
            rows.append(d)
        r = pd.DataFrame(rows)
        r["sig"] = r["z"].abs() > 1.96
        tag = param.replace("est_", "")
        out = os.path.join(HERE, f"bias_se_zscores_psi1_n24k_{tag}_mean_clipped.csv")
        r.to_csv(out, index=False)
        print(f"{tag}: significant |z|>1.96 = {r['sig'].sum()}/{len(r)}  -> {os.path.basename(out)}")
        print(r.groupby("method")["sig"].sum().to_string())
        print()
