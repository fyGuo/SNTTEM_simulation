"""psi05, Oracle vs Super-learner nuisances, n=144000 hard cell (CausaLab cut).

Same point + 95% CI figure as ../n72000/compare_oracle_vs_ml_psi05_causalab.py
(whose load/summarize/plot functions are reused here), but for this folder's
results. n144000/ only holds the hard cell (p_I=0.2, ph=0.3), so the figure
is a single panel -- there are no other cells to build step figures from.
"""
import os
import sys
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "n72000"))
import compare_oracle_vs_ml_psi05_causalab as base

OUTPUT_PATH = os.path.join(HERE, "psi05_oracle_vs_ml_causalab.pdf")


def main():
    df_oracle = base.load("Oracle", os.path.join(HERE, "oracle", "simulation_results.pkl"))
    df_ml = base.load("Super-learner", os.path.join(HERE, "ML_nuisance", "simulation_results.pkl"))
    df = pd.concat([df_oracle, df_ml], ignore_index=True)
    df = df[df["method"].isin(base.METHODS)]

    summary = base.summarize(df, base.PSI_PARAM)
    print(summary.to_string(index=False))

    pi_combos = sorted(df[base.P_I_COLS].drop_duplicates().itertuples(index=False, name=None))
    ph_combos = sorted(df[base.PH_COLS].drop_duplicates().itertuples(index=False, name=None))
    with PdfPages(OUTPUT_PATH) as pdf:
        base.plot_point_ci(summary, pdf, pi_combos, ph_combos)
    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
