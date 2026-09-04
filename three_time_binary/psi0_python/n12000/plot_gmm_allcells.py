"""Regenerate the IQR variance figure with the GMM estimator shown in EVERY cell.

The IQR variance uses only the middle 50% of estimates, so it is immune to GMM's
divergent tails -- GMM is well-behaved in all 36 cells there (max IQR variance
~0.99), so nothing should be blanked. This overwrites the default IQR figure
(`simulation_results_n12k_plots_variance_iqr.pdf`) to include GMM everywhere.

The 2.5/97.5 percentile figure is left alone: there GMM's divergent estimates
DO enter the spread, so the default behaviour (blank GMM where its failure rate
exceeds 5%) is kept.
"""
import os

import make_report_pdfs as mr

HERE = os.path.dirname(os.path.abspath(__file__))

# No per-cell drop: show GMM in every cell for the IQR figure.
mr.FAIL_RATE_MAX = 1.0


def main():
    df = mr.rename_methods(mr.load_results())
    df = df[df["method"].isin(mr.METHODS)]
    # _bar_panel derives the filename from mr.OUTPUT_PATH; keep the canonical name.
    mr.OUTPUT_PATH = os.path.join(HERE, "simulation_results_n12k_plots.pdf")
    mr.plot_variance(mr.summarize(df, robust=True, var_method="iqr"),
                     "_variance_iqr.pdf", "variance (IQR)")


if __name__ == "__main__":
    main()
