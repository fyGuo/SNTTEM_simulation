"""Regenerate every bar-chart report PDF with exact value labels on each bar,
keeping the n12k filenames and the agreed GMM handling:

  - IQR variance ........ GMM shown in all 36 cells (outlier-immune)
  - 2.5/97.5 variance ... GMM blanked where its failure rate > 5%
  - summary / bias ...... GMM blanked where its failure rate > 5%
  - failure proportion .. all methods, all cells
"""
import os
import check_results as cr
import make_report_pdfs as mr

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "simulation_results_n12k_plots.pdf")  # suffix stem
mr.OUTPUT_PATH = BASE


def main():
    # ---- check_results 4-metric summary (GMM dropped where it fails > 5%) ----
    cr.FAIL_RATE_MAX = 0.05
    dfc = cr.rename_methods(cr.load_results())
    dfc = dfc[dfc["method"].isin(cr.METHODS)]
    cr.plot_results(cr.summarize(dfc), output_path=BASE)

    # ---- make_report variance / bias / failure proportion ----
    dfm = mr.rename_methods(mr.load_results())
    dfm = dfm[dfm["method"].isin(mr.METHODS)]

    mr.FAIL_RATE_MAX = 1.0   # IQR: keep GMM in every cell
    mr.plot_variance(mr.summarize(dfm, robust=True, var_method="iqr"),
                     "_variance_iqr.pdf", "variance (IQR)")

    mr.FAIL_RATE_MAX = 0.05  # 2.5/97.5 + bias: blank GMM where it fails > 5%
    mr.plot_variance(mr.summarize(dfm, robust=True, var_method="pct"),
                     "_variance_2.5-97.5.pdf", "variance (2.5/97.5)")
    mr.plot_bias(mr.summarize(dfm, robust=True))
    mr.plot_failrate(mr.failrate_summary(dfm))


if __name__ == "__main__":
    main()
