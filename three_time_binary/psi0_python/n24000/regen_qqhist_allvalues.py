"""Regenerate the n24k QQ + histogram using ALL estimates (no |est|<=10 cap)."""
import os
import distribution_plots as dp
dp.DISPLAY_THRESH = float("inf")          # plot every value
HERE = os.path.dirname(os.path.abspath(dp.__file__))
dp.QQ_OUT   = os.path.join(HERE, "simulation_results_n24k_plots_qq.pdf")
dp.HIST_OUT = os.path.join(HERE, "simulation_results_n24k_plots_histogram.pdf")
dp.main()
