"""Median/variance report for a single scale-test run, matching the style of
the original scaletest_48k_ph03_median_variance.pdf.

One page per psi parameter (psi05..psi45); each page is a 2x2 grid: rows =
p_I value (0.2, 1), columns = (median estimate, empirical variance). Bars =
the three methods (Robins' estimator, Three-step-g, Three-step-ipw).

Usage: edit N / CSV_PATH / OUT_PATH below and run.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "Robins' estimator": "#3B4992",
    "Three-step-g": "#EE0000",
    "Three-step-ipw": "#008B45",
}
METHODS = ["Robins' estimator", "Three-step-g", "Three-step-ipw"]
PARAMS = ["psi05", "psi15", "psi25", "psi35", "psi45"]


def make_report(n, csv_path, out_path, ph=0.3):
    df = pd.read_csv(csv_path)
    p_I_values = sorted(df["p_I1"].unique())

    with PdfPages(out_path) as pdf:
        for param in PARAMS:
            col = f"est_{param}"
            fig, axes = plt.subplots(
                len(p_I_values), 2, figsize=(9, 4.4 * len(p_I_values)),
                constrained_layout=True,
            )
            if len(p_I_values) == 1:
                axes = axes.reshape(1, 2)

            for row_i, p_I in enumerate(p_I_values):
                sub = df[df["p_I1"] == p_I]
                medians, variances = [], []
                for m in METHODS:
                    x = sub.loc[sub["method"] == m, col]
                    medians.append(x.median())
                    lo, hi = x.quantile(0.025), x.quantile(0.975)
                    variances.append(((hi - lo) / 3.96) ** 2)

                ax_med, ax_var = axes[row_i, 0], axes[row_i, 1]
                colors = [COLORS[m] for m in METHODS]

                bars = ax_med.bar(METHODS, medians, color=colors, alpha=0.85)
                for b, v in zip(bars, medians):
                    ax_med.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                                ha="center", va="bottom" if v >= 0 else "top",
                                fontsize=9)
                ax_med.axhline(0, color="black", linewidth=0.8)
                ax_med.tick_params(axis="x", labelrotation=0, labelsize=8)
                ax_med.set_ylabel(f"p_I = {p_I:g}", fontsize=11, fontweight="bold")
                if row_i == 0:
                    ax_med.set_title("Median estimate (true = 0)", fontsize=11)

                bars = ax_var.bar(METHODS, variances, color=colors, alpha=0.85)
                for b, v in zip(bars, variances):
                    ax_var.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                                ha="center", va="bottom", fontsize=9)
                ax_var.tick_params(axis="x", labelrotation=0, labelsize=8)
                if row_i == 0:
                    ax_var.set_title("Empirical variance", fontsize=11)

            fig.suptitle(
                rf"$\psi_{{{param[3:]}}}$ at n={n:,}, ph={ph:g} "
                "— median & variance (300 iterations, true "
                rf"$\psi_{{{param[3:]}}}$ = 0)",
                fontsize=13,
            )
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_report(
        n=72000,
        csv_path=os.path.join(HERE, "scaletest_72k_ph03_300iter_results.csv"),
        out_path=os.path.join(HERE, "scaletest_72k_ph03_median_variance.pdf"),
    )
    make_report(
        n=96000,
        csv_path=os.path.join(HERE, "scaletest_96k_ph03_300iter_results.csv"),
        out_path=os.path.join(HERE, "scaletest_96k_ph03_median_variance.pdf"),
    )
