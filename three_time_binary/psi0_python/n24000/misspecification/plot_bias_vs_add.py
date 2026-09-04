"""Bias vs the additive misspecification constant c, across c in {0, 1.5, 3}.

The wrong blips are gamma_wrong = psi*(L+10)/(L+4) + c. Reads the three tagged
result files simulation_results_add{c}.pkl and plots, for the both-wrong scenario
(gamma13 and gamma23 both misspecified), the Monte Carlo mean bias as a function
of c. Under the null (true psi = 0) the theory predicts bias ~ -c / E[h(L)] with
h(L) = (L+10)/(L+4) ~ 2.85, i.e. a straight line through the origin.

Layout: rows = parameter (psi03, psi13, psi23), columns = the two settings
(p_I1 = p_I2 in {0.2, 1}); x-axis = c; one line per estimator.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ADDS = [0.0, 1.5, 3.0]
OUT = os.path.join(HERE, "misspecification_bias_vs_add.pdf")

METHOD_MAP = {"Three-step": "Three-step",
              "Robins' estimator": "Simple $g$",
              "GMM estimator": "GMM"}
METHODS = ["Three-step", "Simple $g$", "GMM"]
COLORS = {"Three-step": "#3B4992", "Simple $g$": "#EE0000", "GMM": "#008B45"}

PARAMS = [("est_psi03", r"$\widehat\psi_{03}$"),
          ("est_psi13", r"$\widehat\psi_{13}$"),
          ("est_psi23", r"$\widehat\psi_{23}$")]
TRUE_PSI = 0.0
EH = 2.85  # E[(L+10)/(L+4)] over L in [-1.5, -0.5]
PSI03_CAP = 3.0  # cap the psi03 row; Simple-g diverges far above this at c=3


def load_all():
    frames = []
    for c in ADDS:
        df = pd.read_pickle(os.path.join(HERE, f"simulation_results_add{c}.pkl"))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["method"] = df["method"].map(METHOD_MAP)
    return df


def main():
    df = load_all()
    # both-wrong scenario: gamma13 and gamma23 both misspecified.
    ww = df[(df.gamma13 == "wrong") & (df.gamma23 == "wrong")]
    pI_vals = sorted(df["p_I1"].unique())

    fig, axes = plt.subplots(len(PARAMS), len(pI_vals),
                             figsize=(4.2 * len(pI_vals), 3.0 * len(PARAMS)),
                             sharex=True, sharey="row", squeeze=False)

    for r, (col, plabel) in enumerate(PARAMS):
        for c_i, pI in enumerate(pI_vals):
            ax = axes[r, c_i]
            ax.axhline(0, color="0.5", lw=0.8, zorder=1)
            # theory line -c/E[h] for the directly-affected parameters
            if col in ("est_psi13", "est_psi23"):
                cc = np.array(ADDS)
                ax.plot(cc, -cc / EH, color="0.6", ls="--", lw=1.0, zorder=1,
                        label=r"theory $-c/\bar h$")
            for m in METHODS:
                biases, ndiv = [], []
                for c in ADDS:
                    v = ww[(ww.p_I1 == pI) & (ww.method == m)
                           & (ww.add_const == c)][col].dropna().values
                    biases.append(v.mean() - TRUE_PSI)
                    ndiv.append(int((np.abs(v) > 10).mean() * 100))
                ax.plot(ADDS, biases, marker="o", ms=5, color=COLORS[m],
                        lw=1.5, label=m, zorder=3)
                # Annotate points that fall outside the psi03 capped view.
                if col == "est_psi03":
                    for cc, b, nd in zip(ADDS, biases, ndiv):
                        if b > PSI03_CAP:
                            ax.annotate(f"{b:.1f}\n({nd}% div.)",
                                        xy=(cc, PSI03_CAP), xytext=(cc, PSI03_CAP),
                                        ha="right", va="top", fontsize=7,
                                        color=COLORS[m])
            if col == "est_psi03":
                ax.set_ylim(-0.4, PSI03_CAP)
            if r == 0:
                ax.set_title(f"$p_{{I1}}=p_{{I2}}={pI:g}$", fontsize=11)
            if c_i == 0:
                ax.set_ylabel(f"Bias of {plabel}", fontsize=11)
            if r == len(PARAMS) - 1:
                ax.set_xlabel("additive constant $c$", fontsize=10)
            ax.set_xticks(ADDS)
            ax.grid(True, axis="y", lw=0.4, alpha=0.5)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Bias vs additive misspecification constant $c$ "
                 "(both blips wrong; true $\\psi=0$, $n=24000$, 300 reps)",
                 fontsize=12, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
