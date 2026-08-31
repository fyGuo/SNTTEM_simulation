"""Trace eq1's mean value across a grid of psi05, with oracle nuisances, for
several independent large-n draws -- to see whether the nonzero root found by
fsolve (with theta=0 nuisances) is a stable feature of the equation or noise
from a single draw, and to see how flat/steep the curve is near psi05=0.
"""
import numpy as np
from scipy.special import expit

from generate_data import generate_data
from estimators import ee_three_step_g, ee_three_step_ipw, ee_robins

N = 2_000_000
TRUE_MU = np.exp(-1.5)
GRID = np.linspace(-0.3, 0.3, 13)


def oracle_nuisances(df, ph1, ph2, ph3, ph4):
    df = df.copy()
    df["ps0"] = expit(-1 + df["L0"].values)
    df["ps1"] = ph1 * df["A0"] + (1 - ph1) * (1 - df["A0"])
    df["ps2"] = ph2 * df["A1"] + (1 - ph2) * (1 - df["A1"])
    df["ps3"] = ph3 * df["A2"] + (1 - ph3) * (1 - df["A2"])
    df["ps4"] = ph4 * df["A3"] + (1 - ph4) * (1 - df["A3"])
    for col in ["mu05", "mu15", "mu25", "mu35", "mu45"]:
        df[col] = TRUE_MU
    return df


for seed in [1, 2, 3]:
    rng = np.random.default_rng(seed)
    df = generate_data(
        N, psi05=0.0, psi15=0.0, psi25=0.0, psi35=0.0, psi45=0.0,
        p_I1=0.2, p_I2=0.2, p_I3=0.2, p_I4=0.2,
        ph1=0.3, ph2=0.3, ph3=0.3, ph4=0.3, rng=rng,
    )
    df = oracle_nuisances(df, 0.3, 0.3, 0.3, 0.3)

    print(f"\n{'='*70}\nseed={seed}, n={N:,}  -- eq1 mean value vs psi05 "
          f"(other thetas fixed at 0)\n{'='*70}")
    print(f"{'psi05':>8} {'g (mean)':>14} {'ipw (mean)':>14}")
    for psi05 in GRID:
        theta = np.array([psi05, 0, 0, 0, 0])
        g_val = ee_three_step_g(theta, df)[0] / N
        ipw_val = ee_three_step_ipw(theta, df)[0] / N
        print(f"{psi05:8.3f} {g_val:14.6f} {ipw_val:14.6f}")
