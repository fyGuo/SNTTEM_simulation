"""Outer loop: run simulation over a grid of (ph, p_I) values.

psi=1 (non-null) mechanism, ported from psi0_python/n72000 with PSI05..PSI45
set to 1.0 instead of 0.0. Grid and sample size match the psi0 production
run: n=72000, ph in {0.3,0.5,0.8}, p_I in {0.2,1.0}, 6 cells, 300 iterations.

generate_data.py's intercept is lowered to -2.6 (from -1.5) here to keep
exp(log_p) < 1 under the compounded psi=1 blip terms -- see its docstring
and CLAUDE.md "Known issues".

Results are saved to simulation_results.pkl (and also as simulation_results.csv
for easy inspection).

NOTE: the GMM estimator is not computed. The Three-step-g, Three-step-ipw,
and Robins' estimators are reported.
"""
import os
import time
import itertools
import pandas as pd

from simulation_output import run_simulation

# True parameter values (psi = 1 mechanism)
PSI05 = 1.0
PSI15 = 1.0
PSI25 = 1.0
PSI35 = 1.0
PSI45 = 1.0

# Simulation settings
N = 72000
N_ITER = 300
SEED = 3411

# Grid:
#   PH_VALUES:  probability that A_t stays the same as A_{t-1}
#   P_I_VALUES: probability of being eligible (I_t = 1)
#
# With five time points a fully factorial per-time-point grid would be
# 2^4 * 3^4 = 1296 cells, so the value is held COMMON across t = 1..4 here:
# each grid cell sets ph1 = ph2 = ph3 = ph4 and p_I1 = ... = p_I4.
# That gives 2 * 3 = 6 cells.
P_I_VALUES = [0.2, 1]
PH_VALUES = [0.3, 0.5, 0.8]

# Optional heterogeneous scenarios, run in addition to the common-value grid.
# Each entry is (label, (p_I1, p_I2, p_I3, p_I4), (ph1, ph2, ph3, ph4)).
# Set to [] to run the common-value grid only.
EXTRA_SCENARIOS = []

output_path = os.path.join(os.path.dirname(__file__), "simulation_results.pkl")

combos = [
    ((p_I, p_I, p_I, p_I), (ph, ph, ph, ph))
    for p_I, ph in itertools.product(P_I_VALUES, PH_VALUES)
]
combos += [(p_I_vec, ph_vec) for _label, p_I_vec, ph_vec in EXTRA_SCENARIOS]
n_combos = len(combos)

all_results = []
combo_times = []
t_start = time.time()

for k, (p_I_vec, ph_vec) in enumerate(combos, start=1):
    p_I1, p_I2, p_I3, p_I4 = p_I_vec
    ph1, ph2, ph3, ph4 = ph_vec
    print(f"\n{'='*60}")
    print(f"[{k}/{n_combos}] Running: "
          f"p_I=({p_I1}, {p_I2}, {p_I3}, {p_I4}), "
          f"ph=({ph1}, {ph2}, {ph3}, {ph4})")
    print(f"{'='*60}")
    t0 = time.time()

    est = run_simulation(
        n=N,
        n_iter=N_ITER,
        psi05=PSI05,
        psi15=PSI15,
        psi25=PSI25,
        psi35=PSI35,
        psi45=PSI45,
        p_I1=p_I1,
        p_I2=p_I2,
        p_I3=p_I3,
        p_I4=p_I4,
        ph1=ph1,
        ph2=ph2,
        ph3=ph3,
        ph4=ph4,
        seed=SEED,
        n_jobs=-1,
    )
    all_results.append(est)
    elapsed = time.time() - t0
    combo_times.append(
        {"p_I1": p_I1, "p_I2": p_I2, "p_I3": p_I3, "p_I4": p_I4,
         "ph1": ph1, "ph2": ph2, "ph3": ph3, "ph4": ph4,
         "seconds": elapsed}
    )
    print(f"Done in {elapsed:.1f}s")

results = pd.concat(all_results, ignore_index=True)
results.to_pickle(output_path)
results.to_csv(output_path.replace(".pkl", ".csv"), index=False)

# --- Timing report: per-combination and average ---
times_df = pd.DataFrame(combo_times)
times_path = output_path.replace(".pkl", "_timings.csv")
times_df.to_csv(times_path, index=False)

total = time.time() - t_start
print(f"\n{'='*60}\nTIMING REPORT  ({N_ITER} iterations per combination)\n{'='*60}")
print(times_df.to_string(index=False))
print(f"\nAverage simulation time per combination: "
      f"{times_df['seconds'].mean():.1f}s "
      f"(min {times_df['seconds'].min():.1f}s, "
      f"max {times_df['seconds'].max():.1f}s)")
print(f"Total elapsed over {n_combos} combinations: {total:.1f}s")
print(f"Saved results to {output_path}")
print(f"Saved timings to {times_path}")
