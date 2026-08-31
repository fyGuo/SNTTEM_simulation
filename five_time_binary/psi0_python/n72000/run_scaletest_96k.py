"""One-off timing/stability probe: n=96000, 5 iterations per cell.

NOT the production run. Mirrors run_simulation.py's grid structure exactly
(same P_I_VALUES/PH_VALUES, same true psi values, same seed) but with N
scaled 4x (24000 -> 96000) and N_ITER cut to 5, purely to measure how long a
cell takes at the larger sample size -- now with the updated three-method
estimator set (Three-step-g, Three-step-ipw, Robins') -- before committing to
a full 300-iteration run at 96000.

Writes to separate output files so the existing n=24000, 300-iteration
production results (simulation_results.pkl/.csv) are untouched.
"""
import os
import time
import itertools
import pandas as pd

from simulation_output import run_simulation

PSI05 = 0.0
PSI15 = 0.0
PSI25 = 0.0
PSI35 = 0.0
PSI45 = 0.0

N = 96000
N_ITER = 5
SEED = 3411

P_I_VALUES = [0.2, 1]
PH_VALUES = [0.2, 0.5, 0.8]

HERE = os.path.dirname(__file__)
output_path = os.path.join(HERE, "scaletest_96k5iter_v2_results.pkl")

combos = [
    ((p_I, p_I, p_I, p_I), (ph, ph, ph, ph))
    for p_I, ph in itertools.product(P_I_VALUES, PH_VALUES)
]
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

times_df = pd.DataFrame(combo_times)
times_path = output_path.replace(".pkl", "_timings.csv")
times_df.to_csv(times_path, index=False)

total = time.time() - t_start
print(f"\n{'='*60}\nTIMING REPORT  ({N_ITER} iterations per combination, n={N})\n{'='*60}")
print(times_df.to_string(index=False))
print(f"\nAverage simulation time per combination: "
      f"{times_df['seconds'].mean():.1f}s "
      f"(min {times_df['seconds'].min():.1f}s, "
      f"max {times_df['seconds'].max():.1f}s)")
print(f"Total elapsed over {n_combos} combinations: {total:.1f}s")
print(f"Saved results to {output_path}")
print(f"Saved timings to {times_path}")
