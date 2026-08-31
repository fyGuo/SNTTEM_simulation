"""Outer loop: run simulation over a grid of (ph, p_I) values.

Equivalent to run_Rscript.R. Results are saved to simulation_results.pkl
(and also as simulation_results.csv for easy inspection).
"""
import os
import time
import pandas as pd

from simulation_output import run_simulation

# True parameter values (psi1 setting: all blips active at psi = 1)
PSI03 = 1.0
PSI13 = 1.0
PSI23 = 1.0

# Simulation settings
N = 24000
N_ITER = 300
SEED = 3411

# Grid (misspecification study): only two settings.
#   ph1 = ph2 = 0.5 fixed (per-time-point probability A_t stays the same as A_{t-1})
#   p_I1 = p_I2 in {0.2, 1} paired (per-time-point probability of being eligible)
PI_VALUES = [0.2, 1]      # p_I1 and p_I2 always taken equal
PH1 = 0.5
PH2 = 0.5

# Blip misspecification combos: (gamma13_misspec, gamma23_misspec).
#   (False, False) -> both correct (correctly-specified baseline reference)
#   (False, True)  -> correct gamma13, wrong gamma23
#   (True,  False) -> wrong gamma13,   correct gamma23
#   (True,  True)  -> both wrong
MISSPEC_COMBOS = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]

output_path = os.path.join(os.path.dirname(__file__), "simulation_results.pkl")

settings = [(pI, pI, PH1, PH2) for pI in PI_VALUES]
combos = [(p_I1, p_I2, ph1, ph2, g13m, g23m)
          for (p_I1, p_I2, ph1, ph2) in settings
          for (g13m, g23m) in MISSPEC_COMBOS]
n_combos = len(combos)

all_results = []
combo_times = []
t_start = time.time()

for k, (p_I1, p_I2, ph1, ph2, g13m, g23m) in enumerate(combos, start=1):
    g13_lbl = "wrong" if g13m else "correct"
    g23_lbl = "wrong" if g23m else "correct"
    print(f"\n{'='*60}")
    print(f"[{k}/{n_combos}] Running: p_I1={p_I1}, p_I2={p_I2}, "
          f"ph1={ph1}, ph2={ph2} | gamma13={g13_lbl}, gamma23={g23_lbl}")
    print(f"{'='*60}")
    t0 = time.time()

    est = run_simulation(
        n=N,
        n_iter=N_ITER,
        psi03=PSI03,
        psi13=PSI13,
        psi23=PSI23,
        p_I1=p_I1,
        p_I2=p_I2,
        ph1=ph1,
        ph2=ph2,
        seed=SEED,
        n_jobs=-1,
        gamma13_misspec=g13m,
        gamma23_misspec=g23m,
    )
    all_results.append(est)
    elapsed = time.time() - t0
    combo_times.append(
        {"p_I1": p_I1, "p_I2": p_I2, "ph1": ph1, "ph2": ph2,
         "gamma13": g13_lbl, "gamma23": g23_lbl, "seconds": elapsed}
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
