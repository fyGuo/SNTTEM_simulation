"""Outer loop: run simulation over a grid of (ph, p_I) values.

Equivalent to run_Rscript.R. Results are saved to simulation_results.pkl
(and also as simulation_results.csv for easy inspection).
"""
import os
import time
import pandas as pd

from simulation_output import run_simulation

# True parameter values
PSI03 = 1.0
PSI13 = 1.0
PSI23 = 1.0
SIGMA = 0

# Simulation settings
N = 4000
N_ITER = 100
SEED = 3411

# Grid matching run_Rscript.R:
#   p1 (= ph):   probability that A_t stays the same as A_{t-1}
#   p_I1 (= p_I): probability of being eligible (I=1)
PH_VALUES = [0.4, 0.6, 0.8]
P_I_VALUES = [ 0.25, 0.5, 0.75, 1]

output_path = os.path.join(os.path.dirname(__file__), "simulation_results.pkl")

all_results = []
t_start = time.time()

for ph in PH_VALUES:
    for p_I in P_I_VALUES:
        print(f"\n{'='*60}")
        print(f"Running: ph={ph}, p_I={p_I}")
        print(f"{'='*60}")
        t0 = time.time()

        est = run_simulation(
            n=N,
            n_iter=N_ITER,
            psi03=PSI03,
            psi13=PSI13,
            psi23=PSI23,
            p_I=p_I,
            ph=ph,
            sigma = SIGMA,
            seed=SEED,
            n_jobs=-1,
        )
        all_results.append(est)
        print(f"Done in {time.time() - t0:.1f}s")

results = pd.concat(all_results, ignore_index=True)
results.to_pickle(output_path)
results.to_csv(output_path.replace(".pkl", ".csv"), index=False)

print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")
print(f"Saved to {output_path}")
