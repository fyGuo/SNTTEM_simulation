"""Full 300-iteration confirmatory run, n=72000, ph=0.3 cells only.

NOT the production run (that's run_simulation.py, n=24000, full 6-cell grid).
Follow-up to run_scaletest_48k_ph03_full.py and run_scaletest_96k_ph03_full.py:
at ph=0.3, n=96000 gave 0% psi05 failure rate and low variance; n=48000 also
gave 0% failure rate but psi05 variance was ~3.4-3.5x higher than at n=96000
(more than the ~2x that 1/n scaling alone would predict -- consistent with
the three-step weight being a product of four w_t terms, so halving n roughly
halves the count of constant-treatment-path subjects too). This run checks
whether an intermediate n=72000 recovers most of n=96000's stability at less
runtime cost, or whether the extra precision requires close to the full
96000.

Uses the current three-method estimator set (Three-step-g, Three-step-ipw,
Robins'). Writes to separate output files so the existing n=24000 production
results (simulation_results.pkl/.csv) and the other scale-test probes are
untouched.
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

N = 72000
N_ITER = 300
SEED = 3411

P_I_VALUES = [0.2, 1]
PH_VALUES = [0.3]  # intermediate check between the 48k and 96k ph=0.3 runs

HERE = os.path.dirname(__file__)
output_path = os.path.join(HERE, "scaletest_72k_ph03_300iter_results.pkl")

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
