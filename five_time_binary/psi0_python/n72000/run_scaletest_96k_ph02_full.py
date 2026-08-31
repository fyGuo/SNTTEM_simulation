"""Full 300-iteration confirmatory run, n=96000, ph=0.2 cells only.

NOT the production run (that's run_simulation.py, n=24000, full 6-cell grid).
This targets specifically the two cells that were unstable at n=24000
(p_I=0.2 and p_I=1, both ph=0.2 -- see check_results.py / CLAUDE.md "Known
issues": at ph=0.2 only a tiny fraction of subjects follow a constant
treatment path, so the three-step/robins weights blow up). The question is
whether n=96000 (4x the subjects, so ~4x as many constant-treatment-path
subjects) stabilizes these two cells enough that check_results.py's 5%
failure-rate threshold no longer drops them.

Uses the current three-method estimator set (Three-step-g, Three-step-ipw,
Robins'). Writes to separate output files so the existing n=24000 production
results (simulation_results.pkl/.csv) and the smaller scale-test probes are
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

N = 96000
N_ITER = 300
SEED = 3411

P_I_VALUES = [0.2, 1]
PH_VALUES = [0.2]  # only the unstable cells

HERE = os.path.dirname(__file__)
output_path = os.path.join(HERE, "scaletest_96k_ph02_300iter_results.pkl")

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
