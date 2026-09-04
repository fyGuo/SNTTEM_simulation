# ML-nuisance simulation, n = 72000 (five time points, psi1)

Non-null (psi05..psi45 = 1.0) counterpart to
`../../../psi0_python/n72000/ML_nuisance/`. This is the original
`psi1_python/n72000/` production pipeline, moved into its own subfolder so it
sits alongside `../oracle/`, mirroring the psi0 restructuring.

## What this is

Nuisances (`ps0..ps4`, `mu05..mu45`) are fit from data with the RF/GBM/poly
ensemble in `working_models_ml.py` (cross-fitted), used by Three-step-g and
Robins'; Three-step-ipw gets `working_model_true()` -- a correctly-specified
plain logistic/linear working model with the same covariate sets, instead.

**Rerun post-`ee_three_step_ipw`-fix, completed 2026-09-02.** An earlier run
in this folder (finished 2026-09-01 11:59, 11h39m) predated the fix and its
Three-step-ipw numbers reflected the old, buggy equation. It was
superseded: `run.sh` was relaunched 2026-09-01 20:41 with the corrected
`estimators.py` and ran to completion 2026-09-02 13:37 (16h56m). The
results now in this folder are from that post-fix run.

Self-contained like every other experiment folder in this tree:
`generate_data.py` and `estimators.py` here are copies of `../`'s (intercept
-2.6, corrected `ee_three_step_ipw`).

## Run

```bash
cd psi1_python/n72000/ML_nuisance && ./run.sh
```

Full 6-cell grid took ~11.75h in the pre-fix run and ~16.9h in the post-fix
rerun described above (see `simulation_results_timings.csv` in this folder
for the current run's per-cell breakdown) -- the fix itself doesn't change
the RF/GBM ensemble's cost, so the difference is most likely ordinary
run-to-run variance (thread oversubscription, per CLAUDE.md "Known issues").

## Outputs (already present, from the post-fix rerun)

- `simulation_results.pkl` / `.csv` -- per-iteration estimates (5400 rows =
  6 cells x 3 methods x 300 iterations)
- `simulation_results_timings.csv` -- per-cell wall time
- `sim_run*.log` -- run logs

## After a rerun

```bash
PYTHON=../../../.venv/bin/python
$PYTHON check_results.py   # prints per-cell median/variance/MSE/coverage,
                            # writes simulation_results_plots.pdf
```
