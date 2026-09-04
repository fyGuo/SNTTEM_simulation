# ML-nuisance simulation, n = 72000 (five time points, psi0)

This is the original `psi0_python/n72000/` production pipeline, moved into
its own subfolder so it sits alongside `../oracle/` (exact-closed-form
nuisances) for comparison.

**Rerun post-`ee_three_step_ipw`-fix, completed 2026-09-03.** The run
present here now is not the original pre-fix run from the move -- it was
relaunched (started 2026-09-01 17:08, interrupted after ~1.4 cells when the
session ended; relaunched again 2026-09-02 19:41 under `nohup`, ran to
completion 2026-09-03 14:01, elapsed 18h20m) to reflect the corrected
estimating equation. See CLAUDE.md "Status" for the resulting headline
numbers -- Three-step-ipw's psi05 bias dropped from the old +0.11/+0.17 to
roughly +0.004 to +0.009 across cells.

## What this is

Nuisances (`ps0..ps4`, `mu05..mu45`) are fit from data with the RF/GBM/poly
ensemble in `working_models_ml.py` (cross-fitted), used by all three
estimators (Three-step-g, Three-step-ipw, Robins'). This is the pipeline that
originally showed Three-step-ipw's psi05/psi15/psi25 bias at low `ph` --
see the top-level CLAUDE.md "Known issues" and `../check_ipw_ee_oracle*.py`
for the diagnosis (an `ee_three_step_ipw` equation bug, since fixed, plus a
separate, still-present lack of double robustness in `ps_t`).

Self-contained like every other experiment folder in this tree:
`generate_data.py` and `estimators.py` here are copies of `../`'s (so this
reflects the corrected `ee_three_step_ipw`).

## Run

```bash
cd psi0_python/n72000/ML_nuisance && ./run.sh
```

Full 6-cell grid took ~13.25h in the original (pre-fix) run and ~18.3h in
the post-fix rerun described above (see `simulation_results_timings.csv` in
this folder for the current run's per-cell breakdown) -- the difference is
most likely ordinary run-to-run variance (thread oversubscription, per
CLAUDE.md "Known issues"), not something caused by the equation fix itself.

## Outputs (already present, from the post-fix rerun)

- `simulation_results.pkl` / `.csv` -- per-iteration estimates (5400 rows =
  6 cells x 3 methods x 300 iterations)
- `simulation_results_timings.csv` -- per-cell wall time
- `simulation_results_plots.pdf` -- var/median/MSE/coverage, from `check_results.py`
- `sim_run*.log`, `nohup_run72k.out` -- run logs

## After a rerun

```bash
PYTHON=../../../.venv/bin/python
$PYTHON check_results.py   # prints per-cell median/variance/MSE/coverage,
                            # writes simulation_results_plots.pdf
```
