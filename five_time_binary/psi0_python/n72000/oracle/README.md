# Oracle-nuisance simulation, n = 72000 (five time points, psi0)

Full-grid counterpart to `../check_ipw_ee_oracle.py` / `../check_ipw_ee_oracle_mc.py`.
Those scripts checked one cell (`p_I=0.2,1.0`, `ph=0.3`) and only Three-step-ipw.
This folder productionizes the same idea to the full production
6-cell x 300-iteration grid, for **all three estimators**
(Three-step-g, Three-step-ipw, Robins').

## What's different from `../`

Every nuisance -- `ps0..ps4`, `mu05..mu45` -- is the EXACT closed-form value
implied by the known DGP (`oracle_nuisances.py`), not a fitted model:

- `ps0 = expit(-1 + L0)`
- `ps_t = ph_t*A_{t-1} + (1-ph_t)*(1-A_{t-1})` for t=1..4 (A_t doesn't depend
  on L_t at all in this DGP)
- `mu05 = mu15 = mu25 = mu35 = mu45 = exp(-1.5)` (Y is Bernoulli(exp(-1.5)),
  independent of everything, since psi05..psi45 = 0 here)

No ML ensemble, no plain logistic/linear working models, no cross-fitting --
`simulation_output.py` calls `oracle_nuisances()` directly on the full sample
and hands the same nuisance columns to all three estimating equations. This
removes nuisance-estimation error entirely, isolating each estimating
equation's own finite-sample bias/variance/coverage.

`generate_data.py` and `estimators.py` are byte-identical copies of `../`'s
(so this reflects the corrected `ee_three_step_ipw` -- see CLAUDE.md).

## Run

```bash
cd psi0_python/n72000/oracle && ./run.sh
```

No RF/GBM fitting means this should run much faster than the ~13h ML-nuisance
production run in `../` -- expect low minutes per cell rather than ~100-165.

## Outputs

- `simulation_results.pkl` / `.csv` -- per-iteration estimates
- `simulation_results_timings.csv` -- per-cell wall time
- `sim_run.log` -- progress, failures, timing report

## After the run

```bash
PYTHON=../../../.venv/bin/python
$PYTHON check_results.py   # prints per-cell median/variance/MSE/coverage,
                            # writes simulation_results_plots.pdf
```
