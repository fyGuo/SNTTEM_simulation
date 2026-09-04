# ML-nuisance simulation, n = 144000, HARD CELL ONLY (five time points, psi1)

Sample-size check on the single hardest cell from
`../../n72000/ML_nuisance/`'s 6-cell production grid: `p_I=0.2, ph=0.3`. At
that cell only `0.3^4 ≈ 0.81%` of subjects follow a constant-treatment path
(about 584 of 72000), with three-step weights up to `(1/0.3)^4 ≈ 123` — see
CLAUDE.md "Known issues" for the full picture. This folder doubles the
sample size to n=144000, keeping everything else (300 iterations, seed
3411, the same fitted-nuisance pipeline) identical to the n=72000 run, to
see whether n=144000 is enough to stabilize psi05/psi15/psi25's bias and
variance at this cell, or whether the instability is more fundamental
(low-`ph` positivity, not just finite-`n` noise).

Run alongside `../oracle/` (same hard cell, exact closed-form nuisances,
n=144000) to separate two questions: does more data help the estimating
equations themselves (oracle), and does it help nuisance estimation on top
of that (ML_nuisance)?

## What this is

Nuisances (`ps0..ps4`, `mu05..mu45`) are fit from data with the RF/GBM/poly
ensemble in `working_models_ml.py` (cross-fitted), used by Three-step-g and
Robins'; Three-step-ipw gets `working_model_true()` — a correctly-specified
plain logistic/linear working model with the same covariate sets, instead.

`generate_data.py` and `estimators.py` here are copies of
`../../n72000/ML_nuisance/`'s (intercept -2.6, corrected `ee_three_step_ipw`
— see CLAUDE.md). The -2.6 intercept's validity is a per-observation
probability bound, independent of n, so it applies unchanged here.

## Run

```bash
cd psi1_python/n144000/ML_nuisance && ./run.sh
```

Single cell instead of six, but at double `n` — RF/GBM training doesn't
scale linearly, so expect this to run several hours longer than the ~2.8h
the same cell took at n=72000 inside the 6-cell grid (see
`../../n72000/ML_nuisance/simulation_results_timings.csv` for that
per-cell number). Not yet measured standalone at n=144000.

## Outputs

- `simulation_results.pkl` / `.csv` — per-iteration estimates (900 rows =
  1 cell x 3 methods x 300 iterations)
- `simulation_results_timings.csv` — wall time for the one cell
- `sim_run*.log` — run logs

## After the run

```bash
PYTHON=../../../.venv/bin/python
$PYTHON check_results.py   # prints per-cell median/variance/MSE/coverage,
                            # writes simulation_results_plots.pdf
```

Compare against the same cell's row in `../../n72000/ML_nuisance/`'s output
(`p_I1..p_I4=0.2`, `ph1..ph4=0.3`) to see whether variance/MSE/coverage
improved at n=144000.
