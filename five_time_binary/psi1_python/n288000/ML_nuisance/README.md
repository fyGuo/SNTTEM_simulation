# ML-nuisance simulation, n = 288000, HARD CELL ONLY (five time points, psi1)

Continuation of the sample-size progression on the production grid's
hardest cell (`p_I=0.2, ph=0.3`): `../../n72000/ML_nuisance/` (production,
6-cell grid) → `../../n144000/ML_nuisance/` (this cell only, 2x n) → this
folder (4x n). `../../n144000/ML_nuisance/README.md` has the full context
on why this cell is hard; the short version is only `0.3^4 ≈ 0.81%` of
subjects follow a constant-treatment path, with three-step weights up to
`(1/0.3)^4 ≈ 123`.

n=144000's result (see CLAUDE.md "Status: n=144000 hard-cell check"):
variance dropped ~3-4x from n=72000 with medians essentially unchanged, and
Three-step-ipw's ML-fitted variance (0.53) nearly matched its oracle
variance (0.52) — but Three-step-g/Robins' still ran ~3x higher variance
than their oracle counterparts (1.65/1.81 vs 0.51/0.73), meaning the
RF/GBM/poly ensemble was still contributing real noise at n=144000. This
folder checks whether n=288000 closes that remaining gap for
Three-step-g/Robins', and continues tracking whether more `n` moves the
Three-step estimators' median any closer to the target 1.0 (it was ~0.91-0.92
at both n=72000 and n=144000, essentially unchanged, suggesting this bias is
not a sample-size artifact).

Run alongside `../oracle/` (same hard cell, exact closed-form nuisances,
n=288000).

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
cd psi1_python/n288000/ML_nuisance && ./run.sh
```

n=144000 took 6h32m for this one cell (RF/GBM training doesn't scale
linearly with n, so this — 2x that n again — likely takes noticeably
longer; not yet measured).

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

Compare against the same cell's row in `../../n72000/ML_nuisance/` and
`../../n144000/ML_nuisance/` output (`p_I1..p_I4=0.2`, `ph1..ph4=0.3`) to
continue the variance-vs-n trend, and against `../oracle/`'s numbers at the
same n to see how much nuisance-estimation gap remains for Three-step-g/
Robins' specifically.
