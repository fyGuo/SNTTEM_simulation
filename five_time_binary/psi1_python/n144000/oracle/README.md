# Oracle-nuisance simulation, n = 144000, HARD CELL ONLY (five time points, psi1)

Counterpart to `../ML_nuisance/` at the same hard cell (`p_I=0.2, ph=0.3`,
n=144000): every nuisance (`ps0..ps4`, `mu05..mu45`) is the EXACT
closed-form value implied by the known DGP, for all three estimators —
no ML, no parametric working model, no cross-fitting. Isolates whether the
n=72000 hard cell's instability (see
`../../n72000/ML_nuisance/README.md` and CLAUDE.md "Known issues") is
inherent to the estimating equations themselves at this `ph`, or mostly an
artifact of nuisance-estimation noise that more data (via `../ML_nuisance/`)
can fix.

`generate_data.py` and `estimators.py` here are copies of
`../../n72000/oracle/`'s (intercept -2.6, corrected `ee_three_step_ipw` —
see CLAUDE.md).

## Run

```bash
cd psi1_python/n144000/oracle && ./run.sh
```

No fitting means this runs in well under a minute for a single cell — the
full 6-cell n=72000 grid took ~260s in `../../n72000/oracle/`.

## Outputs

- `simulation_results.pkl` / `.csv` — per-iteration estimates (900 rows =
  1 cell x 3 methods x 300 iterations)
- `simulation_results_timings.csv` — wall time for the one cell
- `sim_run.log` — progress, failures, timing report

## After the run

```bash
PYTHON=../../../.venv/bin/python
$PYTHON check_results.py   # prints per-cell median/variance/MSE/coverage,
                            # writes simulation_results_plots.pdf
```

Compare against the same cell's row in `../../n72000/oracle/`'s output
(`p_I1..p_I4=0.2`, `ph1..ph4=0.3`) to see whether variance/MSE/coverage
improved at n=144000, purely from the estimating-equation side.
