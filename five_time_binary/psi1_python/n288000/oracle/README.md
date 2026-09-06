# Oracle-nuisance simulation, n = 288000, HARD CELL ONLY (five time points, psi1)

Counterpart to `../ML_nuisance/` at the same hard cell (`p_I=0.2, ph=0.3`,
n=288000): every nuisance (`ps0..ps4`, `mu05..mu45`) is the EXACT
closed-form value implied by the known DGP, for all three estimators —
no ML, no parametric working model, no cross-fitting. Third point in the
n=72000 → n=144000 → n=288000 progression on this cell; see
`../../n144000/oracle/README.md` and CLAUDE.md "Status: n=144000 hard-cell
check" for the n=144000 result (variance ~2.5-3x lower than n=72000,
coverage unchanged ~94-95%, median unchanged from n=72000's estimating-
equation-level bias).

`generate_data.py` and `estimators.py` here are copies of
`../../n72000/oracle/`'s (intercept -2.6, corrected `ee_three_step_ipw` —
see CLAUDE.md).

## Run

```bash
cd psi1_python/n288000/oracle && ./run.sh
```

No fitting means this runs in well under a minute for a single cell even
at n=288000 (n=144000 took 15s).

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

Compare against the same cell's row in `../../n72000/oracle/` and
`../../n144000/oracle/` output (`p_I1..p_I4=0.2`, `ph1..ph4=0.3`) to
continue the variance-vs-n trend at the estimating-equation level.
