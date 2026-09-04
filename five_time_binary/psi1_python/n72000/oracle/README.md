# Oracle-nuisance simulation, n = 72000 (five time points, psi1)

Non-null (psi05..psi45 = 1.0) counterpart to
`../../../psi0_python/n72000/oracle/`. Same idea: every nuisance (`ps0..ps4`,
`mu05..mu45`) is the EXACT closed-form value implied by the known DGP, for
**all three estimators** (Three-step-g, Three-step-ipw, Robins') -- no ML, no
parametric working model, no cross-fitting.

## What's different from psi0's oracle/

Unlike psi0 (where psi=0 makes Y an unconditional constant, so
`mu05==mu15==...==mu45==exp(-1.5)`), psi1's mu's are **not constant** --
Y genuinely depends on treatment history. `oracle_nuisances.py`'s docstring
derives the closed form: each `mu_{j5}` is the DGP's `exp(log_p)` truncated
to only the blip terms *before* step j (the recursive `A_j==A_{j-1}` masking
in `working_model()` exactly zeroes out step j's own blip term at every
level of the recursion). Propensity scores (`ps0..ps4`) are unchanged from
psi0's derivation -- `A_t`'s generation doesn't depend on psi at all.

This derivation was verified, not just derived: `check_oracle_derivation.py`
checks the per-observation mean of all three estimating equations is ~0 at
the true theta = (1,1,1,1,1) on a 5,000,000-row draw (both `p_I` values,
`ph=0.3`) -- all |z| < 2. The full 6-cell/300-iteration production run below
confirms it further: psi05 median lands at 0.95-1.02 (target 1.0) across all
cells, ~94-96% coverage.

`generate_data.py` and `estimators.py` are copies of `../`'s (intercept
-2.6, corrected `ee_three_step_ipw` -- see CLAUDE.md).

## Run

```bash
cd psi1_python/n72000/oracle && ./run.sh
```

No fitting means this runs in minutes, not hours -- the full 6-cell grid took
~260s here (a bit slower than psi0's oracle ~40s since fsolve has to find
roots near 1.0 rather than 0.0, and the mu's require more arithmetic per row,
but still trivial next to `../ML_nuisance/`'s ~13h).

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
