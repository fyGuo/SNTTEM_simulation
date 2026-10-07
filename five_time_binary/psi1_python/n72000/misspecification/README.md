# Blip-misspecification simulation, oracle nuisances, n = 72000 (five time points, psi1)

Asks what happens to the three estimators (Three-step-g, Three-step-ipw,
Robins') when **some of the later blip-down models are misspecified**. To
isolate that from nuisance-estimation error, every nuisance (`ps0..ps4`,
`mu05..mu45`) is the exact closed-form value implied by the DGP -- no ML, no
parametric working model, no cross-fitting -- exactly as in `../oracle/`.

## Status: strong misspecification, run 2026-10-01

Only the **fitted** blips are misspecified: `gamma15..gamma45` in
`generate_data.py` (the functions `estimators.py` imports) are
`psi * (-L - 0.5)`, while the DGP's `log_p` keeps the true `psi * (1 + L)`.
`gamma05` is correct. Because the DGP is unchanged, `oracle_nuisances.py`
is still exact and needed no re-derivation. An earlier, milder choice
`psi * (L + 10)/(L + 5)` (near-constant on L in [-1.5, -0.5]) gave Robins'
psi05 medians of 1.08-1.47; it was replaced to make the bias more visible.

Full grid (300 iterations, ~45s): Robins' psi05 median is 1.79-2.43 at
`p_I=1` (coverage 15-71%) and 1.22-1.56 at `p_I=0.2` (coverage 83-91%);
Three-step-g/-ipw stay at 0.95-1.02 with ~94-95% coverage, since their
psi05 equation only uses constant-treatment paths and never applies the
later blips. psi15..psi45 no longer target 1, so `check_results.py`'s
coverage/MSE for those is not meaningful here.

`plot_psi05_causalab.py` writes `psi05_misspecification_causalab.pdf`
(SNTTEM = Robins', "Three-step" = Three-step-ipw; median +/- 1.96 robust SD).

## Run

```bash
cd psi1_python/n72000/misspecification && ./run.sh
```

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
