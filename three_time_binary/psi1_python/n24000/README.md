# n = 24000 simulation (psi = 1 mechanism, psi0-style grid)

Self-contained copy of the SNTTEM simulation for the **psi = 1** data-generating
mechanism (true `psi03 = psi13 = psi23 = 1`), restructured to use the **psi0-style
independent per-time-point grid** (`p_I1, p_I2, ph1, ph2`).

- `N = 24000`, `N_ITER = 300`, `SEED = 3411`, `n_boot = 200`, GMM included
- 36-cell grid: `p_I1, p_I2 in {0.2, 1}`, `ph1, ph2 in {0.2, 0.5, 0.8}`
  (matches `psi0_python/n24000` exactly, so the psi=1 study is directly
  comparable to the psi=0 study)

This uses psi0's `generate_data.py` / `simulation_output.py` (the only real
difference from psi1's original code was the single-`ph`/`p_I` parametrization;
the outcome model, covariates, and blip functions are identical). The psi=1
value is set in `run_simulation.py` (PSI03 = PSI13 = PSI23 = 1.0).

> The previous single-`ph`/`p_I` 12-cell run and its reports are preserved in
> `_archive_12cell/`.

## Run

```bash
cd n24000
PYTHON=/path/to/python ./run.sh
```

## Reports (after the run)

```bash
$PYTHON regen_bar_reports.py        # bar plots (value labels): summary, variance, bias, failure
$PYTHON regen_qqhist_allvalues.py   # QQ + histogram (all values)
$PYTHON bias_zscores.py             # bias / sqrt(var/300) z-score CSVs (psi03/13/23)
$PYTHON gmm_failure_report.py       # GMM failure table
```

## Note on runtime

Same as psi0/n24000 (36 cells x 300 iter at n=24000): ~11-12 hr of awake compute,
peak RAM ~8 GB with 12 workers.
