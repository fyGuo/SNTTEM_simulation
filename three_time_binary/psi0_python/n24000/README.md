# n = 24000 simulation (run on the second Mac)

Self-contained copy of the SNTTEM simulation, configured for:

- `N = 24000`, `N_ITER = 300`, full 36-cell grid (p_I1,p_I2 in {0.2,1}; ph1,ph2 in {0.2,0.5,0.8})
- `n_boot = 200`, GMM included, `SEED = 3411` (matches the n=6000 / n=12000 runs)

## Run

```bash
cd n24000
PYTHON=/path/to/python ./run.sh      # e.g. PYTHON=/Users/<you>/opt/anaconda3/bin/python
# or, if `python3` already has the deps:
./run.sh
```

Dependencies (see `requirements.txt`): numpy, pandas, scipy, scikit-learn, joblib, matplotlib.

## Outputs (written here)

- `simulation_results.pkl` / `.csv`     -- per-iteration estimates
- `simulation_results_timings.csv`      -- per-cell wall time + average
- `sim_run.log`                         -- progress, failures, timing report

## Figures (after the run)

```bash
$PYTHON make_report_pdfs.py     # variance (IQR + 2.5/97.5), bias, failure proportion
$PYTHON distribution_plots.py   # QQ + histograms
$PYTHON check_results.py        # psi03 median/variance/MSE/coverage
$PYTHON gmm_failure_report.py   # GMM failure distribution
```

## Note on runtime

24000 is ~2x the data of the n=12000 run, so expect roughly ~2x the per-cell
time (~12 hr total on a comparable Mac, depending on cores/CPU).
