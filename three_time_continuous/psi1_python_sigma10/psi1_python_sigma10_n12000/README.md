# n = 12000 simulation (continuous outcome, psi1 DGP, sigma = 10)

Self-contained copy of the continuous-outcome SNTTEM simulation, configured for:

- `N = 12000`, `N_ITER = 300`, full 36-cell grid (p_I1,p_I2 in {0.2,1}; ph1,ph2 in {0.2,0.5,0.8})
- `n_boot = 200`, GMM included, `SEED = 3411`
- True parameters `PSI03 = PSI13 = PSI23 = 1.0`, **`SIGMA = 10`**

Same settings as the `psi1_python/n24000/` run, except `N = 12000` and the
outcome noise scale `SIGMA = 10` (so `Y ~ Normal(mu_Y, sigma*(A0+A1+A2))` is
genuinely noisy rather than deterministic). Eligibility/persistence are tuned
per time point:

- `p_I1`, `p_I2`: probability of being eligible (I1=1, I2=1)
- `ph1`, `ph2`: probability that the treatment stays the same (A1=A0, A2=A1)

## Run

```bash
cd psi1_python_sigma10_n12000
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
$PYTHON check_results.py        # psi03 median/variance/MSE/coverage (faceted bars)
$PYTHON make_report_pdfs.py     # separate bias + variance (IQR & 2.5/97.5) PDFs
```

## Note on runtime

12000 is ~half the data of the n=24000 run, so expect roughly half the per-cell
time (n=24000 averaged ~16.7 min/cell, ~10 hr total, on 12 workers).
