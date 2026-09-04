# n = 24000 simulation (continuous outcome, psi1 DGP)

Self-contained copy of the continuous-outcome SNTTEM simulation, configured for:

- `N = 24000`, `N_ITER = 300`, full 36-cell grid (p_I1,p_I2 in {0.2,1}; ph1,ph2 in {0.2,0.5,0.8})
- `n_boot = 200`, GMM included, `SEED = 3411`
- True parameters `PSI03 = PSI13 = PSI23 = 1.0`, `SIGMA = 0`

The data-generating process is the original `psi1_python` continuous DGP
(`Y ~ Normal(mu_Y, sigma*(A0+A1+A2))`); only the eligibility/persistence
probabilities are tuned more finely, per time point:

- `p_I1`, `p_I2`: probability of being eligible (I1=1, I2=1)
- `ph1`, `ph2`: probability that the treatment stays the same (A1=A0, A2=A1)

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
$PYTHON check_results.py        # psi03 median/variance/MSE/coverage (faceted bars)
```

## Note on runtime

24000 is ~2x the data of the n=12000 run, so expect roughly ~2x the per-cell
time, depending on cores/CPU.
