# SNTTEM simulation — five time points

Monte Carlo study of structural nested treatment-eligibility models, extended
from three time points to **five**. Ported from `../three_time_binary/`, which
remains untouched as the K=3 reference.

## Where the work is

**`psi0_python/n72000/` is the only directory ported to five time points.**
Everything else in this tree is still K=3 code and will break if run:

| Path | State |
|---|---|
| `psi0_python/n72000/` | **K=5, current** (n=72000; renamed from `n24000/` when the production sample size moved to 72000) |
| `psi0_python/n72000/misspecification/` | K=3, not being run |
| `psi0_python/` (parent scripts), `psi0_python/n12000/` | K=3 |
| `psi1_python/` (all) | K=3 |

The original n=24000 production run script (`ph` in `{0.2, 0.5, 0.8}`) is
preserved inside `n72000/` as `run_simulation_n24000.py`, so the current
`run_simulation.py` / `simulation_results.*` names are free for the n=72000
config. Its output (`simulation_results_n24k.{pkl,csv,...}`) and the
`run_scaletest_96k*.py` scale-test outputs (`scaletest_96k*`,
`psi05_96k*`) were deleted 2026-08-31 to reclaim space now that the
n=72000 production run has completed and superseded them — the generating
scripts (`run_simulation_n24000.py`, `run_scaletest_96k*.py`,
`run_scaletest_72k_ph03_full.py`, `run_scaletest_48k_ph03_full.py`) are
still there if those artifacts are needed again.

## Status: n=72000 production run complete (2026-08-31)

The 6-cell grid (`p_I` ∈ {0.2, 1.0}, `ph` ∈ {0.3, 0.5, 0.8}) finished in
**47,697 s ≈ 13.25 h** wall time. Per-cell timings are in
`simulation_results_timings.csv`; results are in `simulation_results.{csv,pkl}`
(5400 rows = 6 cells × 3 methods × 300 iterations).

Post-run summary/plots come from `check_results.py`, which writes
`simulation_results_plots.pdf` (4 pages per ψ: variance, median, MSE,
coverage) and prints a per-cell table (median, robust variance, MSE, 95%
coverage) to stdout. Bar order in the plots is fixed as **Simple g-estimator,
Three-step-g estimator, Three-step-ipw estimator**.

`check_results.py`'s `rename_methods()`/`METHODS` used to key off a method
string literally named `"Three-step"`, which no longer matches the actual
`estimators.py` output names `"Three-step-g"` / `"Three-step-ipw"` — the
`df["method"].isin(METHODS)` filter silently dropped both three-step
estimators, leaving only Robins' in the table and plot. Fixed by mapping
`"Three-step-g"` → `"Three-step-g estimator"` and `"Three-step-ipw"` →
`"Three-step-ipw estimator"` explicitly. If `check_results.py` output is ever
missing a method again, check this mapping first before assuming a data
problem.

Headline pattern from this run: Robins' (Simple g-) and Three-step-g are both
close to unbiased with ~94–95% coverage across all 6 cells. Three-step-ipw
has a clear positive bias in psi05/psi15/psi25 that is worst at `ph=0.3` and
shrinks as `ph` rises — the documented nuisance-estimation artifact (lack of
double robustness in `ps_t`), not an equation bug. All three methods converge
to near-identical estimates for psi35/psi45 (psi45 rows are numerically
identical across methods per cell, since that equation has zero compounded
weight factors).

## Environment — read this before running anything

Use the project venv. **Never bare `python3`**: on both machines that resolves to
Homebrew 3.14, which has no pandas, and the run will fail on import.

```
five_time_binary/.venv/bin/python     # Python 3.13.15, arm64
```

`requirements-lock.txt` (repo root) is the source of truth — 17 exactly pinned
packages: numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.0,
joblib 1.5.3, matplotlib 3.11.1. `psi0_python/n24000/requirements.txt` has loose
`>=` bounds and is **not** what defines the environment.

Rebuild from scratch:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

`python@3.13` is `brew pin`ned on both machines — a blanket `brew upgrade` would
otherwise move the interpreter out from under the venv and break it.

### Two machines

| | Path |
|---|---|
| This Mac | `/Users/fuyuguo/PhD_thesis/Thesis_1_SNTTEM/Codes/five_time_binary/` |
| `fug183@100.107.76.80` (`HSPH-YVWHQMWMDV`) | `/Users/fug183/PhD_thesis/Thesis_1_SNTTEM/Codes/five_time_binary/` |

Both are M4 Pro / 12 cores / arm64 with byte-identical package sets. Verified:
identical seed gives identical data and, with forest threading held fixed,
identical estimates.

Sync code (never the venv — it hardcodes local paths):

```bash
rsync -av --exclude='.venv/' --exclude='__pycache__/' --exclude='.DS_Store' \
  <local>/five_time_binary/ fug183@100.107.76.80:~/PhD_thesis/Thesis_1_SNTTEM/Codes/five_time_binary/
```

## Running

```bash
cd psi0_python/n72000 && ./run.sh
```

`run.sh` resolves the venv relative to its own location, preflights the six
imports, runs under `caffeinate -i`, and tees to `sim_run.log` (rotating any
previous log). At n=72000, the 6-cell grid takes **~13.25 h** total (measured
2026-08-31, see `simulation_results_timings.csv`): ph=0.3 cells ~99–105 min,
ph=0.5 cells ~133–138 min, ph=0.8 cells ~153–167 min. The earlier n=24000
config ran in ~2–3 h (~3.3 s/iteration, ~16 min/cell).

## Conventions specific to K=5

- **Parameters are `psi05, psi15, psi25, psi35, psi45`.** The trailing digit is
  the outcome time (5), not the time index. Blips: `gamma05 … gamma45`.
- Nuisances: `ps0..ps4`, `mu05..mu45`, pseudo-outcomes `dr_mu15..dr_mu45`.
- **No GMM.** `gmm_combine()` still exists in `estimators.py` but is never called;
  `simulation_output.py` and `run_simulation.py` must not compute it. The
  per-iteration bootstrap was removed with it.
- Grid holds `p_I` and `ph` **common across t=1..4** — a literal per-time-point
  factorial would be `2^4 × 3^4 = 1296` cells. Default is 6 cells; add
  heterogeneous profiles via `EXTRA_SCENARIOS` in `run_simulation.py`.

## Code style — important

The time-indexed code is **deliberately written out explicitly, one block per
time point**. Do not refactor `generate_data.py`, `estimators.py`, or
`working_models_ml.py` into loops over `t` or arrays of parameters, even though
the structure is obviously regular. This is a standing preference; readability
against the written model matters more than concision here.

## Estimating equations

Both estimators follow one telescoping pattern. With `mu_5 := Y`,
`ΔA_t := A_t − A_{t-1}` (`A_{-1} := 0`), and
`w_t = 1{A_t = A_{t-1}} / P(A_t = A_{t-1} | L_t, A_{t-1})`, for j = 0..4:

- **Three-step**: blip applied once, to `mu_{j+1}`; plain AIPW terms
  `w·(mu_m − mu_{m-1})` telescope down to `Y`.
- **Robins**: every increment carries its own blip, and weights become
  `w_t^{1−I_t}` so eligible subjects get weight 1.

If these are ever modified, verify by writing a generic loop implementation of
the above, checking it reproduces the untouched K=3 `../three_time_binary/`
estimators exactly, then checking the unrolled K=5 code against it. That
two-way check is what caught nothing and confirmed correctness to ~1e-14.

Note the deepest equation (j=4) has a different shape from the intermediate
ones — the blip lands on `Y` directly rather than on a fitted `mu`. K=5 output
therefore does *not* collapse to K=3 output for the last equation; that is
expected, not a bug.

## Known issues

- **Results are not reproducible run to run.** `RandomForestRegressor(n_jobs=-1)`
  accumulates tree predictions across threads, so summation order varies;
  `mu05` moves ~4% between identical runs on the same machine. Setting the
  forests to `n_jobs=1` makes it exactly deterministic. Left as-is by choice —
  nuisance variation is expected and averages out over 300 iterations.
- **Thread oversubscription.** `run_simulation` parallelises iterations with
  `n_jobs=-1` while each forest also uses `n_jobs=-1` — roughly 12 × 12 threads
  on 12 cores. Setting forests to `n_jobs=1` would likely speed the run up.
- **Low-`ph` cells are unstable by construction.** The three-step weight is a
  product of four `w_t`; at `ph=0.3` (the current lowest production `ph`,
  replacing the earlier `ph=0.2`) only `0.3^4 ≈ 0.81%` of subjects — about 584
  of 72000 — follow a constant-treatment path, with weights up to
  `(1/0.3)^4 ≈ 123`. Expect wild variance and a failure-rate gradient across ψ
  — `psi45` is well-behaved, `psi05` is the worst (bias/variance scale with
  the number of compounded `w_t` factors in each equation: 0 for `psi45`, 4
  for `psi05`). Robins' estimator is partly protected since `w^{1−I} = 1` for
  eligible subjects, so it degrades mainly in the `p_I=0.2` cells. Confirmed
  via oracle-nuisance diagnostics (`n72000/check_ipw_ee_oracle*.py`) that
  `Three-step-ipw`'s bias here is a nuisance-estimation artifact (it lacks
  double robustness in `ps_t`), not a flaw in the estimating equation itself.
- **Positivity when ψ ≠ 0.** `generate_data.py` sums five blip terms onto an
  intercept of `-1.5`; `exp(log_p)` can exceed 1 and `rng.binomial` then raises,
  which `_one_iter` converts to NaN for every iteration in the cell — an
  all-NaN cell with no obvious cause. Fine at ψ=0. Lower the intercept before
  any non-null run.
- `PolynomialFeatures(degree=3)` on `mu45`'s 9 covariates expands to 220
  features, fit on ~`ph·n/2` rows. Near-saturated at low `ph`.
