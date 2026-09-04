# SNTTEM simulation — five time points

Monte Carlo study of structural nested treatment-eligibility models, extended
from three time points to **five**. Ported from `../three_time_binary/`, which
remains untouched as the K=3 reference.

## Where the work is

**`psi0_python/n72000/` and `psi1_python/n72000/` are the directories ported
to five time points.** Everything else in this tree is still K=3 code and
will break if run:

| Path | State |
|---|---|
| `psi0_python/n72000/` | **K=5, current** (n=72000; renamed from `n24000/` when the production sample size moved to 72000). psi05..psi45 = 0.0 (null mechanism). Top level holds `generate_data.py`/`estimators.py` (shared source of truth, also used by the diagnostic `check_ipw_ee_oracle*.py` scripts still here) plus two self-contained pipelines, see below. |
| `psi0_python/n72000/ML_nuisance/` | **K=5, current, post-fix rerun completed 2026-09-03.** The original production pipeline (`working_models_ml.py`, `simulation_output.py`, `run_simulation.py`, `run.sh`, `check_results.py`), moved here 2026-09-01 to sit alongside `oracle/`. Nuisances fit from data (RF/GBM/poly ensemble for Three-step-g/Robins'; a correctly-specified plain logistic/linear working model, `working_model_true()`, for Three-step-ipw only — see "Status" below). Relaunched after the `ee_three_step_ipw` fix (interrupted once, relaunched under `nohup` 2026-09-02, finished 2026-09-03 14:01, 18h20m); its prior completed run (pre-fix) has been superseded. |
| `psi0_python/n72000/oracle/` | **K=5, current, added 2026-09-01.** Same three estimators, but every nuisance (`ps0..ps4`, `mu05..mu45`) is the EXACT closed-form value implied by the DGP — no fitting, no cross-fitting at all. Full 6-cell x 300-iter grid runs in ~40s. See `oracle/README.md`. |
| `psi1_python/n72000/` | **K=5, set up 2026-08-31.** Ported from `psi0_python/n72000/`: same pipeline, psi05..psi45 = 1.0 (non-null). Intercept in `generate_data.py` is lowered to -2.6 (from -1.5) to keep `exp(log_p) < 1` under the compounded psi=1 blip terms — see that file's docstring and "Known issues" below. Restructured into `ML_nuisance/`/`oracle/` 2026-09-01, mirroring psi0; top level holds the shared `generate_data.py`/`estimators.py`. |
| `psi1_python/n72000/ML_nuisance/` | **K=5, current, post-fix rerun completed 2026-09-02.** Mirrors `psi0_python/n72000/ML_nuisance/`. An earlier n=72000/300-iter/6-cell run (finished 2026-09-01 11:59, 11h39m) used the pre-fix, buggy `ee_three_step_ipw` and was superseded: `run.sh` was relaunched 2026-09-01 20:41 with the corrected equation and ran to completion 2026-09-02 13:37 (16h56m) — see `ML_nuisance/README.md`. |
| `psi1_python/n72000/oracle/` | **K=5, current, added 2026-09-01.** Same idea as psi0's oracle/, but psi=1 means `mu05..mu45` are not constant — `oracle_nuisances.py` derives and `check_oracle_derivation.py` verifies the closed form (score ~0 at true theta=(1,1,1,1,1), |z|<2). Full 6-cell x 300-iter grid runs in ~260s; psi05 median lands at 0.95-1.02 (target 1.0). See `oracle/README.md`. |
| `psi1_python/n144000/` | **K=5, current, added 2026-09-03.** Hard-cell-only (`p_I=0.2, ph=0.3` — the worst cell in the production grid, see "Known issues") sample-size check: same `ML_nuisance/`/`oracle/` split as `n72000/`, but n=144000 and only that one cell, 300 iterations, to see whether doubling n stabilizes it. See "Status: n=144000 hard-cell check" below. |
| `psi0_python/n72000/misspecification/` | K=3, not being run |
| `psi0_python/` (parent scripts), `psi0_python/n12000/` | K=3 |
| `psi1_python/` (parent scripts), `psi1_python/n24000/` | K=3 |

The original n=24000 production run script (`ph` in `{0.2, 0.5, 0.8}`) is
preserved inside `n72000/` as `run_simulation_n24000.py`, so the
`run_simulation.py` / `simulation_results.*` names were free for the n=72000
config (now in `ML_nuisance/`, see above). Its output
(`simulation_results_n24k.{pkl,csv,...}`) and the `run_scaletest_96k*.py`
scale-test outputs (`scaletest_96k*`, `psi05_96k*`) were deleted 2026-08-31.

The `run_scaletest_48k_ph03_full.py` / `run_scaletest_72k_ph03_full.py` /
`run_scaletest_96k_ph02_full.py` / `run_scaletest_96k_ph03_full.py` scripts,
their `scaletest_48k_ph03_*` / `scaletest_72k_ph03_*` result files, and the
`all_psi_variance_ph02_vs_ph03.png` / `psi05_ph03_failure_rate.png` plots
were removed 2026-09-01 (this was the closed investigation that decided
`ph=0.3` over `ph=0.2` for production, already reflected in `run_simulation.py`'s
`PH_VALUES`; the scripts are recoverable from git history if that comparison
is ever needed again). The old n24000-era GMM reporting pipeline
(`make_report_pdfs.py`, `distribution_plots.py`, `plot_estimates.py`,
`regen_bar_reports.py`, `regen_qqhist_allvalues.py`,
`combine_variance_figure.py`) and a stale top-level `README.md` (it described
the n=24000/GMM/36-cell config, not this directory) were removed at the same
time — `check_results.py` (in `ML_nuisance/` and `oracle/`) is the current
reporting tool. `run_simulation_n24000.py` itself is still deliberately kept.

## Status: n=72000 production run complete (2026-08-31)

The 6-cell grid (`p_I` ∈ {0.2, 1.0}, `ph` ∈ {0.3, 0.5, 0.8}) finished in
**47,697 s ≈ 13.25 h** wall time in `psi0_python/n72000/ML_nuisance/`
(moved there 2026-09-01 from the `n72000/` top level; paths below are
relative to that subfolder now). Per-cell timings are in
`simulation_results_timings.csv`; results are in `simulation_results.{csv,pkl}`
(5400 rows = 6 cells × 3 methods × 300 iterations).

Post-run summary/plots come from `check_results.py`, which writes
`simulation_results_plots.pdf` (4 pages per ψ: variance, median, MSE,
coverage) and prints a per-cell table (median, robust variance, MSE, 95%
coverage) to stdout. Bar order in the plots is fixed as **Simple g-estimator,
Three-step-g estimator, Three-step-ipw estimator**.

**`ee_three_step_ipw` had an equation bug, fixed 2026-09-01.** In
`estimators.py`, each equation's `mu` term was subtracted *outside* the
telescoping weight product (`w4*w3*w2*w1*Y*exp(blip) - mu05`) instead of
*inside* it (`w4*w3*w2*w1*(Y*exp(blip) - mu05)`) — mathematically wrong,
since the weight has to multiply the whole AIPW-style residual, not just the
`Y` term. Confirmed via oracle-nuisance checks (`check_ipw_ee_oracle.py`,
`check_ipw_ee_oracle_mc.py`): the corrected equation's score is unbiased at
the truth and its psi05 median moved from +0.11/+0.17 (old, buggy) to
+0.011/+0.012 (fixed) at n=72000/ph=0.3, matching Three-step-g almost
exactly. `psi0_python/n72000/oracle/` (see the table above) reproduces this
at the full 6-cell/300-iteration grid for all three estimators, confirming
the fix at production scale, not just one cell.

Separately, `ML_nuisance/working_models_ml.py` now fits Three-step-ipw's
nuisances differently from Three-step-g/Robins': `working_model_true()`
uses the same covariate/history sets as the RF/GBM/poly ensemble but with a
single plain logistic regression (propensity scores, and `mu45` since Y is
binary) or linear regression (`mu05/mu15/mu25/mu35`) instead — a
correctly-specified parametric model rather than a flexible one, to test
whether that closes the residual `ps_t`-estimation bias described below
under "Known issues". Three-step-g and Robins' are unaffected and still use
`working_model()`'s RF/GBM/poly ensemble.

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
shrinks as `ph` rises. All three methods converge to near-identical estimates
for psi35/psi45 (psi45 rows are numerically identical across methods per
cell, since that equation has zero compounded weight factors).

**The paragraph above describes the pre-fix run.** The post-fix rerun
(relaunched 2026-09-01 17:08, interrupted once when the session ended after
~1.4 cells, relaunched again 2026-09-02 19:41 under `nohup`, finished
2026-09-03 14:01, elapsed 18h20m — see `ML_nuisance/README.md`) confirms
most of Three-step-ipw's bias in the pre-fix headline **was** the equation
bug: psi05 bias across the 6 cells is now roughly +0.004 to +0.009 (down
from the old +0.11/+0.17 at the worst cell), tracking Simple g-/Three-step-g
closely, with 93–95% coverage throughout. `psi0_python/n72000/oracle/`
already confirmed this at the equation level (see above); this rerun
confirms it end-to-end with fitted nuisances too. `ph=0.3` cells remain the
noisiest of the six by variance (see "Known issues"), but the systematic
bias that motivated `working_model_true()` is now attributable mostly to
the equation bug, not to `ps_t`-estimation error — see the psi1 status
below and the n=144000 hard-cell check for how much of the remaining
`ph=0.3` variance is a finite-`n` artifact.

## Status: psi1 (non-null) simulation

`psi1_python/n72000/` was restructured to match psi0 on 2026-09-01: same
`ML_nuisance/` (RF ensemble + `working_model_true()` for ipw) /
`oracle/` (exact closed-form nuisances) split, same `ee_three_step_ipw` fix
applied to `estimators.py`.

`psi1_python/n72000/ML_nuisance/`'s original run (11h39m, finished
2026-09-01 11:59, before this session's equation fix) was pre-fix, exactly
like psi0's original run was. It has since been superseded: `run.sh` was
relaunched 2026-09-01 20:41 with the corrected `estimators.py` and ran to
completion 2026-09-02 13:37 (16h56m) — see `ML_nuisance/README.md`. The
post-fix results show psi05 medians landing close to the target 1.0 across
all 6 cells (roughly 0.85–1.17 depending on cell, worst at `ph=0.3` as
expected) with ~89–95% coverage; `ph=0.3` cells still carry the highest
variance/MSE of the six, consistent with the "Known issues" positivity
discussion below.

`psi1_python/n72000/oracle/`'s full 6-cell/300-iteration grid (run
2026-09-01, ~260s) confirms the fix at psi1 too: psi05 median lands at
0.95-1.02 across all cells (target 1.0), ~94-96% coverage. Getting oracle
nuisances right here was less trivial than psi0's: since psi=1 means Y
genuinely depends on treatment history, `mu05..mu45` are not constant (unlike
psi0's null-mechanism `exp(-1.5)` constant) — see
`psi1_python/n72000/oracle/oracle_nuisances.py`'s docstring for the closed-
form derivation and `check_oracle_derivation.py` for its score-at-truth
verification (both confirmed before trusting the production grid run).

## Status: n=144000 hard-cell check (psi1)

`psi1_python/n144000/` (added 2026-09-03) asks whether the production
grid's worst cell — `p_I=0.2, ph=0.3`, where only ~0.81% of subjects follow
a constant-treatment path and three-step weights run up to `(1/0.3)^4 ≈
123` (see "Known issues") — is unstable because of finite `n`, or for a
more fundamental reason that more data won't fix. It reruns just that one
cell, at double the sample size (n=144000, still 300 iterations, same seed
3411), with both the `oracle/` and `ML_nuisance/` pipelines from `n72000/`
copied over unchanged (see each subfolder's README.md).

`oracle/` finished in 15s. Comparing the same cell's psi05 numbers,
n=72000 vs n=144000 (robust variance from `check_results.py`, i.e. the
same methodology as every other table in this document):

| method | n=72000 median (var) | n=144000 median (var) |
|---|---|---|
| Simple g- / Robins' | 1.003 (1.869) | 1.003 (0.728) |
| Three-step-g | 0.954 (1.573) | 0.921 (0.508) |
| Three-step-ipw | 0.956 (1.636) | 0.921 (0.519) |

Coverage stays ~94–95% at both `n`. Variance drops by roughly 2.5–3x from
doubling `n` — more than the ~2x a pure `1/n` scaling would predict, though
with only 300 iterations per cell that ratio itself has sampling noise and
shouldn't be read too precisely. The Three-step estimators' median moving
from 0.954/0.956 to 0.921 (further from the target 1.0, not closer) is
likely within normal Monte Carlo noise at this iteration count rather than
a systematic effect — `check_oracle_derivation.py` already confirmed these
equations are unbiased at the truth in expectation. Net read so far: more
data clearly helps this cell's *variance*, which is consistent with the
instability being substantially a finite-`n` artifact rather than something
intrinsic to the estimating equations at `ph=0.3`.

`ML_nuisance/` was launched 2026-09-03 20:28 under `nohup` (survives the
launching session ending) and was still running as of this writing —
expect several hours longer than the ~2.8h the same cell took at n=72000
inside the 6-cell grid, since RF/GBM training cost doesn't scale linearly
with `n`. Check `psi1_python/n144000/ML_nuisance/sim_run.log` for progress;
once done, `check_results.py` there gives the fitted-nuisance counterpart
to the oracle numbers above, and separates "does more data help the
equations" (oracle, answered above) from "...help nuisance estimation on
top of that" (ML_nuisance, pending).

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
cd psi0_python/n72000/ML_nuisance && ./run.sh   # ML-nuisance production run (psi0, null)
cd psi0_python/n72000/oracle && ./run.sh        # oracle-nuisance run (~40s, all 6 cells)
cd psi1_python/n72000/ML_nuisance && ./run.sh   # ML-nuisance production run (psi1, non-null)
cd psi1_python/n72000/oracle && ./run.sh        # oracle-nuisance run (~260s, all 6 cells)
cd psi1_python/n144000/ML_nuisance && ./run.sh  # hard-cell-only, n=144000 (~several hours, one cell)
cd psi1_python/n144000/oracle && ./run.sh       # hard-cell-only, n=144000 (~15s, one cell)
```

Each `n72000/` pipeline also has a `compare_oracle_vs_ml_psi05.py` script
(one level up, e.g. `psi0_python/n72000/compare_oracle_vs_ml_psi05.py`) that
loads both `oracle/simulation_results.pkl` and
`ML_nuisance/simulation_results.pkl`, and plots grouped bars (Oracle solid,
ML hatched) of psi05's variance/median/MSE/coverage per cell, writing
`psi05_oracle_vs_ml.pdf` next to itself — the quickest way to see how much
of Three-step-ipw's remaining gap from Simple g-/Three-step-g is
nuisance-estimation noise versus something else.

Avoid running both `ML_nuisance/run.sh` jobs (psi0 and psi1) at the same time
on one machine — each already oversubscribes threads roughly 12x12 on its
own (see "Known issues" below); running two concurrently thrashes both far
worse than the ~13h/job serial cost.

`run.sh` resolves the venv relative to its own location, preflights the six
imports, runs under `caffeinate -i`, and tees to `sim_run.log` (rotating any
previous log). At n=72000, the 6-cell grid took **~13.25 h** total in the
original (pre-fix) psi0 run (measured 2026-08-31, see
`simulation_results_timings.csv`): ph=0.3 cells ~99–105 min, ph=0.5 cells
~133–138 min, ph=0.8 cells ~153–167 min. The post-fix reruns of both psi0
and psi1 took noticeably longer end-to-end (~18.3h and ~16.9h respectively)
— see each `ML_nuisance/README.md`; treat wall time as varying run to run
(thread oversubscription, see "Known issues") rather than a fixed number.
The earlier n=24000 config ran in ~2–3 h (~3.3 s/iteration, ~16 min/cell).

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
  eligible subjects, so it degrades mainly in the `p_I=0.2` cells.
  `Three-step-ipw`'s bias at this cell had two separate causes, since
  disentangled: an actual equation bug in `ee_three_step_ipw` (see "Status"
  above, fixed 2026-09-01, confirmed via `n72000/check_ipw_ee_oracle*.py` and
  `oracle/`), and IPW's inherent lack of double robustness in `ps_t` — any
  `ps_t` estimation error translates directly into bias here, unlike the
  AIPW-style Three-step-g/Robins'. `ML_nuisance/working_models_ml.py`'s
  `working_model_true()` (correctly-specified parametric nuisances for
  Three-step-ipw only) tests whether removing `ps_t` estimation noise (vs.
  just flexibility) closes this remaining gap.
- **Positivity when ψ ≠ 0.** `generate_data.py` sums five blip terms onto an
  intercept of `-1.5`; `exp(log_p)` can exceed 1 and `rng.binomial` then raises,
  which `_one_iter` converts to NaN for every iteration in the cell — an
  all-NaN cell with no obvious cause. Fine at ψ=0. Lower the intercept before
  any non-null run. Resolved for `psi1_python/n72000/` (psi=1 mechanism) by
  lowering the intercept there to `-2.6`, checked to give zero `exp(log_p) > 1`
  exceedances across the full production grid (n=72000, 300 iters, all 6
  cells); worst-case theoretical bound at psi=1 is intercept `-2.5` exactly
  (`-1.5 + 2.5`), so `-2.6` is a thin safety margin, not a generous one — a
  different non-null ψ magnitude would need this recomputed, not reused.
- `PolynomialFeatures(degree=3)` on `mu45`'s 9 covariates expands to 220
  features, fit on ~`ph·n/2` rows. Near-saturated at low `ph`.
