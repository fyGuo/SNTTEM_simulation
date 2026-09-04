# Blip (γ) Misspecification Study — psi = 1 setting

This folder reuses the **exact same DGP, estimators, and 8-run design** as the
`psi0_python/.../misspecification` study, but with **true `ψ03 = ψ13 = ψ23 = 1`**
(instead of 0). The g-estimators still fit with a **deliberately wrong blip
functional form**, while the data are generated under the **true** blips. Because
the blips are now active (ψ ≠ 0), misspecification is expected to produce genuine
**bias**, not just a variance effect.

---

## 1. Design

- **DGP (unchanged):** `generate_data.py` generates data under the true blips. The
  true blip forms are
  - `gamma03(L0, ψ) = ψ·L0`
  - `gamma13(L1, ψ) = ψ·(1 + L1)`
  - `gamma23(L2, ψ) = ψ·(L2 + 1)`
- **Fitting (misspecified):** the estimating equations may instead use the wrong forms
  - `gamma13_wrong(L1, ψ) = ψ·(L1 + 10)/(L1 + 4)`
  - `gamma23_wrong(L2, ψ) = ψ·(L2 + 10)/(L2 + 4)`
  - `gamma03` is always kept at its true form.
- **Settings (2):** `ph1 = ph2 = 0.5`, and `p_I1 = p_I2 ∈ {0.2, 1}` (paired).
- **Blip combos (4):** every (correct/wrong γ₁₃) × (correct/wrong γ₂₃).
- **Total runs:** 2 settings × 4 combos = **8**. Each is tagged in the output with
  `gamma13`/`gamma23 ∈ {correct, wrong}`.
- **Truth:** `ψ03 = ψ13 = ψ23 = 1`, `N = 24000`, `N_ITER = 300`, `seed = 3411`.
  (The DGP intercept `-1.5` keeps the outcome probability `exp(log_p) ∈ [0, 1]` even
  at ψ = 1 — verified: max prob = 1.0000, min = 0.082 over the support.)

> **What to expect at ψ = 1 (contrast with the ψ = 0 study).** The blip enters the
> estimating equations as `exp(−γ(L, ψ)·(A − A_prev))`. At the null ψ = 0 that term is
> exactly 1 regardless of the blip form, so the ψ = 0 study showed **no bias** (only a
> variance/identification effect). Here **ψ = 1 makes the blips active**, so fitting the
> wrong functional form solves a *different* estimating equation and is expected to
> yield **genuine bias** in the corresponding parameter:
>
> - misspecifying **γ₁₃** should bias **ψ̂₁₃** (eq. 2), misspecifying **γ₂₃** should bias
>   **ψ̂₂₃** (eq. 3); ψ̂₀₃ can also move indirectly because eq. 1 depends on ψ₁₃/ψ₂₃.
> - the correct/correct combo is the correctly-specified reference (≈ unbiased).
>
> Bias should be read as `mean(estimate) − 1` here (see `plot_bias_scenarios.py`,
> `TRUE_PSI = 1.0`).

---

## 2. Files and exact changes vs. the n24000 originals

Only three files differ from `../` (copies of `generate_data.py`,
`working_models_ml.py`, `requirements.txt`, `run.sh` are byte-identical).

### 2.1 `estimators.py` — define true+wrong blips, thread the choice through

Replace the blip import with true+wrong definitions:

```python
# True blips come from the DGP module (gamma03 always used at its true form).
from generate_data import gamma03, gamma13 as gamma13_true, gamma23 as gamma23_true


def gamma13_wrong(L1, psi13):
    """MISSPECIFIED blip at time 1 (true form: psi13 * (1 + L1))."""
    return psi13 * (L1 + 10) / (L1 + 4)


def gamma23_wrong(L2, psi23):
    """MISSPECIFIED blip at time 2 (true form: psi23 * (L2 + 1))."""
    return psi23 * (L2 + 10) / (L2 + 4)
```

Give the estimating equations two extra args `g13`, `g23` (default = true) and use
them wherever the time-1 / time-2 blip appears (`gamma03` stays as-is):

```python
def ee_three_step(theta, df, g13=gamma13_true, g23=gamma23_true):
    ...
    # eq2 uses g13:  mu23 * np.exp(-g13(L1, psi13) * (A1 - A0))
    # eq3 uses g23:  Y   * np.exp(-g23(L2, psi23) * (A2 - A1))
    ...

def ee_robins(theta, df, g13=gamma13_true, g23=gamma23_true):
    ...
    # eq1 uses g23 and g13; eq2 uses g23 and g13; eq3 uses g23
    ...
```

Forward the choice through the solver:

```python
def m_estimate(ee_fn, df, start=(0.0, 0.0, 0.0),
               g13=gamma13_true, g23=gamma23_true):
    sol = fsolve(ee_fn, x0=np.array(start, dtype=float), args=(df, g13, g23),
                 full_output=True)
    return sol[0]
```

### 2.2 `simulation_output.py` — pass the blip choice down, tag results

Import the blip functions and give `_bootstrap_gmm` / `_one_iter` the `g13`, `g23`
arguments:

```python
from estimators import (
    ee_three_step, ee_robins, m_estimate,
    gamma13_true, gamma13_wrong, gamma23_true, gamma23_wrong,
)

def _bootstrap_gmm(df, actual_old, actual_rbs, n_boot, rng, g13, g23):
    ...
    est_old_b = m_estimate(ee_three_step, df_b, g13=g13, g23=g23)
    est_rbs_b = m_estimate(ee_robins,     df_b, g13=g13, g23=g23)
    ...

def _one_iter(..., g13, g23):
    ...
    est_old = m_estimate(ee_three_step, df, start=(0.,0.,0.), g13=g13, g23=g23)
    est_rbs = m_estimate(ee_robins,     df, start=(0.,0.,0.), g13=g13, g23=g23)
    gmm_est = _bootstrap_gmm(df, est_old, est_rbs, n_boot=n_boot, rng=rng,
                             g13=g13, g23=g23)
```

Add the two switches to `run_simulation`, choose the blips, forward them, tag output:

```python
def run_simulation(..., gamma13_misspec=False, gamma23_misspec=False):
    g13 = gamma13_wrong if gamma13_misspec else gamma13_true
    g23 = gamma23_wrong if gamma23_misspec else gamma23_true
    common = dict(..., g13=g13, g23=g23)         # passed into _one_iter
    ...
    est_final["gamma13"] = "wrong" if gamma13_misspec else "correct"
    est_final["gamma23"] = "wrong" if gamma23_misspec else "correct"
    return est_final
```

### 2.3 `run_simulation.py` — restrict grid, sweep the 4 blip combos

```python
# 2 settings: ph1 = ph2 = 0.5, p_I1 = p_I2 in {0.2, 1} (paired).
PI_VALUES = [0.2, 1]
PH1 = 0.5
PH2 = 0.5

# (gamma13_misspec, gamma23_misspec)
MISSPEC_COMBOS = [(False, False), (False, True), (True, False), (True, True)]

settings = [(pI, pI, PH1, PH2) for pI in PI_VALUES]
combos = [(p_I1, p_I2, ph1, ph2, g13m, g23m)
          for (p_I1, p_I2, ph1, ph2) in settings
          for (g13m, g23m) in MISSPEC_COMBOS]          # -> 8 runs

for k, (p_I1, p_I2, ph1, ph2, g13m, g23m) in enumerate(combos, start=1):
    est = run_simulation(n=N, n_iter=N_ITER, psi03=PSI03, psi13=PSI13, psi23=PSI23,
                         p_I1=p_I1, p_I2=p_I2, ph1=ph1, ph2=ph2, seed=SEED, n_jobs=-1,
                         gamma13_misspec=g13m, gamma23_misspec=g23m)
```

---

## 3. Output

`run_simulation.py` writes (next to itself):

- `simulation_results.pkl` / `simulation_results.csv` — one tidy row per
  (iteration × method × setting × blip-combo). Columns: `id, est_psi03, est_psi13,
  est_psi23, method, p_I1, p_I2, ph1, ph2, seed, psi03, psi13, psi23,
  gamma13, gamma23`. `method ∈ {Three-step, Robins' estimator, GMM estimator}`.
- `simulation_results_timings.csv` — seconds per run, tagged with `gamma13`/`gamma23`.

To analyze one blip-combo, filter e.g. `df[(df.gamma13=="wrong") & (df.gamma23=="correct")]`.

---

## 4. How to run

Use the `snttem` conda env (see below):

```bash
cd n24000/misspecification
PYTHON=~/miniforge3/envs/snttem/bin/python ./run.sh      # honors the PYTHON override
# or directly:
~/miniforge3/envs/snttem/bin/python run_simulation.py
```

Full run = 8 configurations × 300 iters × N = 24000, each with a 200-resample
bootstrap and cross-fitted ML nuisance models — plan for a long wall-clock time
(same order as the original n24000 run, ×(8/36) of its grid). `run.sh` wraps it in
`caffeinate` to keep the Mac awake.

---

## 5. Reusing in other settings

- **Other sample size:** copy this folder next to the target `nXXXX/`, set `N` in
  `run_simulation.py`. (Or point `DATASETS` of any analysis script at the new pickle.)
- **Non-zero ψ (to see bias):** set `PSI03/PSI13/PSI23` in `run_simulation.py`. Check
  first that the DGP outcome probability `np.exp(log_p)` stays ≤ 1 over the support of
  `L0∈[0,0.5]`, `L1,L2∈[-1.5,-0.5]`; adjust the `-1.5` intercept in
  `generate_data.py` if needed.
- **Different wrong blip:** edit `gamma13_wrong` / `gamma23_wrong` in `estimators.py`.
  Keep the denominator away from 0 over `L∈[-1.5,-0.5]` (here `L+4∈[2.5,3.5]`, safe).
- **Also misspecify γ₀₃:** add a `gamma03_wrong`, a `g03` arg to the ee/`m_estimate`
  functions, and a `gamma03_misspec` switch mirroring §2.
- **More settings:** extend `PI_VALUES`, `PH1/PH2`, or `MISSPEC_COMBOS`.

---

## 6. Environment (`snttem`)

Created with miniforge conda, Python 3.11:

```bash
~/miniforge3/bin/conda create -n snttem -y -c conda-forge \
  python=3.11 numpy pandas scipy scikit-learn joblib matplotlib
```

Interpreter: `~/miniforge3/envs/snttem/bin/python`. Verified deps: numpy, pandas,
scipy, scikit-learn, joblib, matplotlib (all `requirements.txt` constraints satisfied).
Activate with `conda activate snttem` (once `conda init` is set up for your shell), or
just call the interpreter by full path as shown above.

> Note: the old `snttem311` Jupyter kernel points to a deleted `anaconda3` env and is
> stale. To register this new env as a Jupyter kernel:
> `~/miniforge3/envs/snttem/bin/python -m ipykernel install --user --name snttem`
> (install `ipykernel` into the env first).
