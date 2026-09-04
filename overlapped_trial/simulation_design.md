# Simulation Design: Overlapped vs. Non-overlapped Population Comparison

We work with three decision points. The treatments are $A_0, A_1, A_2$, the
time-varying covariates are $L_0, L_1, L_2$, and the (normally distributed)
outcome is $Y$. Potential outcomes are written $Y^{a_0,a_1,a_2}$. The document is
organized as **data-generating process** (§1), **estimand** (§2), and
**estimators** (§3), followed by the simulation plan (§4).

---

## 1. Data-generating mechanism

Variables are generated in temporal order
$$L_0 \to A_0 \to L_1 \to A_1 \to L_2 \to A_2 \to Y.$$
Let $\operatorname{expit}(x)=1/(1+e^{-x})$.

### Baseline
$$
L_0 \sim \mathcal{N}(0,1),\qquad
A_0 \mid L_0 \sim \text{Bernoulli}\bigl(\pi_0(L_0)\bigr),\quad
\pi_0(L_0)=\operatorname{expit}(\alpha_{00}+\alpha_{01}L_0).
$$

### Time 1
$$
L_1 \mid L_0,A_0 \sim \mathcal{N}\bigl(\lambda_{10}+\lambda_{11}L_0+\lambda_{12}A_0,\ \sigma_{L_1}^2\bigr),
$$
$$
A_1 \mid \bar S_1 \sim \text{Bernoulli}\bigl(\pi_1(\bar S_1)\bigr),\quad
\pi_1(\bar S_1)=\operatorname{expit}(\alpha_{10}+\alpha_{11}L_0+\alpha_{12}A_0+\alpha_{13}L_1).
$$

### Time 2
$$
L_2 \mid \bar S_1,A_1 \sim \mathcal{N}\bigl(\lambda_{20}+\lambda_{21}L_1+\lambda_{22}A_1+\lambda_{23}A_0,\ \sigma_{L_2}^2\bigr),
$$
$$
A_2 \mid \bar S_2 \sim \text{Bernoulli}\bigl(\pi_2(\bar S_2)\bigr),\quad
\pi_2(\bar S_2)=\operatorname{expit}(\alpha_{20}+\alpha_{21}L_1+\alpha_{22}A_1+\alpha_{23}L_2).
$$

Note $L_2$ is a **time-2 confounder affected by prior treatment $A_1$**
(treatment–confounder feedback), the feature that makes the problem
non-trivial.

### Outcome (structural nested model form)

We generate $Y$ from **primitive** structural blips $\delta_1,\delta_2$ — the
additive effects we control directly — and then *derive* the paper's
$\gamma_1,\gamma_2$ from them (§2):

$$
Y = \underbrace{\beta_0+\beta_1 L_0+\beta_2 A_0+\beta_3 L_1}_{\text{treatment-free mean } h(L_0,A_0,L_1)}
\;+\; A_1\,\delta_1(\bar S_1) \;+\; A_2\,\delta_2(\bar S_2) \;+\;\varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,\sigma_Y^2),
$$

with linear primitive blips
$$
\delta_1(\bar S_1)=\psi_{10}+\psi_{11}L_1,\qquad
\delta_2(\bar S_2)=\psi_{20}\quad(\text{i.e. }\psi_{21}=0,\ \text{constant } A_2 \text{ effect}).
$$

**Key modeling choice:** the treatment-free mean $h$ does **not** depend on
$L_2$. This keeps the time-1 blip free of contamination from the
$A_1\!\to\!L_2$ feedback path, while $L_2$ still acts as a genuine time-2
confounder (it drives $\pi_2$ and scales the $A_2$ effect $\delta_2$).

**History summaries.** Throughout we take
$$
\bar S_1 = (L_0, A_0, L_1), \qquad \bar S_2 = (L_0, A_0, L_1, A_1, L_2).
$$

### Default parameter values

| Group | Parameters | Values |
|---|---|---|
| $\pi_0$ | $\alpha_{00},\alpha_{01}$ | $0.0,\ 0.5$ |
| $L_1$ | $\lambda_{10},\lambda_{11},\lambda_{12},\sigma_{L_1}$ | $0.0,\ 0.6,\ 0.5,\ 1.0$ |
| $\pi_1$ | $\alpha_{10},\alpha_{11},\alpha_{12},\alpha_{13}$ | $0.0,\ 0.3,\ 0.4,\ 0.5$ |
| $L_2$ | $\lambda_{20},\lambda_{21},\lambda_{22},\lambda_{23},\sigma_{L_2}$ | $0.0,\ 0.6,\ 0.5,\ 0.3,\ 1.0$ |
| $\pi_2$ | $\alpha_{20},\alpha_{21},\alpha_{22},\alpha_{23}$ | $0.0,\ 0.3,\ 0.4,\ 0.5$ |
| $h$ | $\beta_0,\beta_1,\beta_2,\beta_3$ | $1.0,\ 0.5,\ 1.0,\ 0.8$ |
| $\delta_1$ | $\psi_{10},\psi_{11}$ | $0.5,\ 0.7$ |
| $\delta_2$ | $\psi_{20},\psi_{21}$ | $0.4,\ 0$ |
| noise | $\sigma_Y$ | $1.0$ |

---

## 2. Estimand

### Target contrast

The estimand is the baseline-arm contrast holding $A_1=1$ and $A_2=0$,
$$
\boxed{\;\mathbb{E}\!\left(Y^{0,1,0} - Y^{1,1,0}\mid L_0\right),\;}
$$
i.e. the effect of switching $A_0$ from $1$ (overlapped) to $0$ (non-overlapped)
for subjects who go on to $A_1=1,\ A_2=0$. Under this linear DGM the truth is
constant in $L_0$ (see §2.3), so we also report the marginal
$\mathbb{E}(Y^{0,1,0}-Y^{1,1,0})$.

### 2.1 Structural nested mean (blip) models

The estimators below use two **structural nested mean (blip) models**.

1. **Time-2 blip** — effect of the last treatment given history $\bar S_2$:
$$
\mathbb{E}\!\left(Y^{a_0,a_1,1} - Y^{a_0,a_1,0}\mid \bar S_2\right)=\gamma_2(\bar S_2).
$$

2. **Time-1 blip** — effect of $A_1$ holding $A_2$ at the **dynamic reference
   regime**
   $$
   h_2^1(\bar S_2) = (1-A_0)\,A_1,
   $$
   given history $\bar S_1$:
$$
\mathbb{E}\!\left(Y^{a_0,1,h_2^1} - Y^{a_0,0,h_2^1}\mid \bar S_1\right)=\gamma_1(\bar S_1).
$$
   The rule sets $A_2 = 1$ only for subjects untreated at baseline who take
   $A_1=1$ (the *non-overlapped, sustained-treatment* path), and $A_2 = 0$
   otherwise. Because $h_2^1$ depends on $A_1$, the contrast actually compares
   $$
   \gamma_1(\bar S_1)=\mathbb{E}\!\left(Y^{a_0,1,\,1-a_0} - Y^{a_0,0,\,0}\mid \bar S_1\right):
   $$
   the $a_1=1$ arm continues to $A_2=1-a_0$, the $a_1=0$ arm gets $A_2=0$. The
   downstream time-2 effect therefore enters $\gamma_1$ **only when $A_0=0$**.

### 2.2 Deriving the paper's blips

$Y^{a_0,a_1,a_2}=h(L_0,a_0,L_1^{a_0})+a_1\,\delta_1(\bar S_1^{a_0})+a_2\,\delta_2(\bar S_2^{a_0,a_1})+\varepsilon$,
where $\bar S_1^{a_0}=(L_0,a_0,L_1^{a_0})$ and
$\bar S_2^{a_0,a_1}=(L_0,a_0,L_1^{a_0},a_1,L_2^{a_0,a_1})$.

**Time-2 blip.** $A_2$ is terminal, so $\bar S_2$ is fixed before it and
$$
\gamma_2(\bar S_2)=Y^{a_0,a_1,1}-Y^{a_0,a_1,0}=\delta_2(\bar S_2)=\psi_{20}.\quad\checkmark
$$

**Time-1 blip with the dynamic reference $h_2^1=(1-A_0)A_1$.** The $a_1=1$ arm
continues to $A_2=1-a_0$:
$$
\gamma_1(\bar S_1)=\mathbb{E}\!\left(Y^{a_0,1,\,1-a_0}-Y^{a_0,0,\,0}\mid\bar S_1\right)
=\delta_1(\bar S_1)+(1-a_0)\,\mathbb{E}\!\left[\delta_2(\bar S_2^{a_0,1})\mid \bar S_1\right].
$$
Since $\delta_2\equiv\psi_{20}$ is constant, the conditional expectation is just
$\psi_{20}$ and the $A_0\times L_1$ interaction vanishes:

$$
\boxed{\;\gamma_1(\bar S_1)=\underbrace{\psi_{10}+\psi_{11}L_1}_{\delta_1}
+(1-A_0)\,\psi_{20}.\;}
$$

So $\gamma_1$ is linear in $L_1$ **with only an $A_0$ main effect** (a level
shift $\psi_{20}$ for the non-overlapped path). A correctly specified working
model is
$$
\gamma_1(\bar S_1;\theta)=\theta_0+\theta_1 L_1+\theta_2 A_0 .
$$
In the estimator the time-1 blip is used only on the overlapped population
($A_0=1$, via the factor $A_0/\pi_0$), where it collapses to
$\gamma_1=\delta_1=\psi_{10}+\psi_{11}L_1$.

### 2.3 Closed-form truth for the target contrast

The target fixes $A_2=0$ in **both** arms, so $\delta_2$ drops out. Let
$L_1^{a_0}=\lambda_{10}+\lambda_{11}L_0+\lambda_{12}a_0+\eta_1$ (same noise draw
$\eta_1$ across regimes):

$$
Y^{0,1,0}-Y^{1,1,0}
= \bigl[h(L_0,0,L_1^{0})-h(L_0,1,L_1^{1})\bigr]+\bigl[\delta_1(\bar S_1^{0})-\delta_1(\bar S_1^{1})\bigr].
$$

With the linear specification this simplifies to a **constant**:

$$
\boxed{\;\mathbb{E}\!\left(Y^{0,1,0}-Y^{1,1,0}\mid L_0\right)
= -\bigl(\beta_2+\beta_3\lambda_{12}+\psi_{11}\lambda_{12}\bigr).\;}
$$

(It is independent of $L_0$ under this linear DGM; add an $A_0\times L_0$ term in
$h$ if an $L_0$-varying truth is desired.) With the §1 defaults,
$$
-\bigl(\beta_2+\beta_3\lambda_{12}+\psi_{11}\lambda_{12}\bigr)
= -\bigl(1.0+0.8\cdot0.5+0.7\cdot0.5\bigr) = -1.75 .
$$

The implied **true paper-blips** (for the oracle / to check estimation):
$$
\gamma_2(\bar S_2)=0.4,\qquad
\gamma_1(\bar S_1)=
\begin{cases}
0.5+0.7\,L_1 & A_0=1\ \text{(overlapped)}\\[2pt]
0.9+0.7\,L_1 & A_0=0\ \text{(non-overlapped)}
\end{cases}
$$
i.e. $\theta=(\theta_0,\theta_1,\theta_2)=(0.9,\,0.7,\,-0.4)$
in $\gamma_1=\theta_0+\theta_1 L_1+\theta_2 A_0$, and $\psi_2=(0.4,\,0)^\top$ in
$\gamma_2=\psi_{20}+0\cdot L_2$.

---

## 3. Estimators

### 3.1 Identification

Under sequential randomization, the target contrast admits the identifying
expression

$$
\begin{aligned}
\mathbb{E}\!\left(Y^{0,1,0} - Y^{1,1,0}\mid L_0\right)
=\;& \underbrace{\mathbb{E}\!\left[\frac{1-A_0}{1-\pi_0(L_0)}\,
   \frac{A_1}{\pi_1(\bar S_1)}\bigl\{Y - A_2\gamma_2(\bar S_2)\bigr\}\;\middle|\;L_0\right]}_{\text{non-overlapped population } (A_0=0)} \\[4pt]
 &-\underbrace{\mathbb{E}\!\left[\frac{A_0}{\pi_0(L_0)}\bigl\{Y - A_2\gamma_2(\bar S_2) - (A_1-1)\gamma_1(\bar S_1)\bigr\}\;\middle|\;L_0\right]}_{\text{overlapped population } (A_0=1)}.
\end{aligned}
$$

Here the propensity scores are
$$
\pi_0(L_0)=\Pr(A_0=1\mid L_0),\qquad
\pi_1(\bar S_1)=\Pr(A_1=1\mid \bar S_1),\qquad
\pi_2(\bar S_2)=\Pr(A_2=1\mid \bar S_2),
$$
which coincide with the exact DGM logistic functions of §1.

### 3.2 Proposed estimator

Replace $\mathbb{E}$ by the empirical average $\mathbb{P}_n[f]=\tfrac1n\sum_{i=1}^n f(O_i)$
over the $n$ i.i.d. observations $O_i=(L_{0i},A_{0i},L_{1i},A_{1i},L_{2i},A_{2i},Y_i)$,
and plug in fitted nuisances $\hat\pi_0,\hat\pi_1,\hat\gamma_1,\hat\gamma_2$. The
estimator of the marginal contrast $\mathbb{E}(Y^{0,1,0}-Y^{1,1,0})$ is

$$
\begin{aligned}
\widehat{\tau}
=\;& \underbrace{\mathbb{P}_n\!\left[\frac{1-A_0}{1-\hat\pi_0(L_0)}\,
   \frac{A_1}{\hat\pi_1(\bar S_1)}\bigl\{Y - A_2\hat\gamma_2(\bar S_2)\bigr\}\right]}_{\text{non-overlapped } (A_0=0)} \\[4pt]
 &-\underbrace{\mathbb{P}_n\!\left[\frac{A_0}{\hat\pi_0(L_0)}\bigl\{Y - A_2\hat\gamma_2(\bar S_2) - (A_1-1)\hat\gamma_1(\bar S_1)\bigr\}\right]}_{\text{overlapped } (A_0=1)} .
\end{aligned}
$$

Equivalently, writing the per-subject contributions
$$
\phi_i^{\text{non}}=\frac{1-A_{0i}}{1-\hat\pi_0(L_{0i})}\,\frac{A_{1i}}{\hat\pi_1(\bar S_{1i})}\bigl\{Y_i - A_{2i}\hat\gamma_2(\bar S_{2i})\bigr\},
\qquad
\phi_i^{\text{ovl}}=\frac{A_{0i}}{\hat\pi_0(L_{0i})}\bigl\{Y_i - A_{2i}\hat\gamma_2(\bar S_{2i}) - (A_{1i}-1)\hat\gamma_1(\bar S_{1i})\bigr\},
$$
the estimator is $\widehat{\tau}=\frac1n\sum_{i=1}^n(\phi_i^{\text{non}}-\phi_i^{\text{ovl}})$,
with influence-function-based standard error
$\widehat{\mathrm{se}}=\sqrt{\tfrac1{n^2}\sum_i\bigl(\phi_i^{\text{non}}-\phi_i^{\text{ovl}}-\widehat{\tau}\bigr)^2}$
(treating nuisances as fixed; a sandwich/bootstrap that accounts for nuisance
estimation is more honest).

**The two terms are different kinds of estimator.**

- The **non-overlapped term $\phi^{\text{non}}$ is a pure IPW estimator**: it
  inverse-weights by *both* propensities, $\tfrac{1-A_0}{1-\pi_0}$ and
  $\tfrac{A_1}{\pi_1}$, to reweight the $A_0=0,\,A_1=1$ subpopulation up to the
  $L_0$ margin. The only blip term is the terminal-treatment removal
  $-A_2\gamma_2$. It therefore relies on **correct $\pi_0,\pi_1$ (and
  $\gamma_2$)** and uses no model for $\gamma_1$.

- The **overlapped term $\phi^{\text{ovl}}$ is a hybrid** (IPW $\times$
  g-estimation): it inverse-weights by $\pi_0$ **only**, and handles $A_1$ and
  $A_2$ by *blip adjustment* $\{Y - A_2\gamma_2 - (A_1-1)\gamma_1\}$ rather than
  by weighting by $\pi_1$. It relies on **correct $\pi_0,\gamma_1,\gamma_2$**
  and uses no model for $\pi_1$.

Because the true contrast is constant in $L_0$ under this DGM, $\widehat{\tau}$
targets it directly. To estimate the **conditional** map
$L_0\mapsto\mathbb{E}(Y^{0,1,0}-Y^{1,1,0}\mid L_0)$, regress
$(\phi_i^{\text{non}}-\phi_i^{\text{ovl}})$ on $L_{0i}$.

### 3.3 Partial IPW estimator

A natural **middle ground** between the proposed estimator and the fully-IPW
benchmark (§3.4) replaces only the **$\pi_2$-weighting** by the $\gamma_2$ blip,
while still handling $A_1$ by inverse-probability weighting in **both**
populations. Fixing $A_1=1$ via $A_1/\pi_1$, blipping $A_2$ down to its reference
through the $H_2$-transform $Y-A_2\gamma_2$, and contrasting the two baseline arms
by $\bigl(\tfrac{1-A_0}{1-\pi_0}-\tfrac{A_0}{\pi_0}\bigr)$:

$$
\boxed{\;
\widehat{\tau}^{\text{pIPW}}
=\mathbb{P}_n\!\left[\frac{A_1}{\hat\pi_1(\bar S_1)}\bigl\{Y-A_2\hat\gamma_2(\bar S_2)\bigr\}
   \left(\frac{1-A_0}{1-\hat\pi_0(L_0)}-\frac{A_0}{\hat\pi_0(L_0)}\right)\right].\;}
$$

Equivalently, with per-subject contributions
$$
\rho_i^{\text{non}}=\frac{1-A_{0i}}{1-\hat\pi_0(L_{0i})}\frac{A_{1i}}{\hat\pi_1(\bar S_{1i})}\bigl\{Y_i-A_{2i}\hat\gamma_2(\bar S_{2i})\bigr\},
\qquad
\rho_i^{\text{ovl}}=\frac{A_{0i}}{\hat\pi_0(L_{0i})}\frac{A_{1i}}{\hat\pi_1(\bar S_{1i})}\bigl\{Y_i-A_{2i}\hat\gamma_2(\bar S_{2i})\bigr\},
$$
we have $\widehat{\tau}^{\text{pIPW}}=\frac1n\sum_i(\rho_i^{\text{non}}-\rho_i^{\text{ovl}})$.

- **It is consistent** under sequential randomization + positivity whenever
  $\pi_0,\pi_1,\gamma_2$ are correct: the non-overlapped piece gives
  $\mathbb{E}[Y^{0,1,0}\mid L_0]$ and
  $\mathbb{E}\bigl[\tfrac{A_0}{\pi_0}\tfrac{A_1}{\pi_1}\{Y-A_2\gamma_2\}\mid L_0\bigr]=\mathbb{E}[Y^{1,1,0}\mid L_0]$.
- **Relation to the proposed estimator.** $\rho^{\text{non}}=\phi^{\text{non}}$
  is *identical* to the proposed non-overlapped term; only the overlapped arm
  changes — here it uses $A_1/\pi_1$ (IPW) instead of the $\gamma_1$ blip. So the
  partial IPW uses **no model for $\gamma_1$ and no model for $\pi_2$**.
- The same IF-based standard error applies, with $\rho$ in place of $\phi$.

### 3.4 Benchmark: fully IPW estimator

For comparison, the **same** target can be estimated by pure inverse-probability
weighting of the two static regimes $(0,1,0)$ and $(1,1,0)$ — no blip models,
only the three propensities $\pi_0,\pi_1,\pi_2$. Each regime fixes $A_1=1$
(weight $A_1/\pi_1$) and $A_2=0$ (weight $(1-A_2)/(1-\pi_2)$), differing only in
the baseline arm:

$$
\boxed{\;
\widehat{\tau}^{\text{IPW}}
=\mathbb{P}_n\!\left[\frac{A_1}{\hat\pi_1(\bar S_1)}\,
   \frac{1-A_2}{1-\hat\pi_2(\bar S_2)}\,
   \left(\frac{1-A_0}{1-\hat\pi_0(L_0)}-\frac{A_0}{\hat\pi_0(L_0)}\right) Y\right].\;}
$$

Equivalently, with per-subject contributions
$$
\psi_i^{\text{non}}=\frac{1-A_{0i}}{1-\hat\pi_0(L_{0i})}\frac{A_{1i}}{\hat\pi_1(\bar S_{1i})}\frac{1-A_{2i}}{1-\hat\pi_2(\bar S_{2i})}Y_i,
\qquad
\psi_i^{\text{ovl}}=\frac{A_{0i}}{\hat\pi_0(L_{0i})}\frac{A_{1i}}{\hat\pi_1(\bar S_{1i})}\frac{1-A_{2i}}{1-\hat\pi_2(\bar S_{2i})}Y_i,
$$
we have $\widehat{\tau}^{\text{IPW}}=\frac1n\sum_i(\psi_i^{\text{non}}-\psi_i^{\text{ovl}})$,
consistent under sequential randomization + positivity whenever
$\pi_0,\pi_1,\pi_2$ are correct.

### 3.5 Comparison

The three estimators form a **progression**: each step replaces one
propensity-weighting by a blip adjustment, trading variance for reliance on a
parametric blip.

| estimator | $\pi_0$ | $\pi_1$ | $\pi_2$ | $\gamma_1$ | $\gamma_2$ |
|---|:--:|:--:|:--:|:--:|:--:|
| fully IPW $\widehat{\tau}^{\text{IPW}}$ | ✓ | ✓ | ✓ | — | — |
| partial IPW $\widehat{\tau}^{\text{pIPW}}$ | ✓ | ✓ | — | — | ✓ |
| proposed $\widehat{\tau}$ | ✓ | ✓ (non-ovl only) | — | ✓ (ovl only) | ✓ |

Moving **down** the table: the fully IPW carries the extra $1/(1-\pi_2)$ weight
and should be **most variable** but avoids all blip models; the partial IPW drops
that weight in favor of $\gamma_2$; the proposed estimator additionally drops the
$\pi_1$-weighting on the overlapped arm in favor of $\gamma_1$. Expect variance
to **decrease** down the table when the blips are correctly specified, at the
cost of relying on those parametric models.

### 3.6 G-estimation of the blip parameters $\gamma_2,\gamma_1$

We use the **true propensity scores** $\pi_0,\pi_1,\pi_2$ (the exact DGM logistic
functions) as the working propensity models, and estimate the blip parameters by
g-estimation. Take blip models linear in their parameters:
$$
\gamma_2(\bar S_2;\psi_2)=m_2(\bar S_2)^\top\psi_2,\quad m_2=(1,\,L_2)^\top,\qquad
\gamma_1(\bar S_1;\psi_1)=m_1(\bar S_1)^\top\psi_1,\quad m_1=(1,\,L_1,\,A_0)^\top .
$$
(True values: $\psi_2=(0.4,\,0)^\top$ and $\psi_1=(0.9,\,0.7,\,-0.4)^\top$,
matching §2.)

**Recursive blip-down (H-transforms).** At the true parameters each $H_t$ equals
the reference potential outcome and is conditionally mean-independent of $A_t$
given the past:
$$
H_2(\psi_2)=Y-A_2\,\gamma_2(\bar S_2;\psi_2)\;\;\xrightarrow{\text{truth}}\;\;Y^{\bar A_1,\,A_2=0},
$$
$$
H_1(\psi_1;\psi_2)=Y-\underbrace{\{A_2-(1-A_0)A_1\}}_{A_2\,\text{blipped to ref }h_2^1}\gamma_2(\bar S_2;\psi_2)-A_1\,\gamma_1(\bar S_1;\psi_1)
\;\;\xrightarrow{\text{truth}}\;\;Y^{A_0,0,0}.
$$
Note the **dynamic reference** $h_2^1=(1-A_0)A_1$ enters here: $A_2$ is blipped
to $0$ for the overlapped arm ($A_0=1$) but to $A_1$ for the non-overlapped arm
($A_0=0$), i.e. the term is $\{A_2-(1-A_0)A_1\}\gamma_2$ rather than $A_2\gamma_2$.

**Centering the H-transforms.** Plain g-estimation centers only the treatment,
$\{A_t-\pi_t\}H_t$. The variance-optimal (and doubly robust) version also
subtracts the **conditional reference mean** $q_t(\bar S_t)=\mathbb{E}[H_t^\ast\mid\bar S_t]$
— the treatment-free outcome regression — from $H_t$, giving the centered
transforms $\tilde H_t=H_t-q_t(\bar S_t)$. Because $q_t$ is a function of the past
only and $\mathbb{E}[A_t-\pi_t\mid\bar S_t]=0$, subtracting it leaves the
estimating equation unbiased while removing the treatment-free signal that
dominates $\operatorname{Var}(H_t)$. At time 2 the reference mean is the $A_2=0$
outcome regression
$$
q_2(\bar S_2)=\mathbb{E}[Y\mid\bar S_2,\,A_2=0],
$$
and at time 1 it is the iterated reference mean along the $A_1=0$ branch (where
$h_2^1=(1-A_0)A_1=0$, so the inner $A_2$ reference is also $0$):
$$
q_1(\bar S_1)=\mathbb{E}\!\Big[\,\underbrace{\mathbb{E}[Y\mid\bar S_2,\,A_2=h_2^1]}_{=\,q_2(\bar S_2)\ \text{at}\ A_1=0}\;\Big|\;\bar S_1,\,A_1=0\Big].
$$

**Correctly specified working models for $q_2,q_1$.** Under the DGM of §1 the
treatment-free mean is $\beta_0+\beta_1L_0+\beta_2A_0+\beta_3L_1$ (no $L_2$),
$\delta_1=\psi_{10}+\psi_{11}L_1$, and $\delta_2\equiv\psi_{20}$. Substituting
$A_2=0$ leaves only the treatment-free mean plus $A_1\delta_1$, so
$$
q_2(\bar S_2)=\beta_0+\beta_1L_0+\beta_2A_0+\beta_3L_1+A_1\,(\psi_{10}+\psi_{11}L_1),
$$
which is **linear in $(1,L_0,A_0,L_1,A_1,A_1L_1)$ and free of $L_2$**. A
correctly specified working model is
$$
q_2(\bar S_2;\xi_2)=\xi_{20}+\xi_{21}L_0+\xi_{22}A_0+\xi_{23}L_1+\xi_{24}A_1+\xi_{25}\,A_1L_1,
$$
fit by OLS of $Y$ on this design **over the $A_2=0$ stratum** (true
$\xi_2=(\beta_0,\beta_1,\beta_2,\beta_3,\psi_{10},\psi_{11})=(1,\,0.5,\,1,\,0.8,\,0.5,\,0.7)$).
On the $A_1=0$ branch the $A_1$ terms drop out and $q_2$ carries no $L_2$, so the
iterated mean collapses to the treatment-free mean,
$$
q_1(\bar S_1)=\beta_0+\beta_1L_0+\beta_2A_0+\beta_3L_1,
$$
**linear in $\bar S_1=(1,L_0,A_0,L_1)$**. A correctly specified working model is
$$
q_1(\bar S_1;\xi_1)=\xi_{10}+\xi_{11}L_0+\xi_{12}A_0+\xi_{13}L_1,
$$
fit by OLS of $\hat q_2(\bar S_2)$ on this design **over the $A_1=0$ stratum**
(true $\xi_1=(\beta_0,\beta_1,\beta_2,\beta_3)=(1,\,0.5,\,1,\,0.8)$).

**Estimating equations** (with true $\pi_1,\pi_2$, solved sequentially):
$$
U_2(\psi_2)=\mathbb{P}_n\!\Big[m_2(\bar S_2)\,\{A_2-\pi_2(\bar S_2)\}\,\big(Y-A_2\gamma_2(\bar S_2;\psi_2)-q_2(\bar S_2)\big)\Big]=0,
$$
$$
U_1(\psi_1)=\mathbb{P}_n\!\Big[m_1(\bar S_1)\,\{A_1-\pi_1(\bar S_1)\}\,\big(H_1(\psi_1;\hat\psi_2)-q_1(\bar S_1)\big)\Big]=0.
$$
Two layers of centering act here. The treatment centering $\{A_t-\pi_t\}$ makes
each equation unbiased at the truth (under sequential randomization
$H_t^\ast\perp A_t\mid\text{past}$, so $\mathbb{E}[\{A_t-\pi_t\}H_t^\ast\mid\text{past}]=0$),
while $-q_t$ residualizes the treatment-free signal to minimize variance.
Subtracting $q_t$ does **not** move the probability limit —
$\mathbb{E}[m_t\{A_t-\pi_t\}q_t]=0$ for *any* $q_t(\bar S_t)$ — so the estimator is
**doubly robust**: consistent for $\psi_t$ if *either* $\pi_t$ *or* $q_t$ is
correct (versus single robustness in $\pi_t$ for the uncentered form).

**Closed-form solutions** (both linear in $\psi$):
$$
\hat\psi_2=\Big\{\mathbb{P}_n\big[m_2\,(A_2-\pi_2)\,A_2\,m_2^\top\big]\Big\}^{-1}\mathbb{P}_n\big[m_2\,(A_2-\pi_2)\,(Y-q_2)\big],
$$
$$
\hat\psi_1=\Big\{\mathbb{P}_n\big[m_1\,(A_1-\pi_1)\,A_1\,m_1^\top\big]\Big\}^{-1}\mathbb{P}_n\big[m_1\,(A_1-\pi_1)\,(R_1-q_1)\big],
\quad R_1=Y-\{A_2-(1-A_0)A_1\}\gamma_2(\bar S_2;\hat\psi_2).
$$

**Plug-in order.** (1) Fit $q_2$ (OLS of $Y$ on the $A_2=0$ stratum) and solve
$U_2$ for $\hat\psi_2\Rightarrow\hat\gamma_2$. (2) Fit $q_1$ (OLS of $\hat q_2$ on
the $A_1=0$ stratum), form $R_1$, and solve $U_1$ for
$\hat\psi_1\Rightarrow\hat\gamma_1$. (3) Plug $\hat\gamma_2,\hat\gamma_1$ (and true
$\pi_0,\pi_1$) into the proposed estimator $\widehat{\tau}$ of §3.2 (and
$\hat\gamma_2$ into the partial IPW of §3.3).

---

## 4. Simulation plan

1. **Generate** $N$ i.i.d. subjects from §1 (e.g. $N=2000$).
2. **Fit nuisance models** (or use the true ones for an oracle check):
   - $\pi_0(L_0)$, $\pi_1(\bar S_1)$, $\pi_2(\bar S_2)$ via logistic regression;
   - centering regressions $q_2$ (OLS on the $A_2=0$ stratum) and $q_1$ (OLS of
     $\hat q_2$ on the $A_1=0$ stratum);
   - blip functions $\gamma_1,\gamma_2$ via centered g-estimation / known truth.
3. **Form the estimator terms** of §3: proposed $\widehat{\tau}$, partial IPW
   $\widehat{\tau}^{\text{pIPW}}$, fully IPW $\widehat{\tau}^{\text{IPW}}$.
4. **Estimate** $\mathbb{E}(Y^{0,1,0}-Y^{1,1,0}\mid L_0)$ (e.g. project onto
   $L_0$, or report the marginal average since the truth is constant here).
5. **Repeat** over $R$ Monte Carlo replications; report bias, empirical SE,
   and coverage against the truth $-1.75$.

**Oracle sanity check:** plug the true $\pi_0,\pi_1,\gamma_1,\gamma_2$ into the
estimator — the sample mean of each population term, differenced, should
converge to $-1.75$.
