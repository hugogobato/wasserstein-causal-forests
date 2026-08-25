# C-WDB proper-score and invariance certificate

**Certificate ID:** `G0-WP0-B-v1`  
**Contract:** `research/estimand_contract.md`  
**Assumptions:** `research/assumption_ledger.md`  
**Executable check:** `research/checks/check_proper_score.py`  
**Gate verdict:** `PASS`

## 1. Audited specification

For positive normalized weights \(w_{1:K}\), let

\[
W=\operatorname{diag}(w_1,\ldots,w_K),\qquad
d_W(p,q)=\{(p-q)^\top W(p-q)\}^{1/2},
\]

on the closed monotone cone

\[
\mathcal Q_K=\{q\in\mathbb R^K:q_1\leq\cdots\leq q_K\}.
\]

The map \(L(q)=W^{1/2}q\) is injective and
\(d_W(p,q)=\lVert L(p)-L(q)\rVert_2\). The unrestricted arm target is
\(P_a^K(x)=\mathcal L\{q(Y^a)\mid X=x\}\), identified for
\(P_X\)-almost every \(x\) under A1 through A4. C-WDB represents a forecast by

\[
P_M=M^{-1}\sum_{m=1}^M\delta_{p_m},\qquad p_m\in\mathcal Q_K.
\]

The representation is an empirical law with \(M\) slots and possible repeated
particles. Its atom masses after merging duplicates are integer multiples of
\(1/M\). Particle labels are not parameters of the reported law.

## 2. Certified score, including collisions

For \(\varepsilon\geq0\), define

\[
\rho_\varepsilon(p,q)
=\left\{d_W^2(p,q)+\varepsilon^2\right\}^{1/2}-\varepsilon
\]

and the loss-oriented energy score

\[
S_\varepsilon(P,y)
=E_{Z\sim P}\rho_\varepsilon(Z,y)
-\frac12E_{Z,Z'\sim P}\rho_\varepsilon(Z,Z').
\]

At \(\varepsilon=0\), this is the weighted finite-grid energy score required by
the phase:

\[
S_{M,0}(p_{1:M},y)
=\frac1M\sum_{m=1}^M d_W(p_m,y)
-\frac1{2M^2}\sum_{m,\ell=1}^M d_W(p_m,p_\ell).
\]

For \(\varepsilon>0\), \(S_\varepsilon\) is the exact score being optimized,
not an undocumented denominator patch. The score manifest must record
\(\varepsilon\), and objective, gradient, tuning risk, and fixed-\(M\) target
must use that same value. Subtracting \(\varepsilon\) makes
\(\rho_\varepsilon(p,p)=0\); constants would cancel from the energy
divergence in any event.

The \(\varepsilon=0\) convention selects the zero subgradient when two
arguments coincide. The recommended differentiable configuration for tree
fitting is \(\varepsilon>0\). Phase 2 must tune or freeze its numerical scale
relative to the rescaled \(z\) coordinates and may not change it after seeing
test outcomes.

## 3. Strict propriety on the weighted monotone cone

Let \(P,Q\in\mathcal P_1(\mathcal Q_K,d_W)\), let \(X,X'\) be independent
from \(P\), and let \(Y,Y'\) be independent from \(Q\). The population risk is

\[
R_\varepsilon(P,Q)=E_{Y\sim Q}S_\varepsilon(P,Y).
\]

Direct expansion gives

\[
\begin{aligned}
R_\varepsilon(P,Q)-R_\varepsilon(Q,Q)
&=\frac12D_{\rho_\varepsilon}(P,Q),\\
D_{\rho_\varepsilon}(P,Q)
&=2E\rho_\varepsilon(X,Y)
-E\rho_\varepsilon(X,X')
-E\rho_\varepsilon(Y,Y').
\end{aligned}
\]

For \(t\geq0\),

\[
\sqrt{t+\varepsilon^2}-\varepsilon
=\frac1{2\sqrt\pi}\int_0^\infty
\{1-\exp(-st)\}\exp(-s\varepsilon^2)s^{-3/2}\,ds.
\]

Applying this identity with
\(t=\lVert L(p)-L(q)\rVert_2^2\) yields

\[
D_{\rho_\varepsilon}(P,Q)
=\frac1{2\sqrt\pi}\int_0^\infty
\operatorname{MMD}_{k_s}^2(L_\#P,L_\#Q)
\exp(-s\varepsilon^2)s^{-3/2}\,ds,
\]

where \(k_s(z,z')=\exp\{-s\lVert z-z'\rVert_2^2\}\). Each integrand is
nonnegative. Gaussian kernels are characteristic on Euclidean space, and the
mixing density is strictly positive for every \(s>0\). Therefore

\[
D_{\rho_\varepsilon}(P,Q)=0
\quad\Longleftrightarrow\quad
L_\#P=L_\#Q
\quad\Longleftrightarrow\quad
P=Q.
\]

Thus \(S_\varepsilon\) is strictly proper on
\(\mathcal P_1(\mathcal Q_K,d_W)\) for every \(\varepsilon\geq0\). Restricting
the observation and forecast space to the closed subset \(\mathcal Q_K\) does
not weaken strictness, and the positive weights make \(L\) injective. This
argument certifies the weighted, projected, and collision-smoothed score that
the implementation is allowed to use.

For the conventional \(\varepsilon=0\) score, this agrees with the energy-score
strict-propriety result in Gneiting and Raftery (2007, Theorem 5). The
energy-distance characterization in Euclidean space is also developed by
Székely and Rizzo (2013), and the distance/RKHS equivalence used above is
formalized by Sejdinovic et al. (2013).

## 4. Empirical particle score and full-ensemble gradient

Write

\[
r_{my,\varepsilon}
=\{(p_m-y)^\top W(p_m-y)+\varepsilon^2\}^{1/2}
\]

and

\[
r_{m\ell,\varepsilon}
=\{(p_m-p_\ell)^\top W(p_m-p_\ell)+\varepsilon^2\}^{1/2}.
\]

Differentiating the attraction term and both ordered appearances of \(p_m\)
in the repulsion term gives

\[
\nabla_{p_m}S_{M,\varepsilon}
=\frac1M\frac{W(p_m-y)}{r_{my,\varepsilon}}
-\frac1{M^2}\sum_{\ell=1}^M
\frac{W(p_m-p_\ell)}{r_{m\ell,\varepsilon}}.
\]

For \(\varepsilon=0\), a zero numerator and denominator contributes the
declared zero subgradient. The second summand couples every particle to the
entire ensemble. A particlewise attraction gradient that omits this term is
not the gradient of the certified score.

The executable central-difference checks use an away-from-kink case at
\(\varepsilon=0\) and a smooth case at \(\varepsilon=10^{-3}\). Their maximum
absolute errors in the recorded run were

| Score | Maximum absolute gradient error | Required tolerance |
|---|---:|---:|
| \(S_{M,0}\) | \(6.03\times10^{-11}\) | \(10^{-6}\) |
| \(S_{M,10^{-3}}\) | \(7.55\times10^{-11}\) | \(10^{-6}\) |

Coincident outcome and particle locations, and coincident particle pairs,
produce finite scores and gradients under both the zero-subgradient and
smooth conventions.

## 5. Unrestricted and fixed-\(M\) targets

Strict propriety establishes

\[
\arg\min_{P\in\mathcal P_1(\mathcal Q_K,d_W)}
E\{S_\varepsilon(P,q(Y^a))\mid X=x\}
=\{P_a^K(x)\}.
\]

It does not establish exact recovery within a finite particle class. The
implemented population target is

\[
P_{a,M,\varepsilon}^{K,\star}(x)
\in\arg\min_{P\in\mathcal P_M^{\mathrm{emp}}(\mathcal Q_K)}
E\{S_\varepsilon(P,q(Y^a))\mid X=x\}.
\]

Existence follows under A8. Parameterize the class by
\(p_{1:M}\in\mathcal Q_K^M\). The risk is continuous. With
\(r_m=d_W(p_m,0)\) and
\(C=E\{d_W(q(Y^a),0)\mid X=x\}<\infty\), the triangle inequality and the
missing diagonal pairs give

\[
R_\varepsilon(p_{1:M})
\geq \frac1{M^2}\sum_{m=1}^M r_m-C-\varepsilon.
\]

Hence every sublevel set is bounded. It is also closed in finite dimension,
so a minimizer exists. Uniqueness of the parameter vector is impossible
because permutations leave the law unchanged, and uniqueness of the optimal
empirical law is not claimed without additional conditions.

Representability is necessary for a fixed-\(M\) minimizer to equal the truth.
The classes for successive \(M\) are not generally nested, so Phase 0 does not
claim universal monotonicity in \(M\). It verifies the declared ladder in a
known law.

### Known-law finite-\(M\) falsifier

Take \(q(Y)=b+U\mathbf 1_K\), where \(b\in\mathcal Q_K\),
\(U\sim\operatorname{Uniform}[-1,1]\), and \(\sum_k w_k=1\). Then
\(d_W\{b+s\mathbf 1_K,b+t\mathbf 1_K\}=|s-t|\). Put \(M\) particles at the
midpoints

\[
c_m=-1+\frac{2m-1}{M}.
\]

For the exact energy score,

\[
R_M=\frac13+\frac1{6M^2},
\qquad
R_\infty=\frac13.
\]

The executable check reproduces

| \(M\) | Exact risk | Excess over unrestricted truth |
|---:|---:|---:|
| 2 | 0.3750 | 0.0416667 |
| 5 | 0.3400 | 0.0066667 |
| 10 | 0.3350 | 0.0016667 |
| 25 | 0.3336 | 0.0002667 |

The formula error was below \(6\times10^{-17}\), and risk decreased at each
declared step. This is a known-law implementation certificate, not a general
rate theorem for conditional tree estimates.

## 6. Particle-permutation invariance

For a permutation \(\pi\) of \(\{1,\ldots,M\}\),

\[
\frac1M\sum_m\rho_\varepsilon(p_{\pi(m)},y)
=\frac1M\sum_m\rho_\varepsilon(p_m,y),
\]

and the double repulsion sum is unchanged by the bijection
\((m,\ell)\mapsto\{\pi(m),\pi(\ell)\}\). Therefore

\[
S_{M,\varepsilon}(p_{\pi(1)},\ldots,p_{\pi(M)};y)
=S_{M,\varepsilon}(p_1,\ldots,p_M;y).
\]

Every integral \(M^{-1}\sum_m h(p_m)\) and every symmetric pairwise
diagnostic is invariant as well. The gradient is equivariant:

\[
\nabla_{p_{\pi(m)}}S(p_{\pi(1:M)},y)
=\nabla_{p_m}S(p_{1:M},y).
\]

The recorded maximum change across the checked law-invariant outputs was
\(2.23\times10^{-16}\), below the required \(10^{-12}\). Gradient
equivariance error was \(1.39\times10^{-17}\). Independent permutations must
be tested separately within each treatment arm in Phase 2.

## 7. Monotone-cone projection

The projection used for a candidate vector \(v\in\mathbb R^K\) is

\[
\Pi_W(v)
=\arg\min_{q_1\leq\cdots\leq q_K}
\frac12\sum_{k=1}^K w_k(q_k-v_k)^2.
\]

Because \(W\) is positive definite and \(\mathcal Q_K\) is nonempty, closed,
and convex, this projection exists and is unique. Weighted
pool-adjacent-violators computes it by replacing each violating adjacent block
with its weighted mean. The check script verifies primal feasibility, dual
feasibility, stationarity, and complementarity; its recorded maximum KKT
residual was \(1.74\times10^{-17}\). Projected collisions yield finite score.

Projection keeps particles inside the same space on which strict propriety was
proved. Phase 0 does not claim that projecting an arbitrary tree step always
decreases the score. One-step descent and projection-magnitude diagnostics
remain Phase 2 obligations.

## 8. Squared-\(W_2\) collapse witness

The improper ablation removes the repulsion term and minimizes

\[
L_{\mathrm{sq}}(p_{1:M})
=\frac1M\sum_{m=1}^M E\{d_W^2(p_m,q(Y))\}.
\]

Each summand is independent of the other particles. Since \(\mathcal Q_K\) is
convex and \(E\{q(Y)\}\in\mathcal Q_K\), every particle has the same unique
minimizer

\[
p_m^\star=E\{q(Y)\},\qquad m=1,\ldots,M.
\]

In one dimension this is the grid quantile vector of the Wasserstein
barycenter. Thus the squared-\(W_2\) particle objective necessarily collapses
all particles, even when the outer law is nondegenerate.

The numerical witness uses \(K=1\), for which squared \(W_2\) is ordinary
squared distance between point masses. Ten particles initialized with spread
4 converge to the outcome mean; final spread was
\(3.47\times10^{-18}\) and maximum center error was
\(1.74\times10^{-18}\), both below \(10^{-12}\).

## 9. Identification and representation checks

The companion contract proves the arm-marginal g-formula and supplies an
observational-equivalence witness showing that the joint law
\(\mathcal L(Y^0,Y^1\mid X)\) is not identified. The automated lint rejects
both a structured declaration with an outcome-level target and barycenter
prediction, and explicit barycenter-as-outcome assignment text.

The common identified target for C-WDB, PTA-BCF, and W-CausalDRF is the
finite-grid causal mean contrast, with predeclared grid-measurable functionals
as additional common targets. Exact continuum reference effects and their grid
approximations have different IDs and cannot be pooled.

## 10. Reproducible verification

From the repository root, run

```bash
python research/checks/check_proper_score.py
```

The script exits with code 0 only if all nine check groups pass. The recorded
environment was Python 3.12.3 with NumPy 2.4.3. The run took less than one
second and used one process.

| Check group | Recorded result | Gate threshold |
|---|---:|---:|
| weighted rescaling identity | \(1.12\times10^{-16}\) error | \(<10^{-12}\) |
| analytic gradients | \(<7.6\times10^{-11}\) error | \(<10^{-6}\) |
| collision behavior | all finite | finite |
| particle permutation | \(2.23\times10^{-16}\) output error | \(<10^{-12}\) |
| monotone projection | \(1.74\times10^{-17}\) KKT residual | \(<10^{-10}\) |
| strict-propriety witness | collapsed-law energy distance \(1.0\) | \(>0\) |
| finite-\(M\) risk ladder | strictly decreasing | required |
| squared-\(W_2\) collapse | \(3.47\times10^{-18}\) final spread | \(<10^{-12}\) |
| estimand lint | both invalid witnesses rejected | required |

## 11. G0 decision

**Decision: `GO`.**

The exact weighted score and its declared smooth collision version are
strictly proper over the unrestricted finite-grid law space. The empirical
gradient matches the selected score, including the ensemble repulsion term.
The restricted finite-\(M\) target is exposed and has a population minimizer
under A8. Law-invariant outputs do not depend on particle labels, monotone
projection is valid, and the improper squared-\(W_2\) alternative collapses as
predicted.

The decision does not certify boosting consistency, projected descent with
tree approximation, a universal finite-\(M\) rate, or continuum recovery as
\(K\to\infty\). Those claims remain outside Phase 0.

## References

Gneiting, T., and Raftery, A. E. (2007). “Strictly Proper Scoring Rules,
Prediction, and Estimation.” *Journal of the American Statistical Association*,
102(477), 359-378.
[DOI](https://doi.org/10.1198/016214506000001437),
[author manuscript](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

Sejdinovic, D., Sriperumbudur, B., Gretton, A., and Fukumizu, K. (2013).
“Equivalence of Distance-Based and RKHS-Based Statistics in Hypothesis
Testing.” *The Annals of Statistics*, 41(5), 2263-2291.
[DOI](https://doi.org/10.1214/13-AOS1140),
[open manuscript](https://arxiv.org/abs/1207.6076).

Székely, G. J., and Rizzo, M. L. (2013). “Energy Statistics: A Class of
Statistics Based on Distances.” *Journal of Statistical Planning and
Inference*, 143(8), 1249-1272.
[DOI](https://doi.org/10.1016/j.jspi.2013.03.018).
