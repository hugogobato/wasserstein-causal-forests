# Frozen estimand and representation contract

**Contract ID:** `G0-WP0-A-v1`  
**Observation regime:** `ORACLE-V1`  
**Status:** frozen for Phases 1 through 6  
**Source of truth:** `research_phases/_phase_shared.md`, with the finite-grid distinctions below made explicit

## 1. Observable data and mathematical spaces

The observational unit is

\[
O=(X,A,Y),\qquad A\in\{0,1\},\qquad Y=Y^A,
\]

where \(X\) is pre-treatment, \(Y^0,Y^1\) are random elements of
\(\mathcal P_2(\mathbb R)\), and \(V=g(X)\) is fixed before outcome
analysis. The covariate space is assumed to be standard Borel so regular
conditional laws can be chosen. All conditional targets below are statements
for \(P_X\)-almost every \(x\), or \(P_V\)-almost every \(v\). They are not
pointwise claims at covariate values outside the observed support.

For \(0<u_1<\cdots<u_K<1\), use the left-continuous generalized inverse

\[
Q_Y(u)=\inf\{t:F_Y(t)\geq u\}.
\]

The finite-grid representation and its state space are

\[
q(Y)=\{Q_Y(u_1),\ldots,Q_Y(u_K)\}\in
\mathcal Q_K,\qquad
\mathcal Q_K=\{q\in\mathbb R^K:q_1\leq\cdots\leq q_K\}.
\]

Weights \(w_k>0\) are fixed in the evaluation manifest and normalized so
\(\sum_k w_k=1\). Define \(W=\operatorname{diag}(w_1,\ldots,w_K)\),
\(z=W^{1/2}q\), and

\[
d_W(q,q')=\lVert q-q'\rVert_W
=\left\{\sum_{k=1}^K w_k(q_k-q'_k)^2\right\}^{1/2}.
\]

The name \(W_{2,K}\) refers to this declared quadrature metric. It is not
silently identified with the continuum \(W_2\).

## 2. Frozen estimand dictionary

| ID | Level | Definition | Identified object |
|---|---|---|---|
| `LAW-A-K` | finite-grid outcome law | \(P_a^K(x)=\mathcal L\{q(Y^a)\mid X=x\}\) | arm-specific conditional marginal law |
| `MEANQ-A-K` | finite-grid barycenter coordinate | \(\bar q_a(x)=E\{q(Y^a)\mid X=x\}\) | conditional mean quantile vector |
| `BARY-A` | continuum barycenter | \(m_a(x)\in\arg\min_\nu E\{W_2^2(Y^a,\nu)\mid X=x\}\) | supporting target, when its outer second moment is finite |
| `TATE-OUT-j` | outcome level | \(E\{T_j(Y^1)\}-E\{T_j(Y^0)\}\) | difference of arm marginal expectations |
| `TCATE-OUT-j` | outcome level | \(E[T_j(Y^1)-T_j(Y^0)\mid V=v]\) | moderator-specific difference of arm marginal expectations |
| `TATE-K-j` | finite-grid outcome level | \(E\{h_j(q(Y^1))\}-E\{h_j(q(Y^0))\}\) | grid-measurable version with \(T_j^K=h_j\circ q\) |
| `TCATE-K-j` | finite-grid outcome level | \(E[h_j(q(Y^1))-h_j(q(Y^0))\mid V=v]\) | grid-measurable moderator effect |
| `REF-A` | continuum outcome level | \(r_a(x)=E\{W_2(Y^a,\nu_\star)\mid X=x\}\) | exact reference-distance regression, when integrable |
| `REF-A-K` | finite-grid outcome level | \(r_a^K(x)=E\{d_W(q(Y^a),q(\nu_\star))\mid X=x\}\) | grid reference-distance regression |
| `REF-ATE` | continuum outcome level | \(\Delta_\star=E\{r_1(X)-r_0(X)\}\) | exact reference effect |
| `REF-ATE-K` | finite-grid outcome level | \(\Delta_\star^K=E\{r_1^K(X)-r_0^K(X)\}\) | grid reference effect |
| `REF-TCATE` | continuum outcome level | \(E\{r_1(X)-r_0(X)\mid V=v\}\) | exact conditional reference effect |
| `REF-TCATE-K` | finite-grid outcome level | \(E\{r_1^K(X)-r_0^K(X)\mid V=v\}\) | grid conditional reference effect |
| `LAW-A-M-K` | finite-particle law | \(P_{a,M,\varepsilon}^{K,\star}(x)\) defined in Section 6 | score projection, not generally \(P_a^K(x)\) |

Continuum targets and finite-grid targets must carry their distinct IDs in
every result row. Equality is claimed only when the functional is determined
exactly by the stored grid. Otherwise the grid target is an approximation
whose \(K\)-sensitivity must be reported. In particular, a fixed grid does not
determine an arbitrary \(T_j(Y)\) or the exact \(W_2(Y,\nu_\star)\).

## 3. Observable g-formulas

Under A1 through A4, for every Borel set \(B\subseteq\mathcal Q_K\),

\[
\begin{aligned}
P_a^K(x)(B)
&=P\{q(Y^a)\in B\mid X=x\}\\
&=P\{q(Y)\in B\mid A=a,X=x\}.
\end{aligned}
\]

Consequently, for every integrable measurable \(h:\mathcal Q_K\to\mathcal B\),

\[
\mu_{a,h}(x)
=E\{h(q(Y))\mid A=a,X=x\}
=E\{h(q(Y^a))\mid X=x\}.
\]

The identified marginal mean and moderator effect are

\[
\theta_{a,h}=E\{\mu_{a,h}(X)\},\qquad
\tau_h(v)=E\{\mu_{1,h}(X)-\mu_{0,h}(X)\mid V=v\}.
\]

The same formulas apply to an exact outcome functional \(T_j(Y)\) using
\(\mu_{a,j}(x)=E\{T_j(Y)\mid A=a,X=x\}\). Exact continuum functionals are
therefore identifiable from the oracle observations, but they are recoverable
from a fitted \(P_a^K(x)\) only when they are measurable functions of the
declared grid or when an explicit approximation contract is supplied.

For the arm marginal grid law,

\[
\mathcal L\{q(Y^a)\}(B)
=E\left[P\{q(Y)\in B\mid A=a,X\}\right].
\]

These formulas identify each arm marginal separately. They do not identify a
cross-arm coupling.

## 4. Outcome-level and barycenter-level effects are different

The one-dimensional conditional barycenter has quantile function

\[
Q_{m_a(x)}(u)=E\{Q_{Y^a}(u)\mid X=x\},
\]

whenever the conditional outer second moment exists. For nonlinear \(T_j\),

\[
T_j\{m_a(x)\}\neq E\{T_j(Y^a)\mid X=x\}
\]

in general. The left side is a functional of a barycenter. The right side is
an outcome-level conditional mean. A barycenter draw, posterior draw of a
mean surface, or mean quantile vector may not be relabeled as a draw from
\(\mathcal L(Y^a\mid X=x)\).

The equality

\[
q\{m_a(x)\}=E\{q(Y^a)\mid X=x\}
\]

holds coordinatewise in one dimension. It permits evaluation of
`MEANQ-A-K`; it does not turn a nonlinear transform of that vector into
`TATE-K-j` or `TCATE-K-j`.

## 5. The joint potential-outcome law is not identified

Observational equivalence fixes

\[
\mathcal L(Y^0\mid X=x),\qquad \mathcal L(Y^1\mid X=x),
\]

but not \(\mathcal L(Y^0,Y^1\mid X=x)\). A concrete witness takes \(X\)
constant, randomized \(A\), and \(Y^a=\delta_{B_a}\), where both \(B_0\) and
\(B_1\) are marginally Bernoulli\((1/2)\). The coupling \(B_1=B_0\) and the
coupling \(B_1=1-B_0\) induce the same observed law of \((A,Y)\), but

\[
P(Y^1=Y^0)=1
\quad\text{and}\quad
P(Y^1=Y^0)=0,
\]

respectively. Individual treatment-effect distributions, cross-arm particle
pairings, ranks shared across arms, and probabilities involving both potential
outcomes are therefore outside this contract.

`TATE-OUT-j` and `TCATE-OUT-j` remain identified because expectation is
linear and each uses only the two arm marginal expectations.

## 6. Particle-law contract

For particles \(p_1,\ldots,p_M\in\mathcal Q_K\),

\[
P_M=M^{-1}\sum_{m=1}^M\delta_{p_m}.
\]

The class \(\mathcal P_M^{\mathrm{emp}}(\mathcal Q_K)\) consists exactly of
such empirical measures, with repeated particles allowed. After repeated
locations are merged, atom masses are integer multiples of \(1/M\); the class
is not the class of arbitrary laws supported on at most \(M\) points.

For the score \(S_\varepsilon\) certified in
`research/cwdb_validity_certificate.md`,

\[
P_{a,M,\varepsilon}^{K,\star}(x)
\in\arg\min_{P\in\mathcal P_M^{\mathrm{emp}}(\mathcal Q_K)}
E\{S_\varepsilon(P,q(Y))\mid A=a,X=x\}.
\]

At fixed \(M\), C-WDB targets this restricted risk projection. It targets the
unrestricted truth \(P_a^K(x)\) only when that truth is representable in the
particle class. The score manifest must record \(K\), \(u_{1:K}\), \(w_{1:K}\),
\(M\), and the collision parameter \(\varepsilon\).

Particles are unordered within each arm. Every public output must be invariant
under independent permutations of the arm-0 and arm-1 particle labels. The
quantity \(p_{1m}(x)-p_{0m}(x)\) has no causal interpretation.

## 7. Common identified targets and metric registry

All three candidate methods can be compared without target substitution on
the grid causal mean

\[
\tau_q^K(x)=E\{q(Y^1)-q(Y^0)\mid X=x\},
\]

its rescaled version \(W^{1/2}\tau_q^K(x)\), and any predeclared
grid-measurable outcome functional that every method outputs. C-WDB integrates
the functional over its arm particle laws, PTA-BCF includes it as a fixed
target coordinate, and W-CausalDRF estimates the same contrast using its
treatment-aware forest weights.

| Metric ID | Truth object | Target level | Notes |
|---|---|---|---|
| `arm_energy_risk` | `LAW-A-K` | full grid law | arm-specific proper-score risk; not a joint-law metric |
| `kernel_law_error` | `LAW-A-K` | full grid law | declared characteristic-kernel discrepancy |
| `mean_quantile_rmse` | `MEANQ-A-K` or its arm contrast | grid outcome level | mandatory common target |
| `tate_functional_rmse` | `TATE-K-j` or explicitly labeled `TATE-OUT-j` | outcome level | target ID must name \(j\) and grid/continuum status |
| `tcate_functional_rmse` | `TCATE-K-j` or explicitly labeled `TCATE-OUT-j` | outcome level | evaluated only on supported \(v\) |
| `reference_effect_rmse` | `REF-ATE-K` or explicitly labeled `REF-ATE` | outcome level | grid and continuum versions cannot be pooled |
| `reference_tcate_rmse` | `REF-TCATE-K` or explicitly labeled `REF-TCATE` | outcome level | grid and continuum versions cannot be pooled |
| `barycenter_rmse` | `BARY-A` or `MEANQ-A-K` | barycenter level | never reported as an outcome-level functional error |
| `tail_calibration` | a predeclared event under `LAW-A-K` | grid law | event and threshold belong in the manifest |
| `mode_coverage` | `LAW-A-K` | grid law diagnostic | diagnostic for multimodal outer laws |
| `runtime` | algorithm execution | operational | not a causal estimand |
| `peak_ram` | algorithm execution | operational | not a causal estimand |

Every scientific result row must contain one target ID from Section 2.
Operational rows use `target_id=NONE_OPERATIONAL`. Method-specific diagnostics
may not be used to claim superiority on a causal estimand.

## 8. Machine-checkable prohibitions

The Phase 0 checker rejects a metric declaration that combines
`target_level=outcome` with `prediction_level=barycenter`, rejects explicit
barycenter-as-outcome assignment text, and verifies that the mandatory metric
families above occur in this contract. Later phase code must reuse that check.

The following semantic substitutions are prohibited:

| Invalid substitution | Required repair |
|---|---|
| \(T_j\{m_a(x)\}\) reported as \(E\{T_j(Y^a)\mid X=x\}\) | estimate \(T_j(Y)\) before averaging, or label a barycenter functional |
| exact \(W_2\) effect computed only from \(K\) grid points | label `REF-ATE-K`/`REF-TCATE-K`, or quantify grid error |
| fixed-\(M\) law reported as the unrestricted truth | label `LAW-A-M-K` and report \(M\)-sensitivity |
| paired particle differences reported as individual effects | integrate arm laws separately and contrast law-invariant summaries |
| a claim at every continuous \(x\) or \(v\) | restrict to the appropriate almost-everywhere version or add regularity |

## 9. WP0-A decision

**Verdict:** `PASS`, conditional on the clarified moment clauses in
`research/assumption_ledger.md`.

The observable g-formulas identify every frozen arm-marginal target. No primary
target requires the joint potential-outcome law. All three methods have the
common identified grid causal mean target. The finite-grid and continuum
targets, and the barycenter and outcome levels, are now explicitly separated.
