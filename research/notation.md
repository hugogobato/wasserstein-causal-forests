# ODCF notation lock

This sheet fixes notation for the repaired formulation. The term “distributional treatment effect” is not used without an explicit qualifier.

## Data and sampling

The outer sample contains `n` regions, indexed by \(i=1,\ldots,n\). Region \(i\) has pretreatment covariates \(X_i\in\mathcal X\subseteq\mathbb R^d\), binary region-level treatment \(Z_i\in\{0,1\}\), and an inner household sample \(\mathcal S_i\). The number of inner observations is \(m_i=|\mathcal S_i|\). The primary target weights regions equally.

The latent potential outcome under treatment \(z\) is a nonnegative raw-income probability distribution \(\mu_{i,\mathrm{raw}}^{(z)}\in\mathcal P_2([0,\infty))\). Its raw quantile uses the generalized-inverse convention

\[
Q_{i,\mathrm{raw}}^{(z)}(p)
=
\inf\{y:F_{i,\mathrm{raw}}^{(z)}(y)\ge p\},
\qquad p\in(0,1).
\]

The analysis quantile is the transformed function

\[
Q_i^{(z)}(p)=\log\{1+Q_{i,\mathrm{raw}}^{(z)}(p)\}.
\]

Under consistency, no hidden treatment versions, and no interference between outer units, the observed latent analysis outcome is \(Q_i=Q_i^{(Z_i)}\). The actual observed data object is

\[
O_i=(X_i,Z_i,\mathcal S_i),
\]

while \(O_i^\star=(X_i,Z_i,Q_i)\) denotes oracle data. The inner sample yields an estimator \(\widehat Q_i\) of the transformed quantile. With bounded \(m_i\), \(\widehat Q_i\) does not generally identify the latent random distribution or its mean quantile. Feasible latent-target claims therefore invoke A8: either \(\min_i m_i\to\infty\) with the required uniform error rate, or an identified measurement, replicate, or validation model.

## Quantile scale and finite grid

The primary outcome coordinate uses \(Y^{\log1p}=\log(1+Y^{\mathrm{raw}})\) and the fixed quantile interval

\[
\mathcal P=[0.05,0.95].
\]

For the primary pilot, \(K=49\) and

\[
p_k=0.05+0.90(k-1)/48,\qquad k=1,\ldots,K.
\]

Let \(w_k\) be trapezoidal quadrature weights on this grid, with \(\sum_{k=1}^K w_k=0.90\). The discrete curve norm is

\[
\|q\|_{K,w}^2=\sum_{k=1}^K w_kq_k^2.
\]

The raw-income quantile associated with a log1p quantile is \(Q^{\mathrm{raw}}(p)=\exp\{Q^{\log1p}(p)\}-1\). The fixed functionals below are applied to the raw-income distribution represented by that recovered quantile.

Empirical quantiles use the same generalized-inverse convention, not linear interpolation between order statistics. Thus a sample empirical cdf \(\widehat F_i\) gives \(\widehat Q_{i,\mathrm{raw}}(p)=\inf\{y:\widehat F_i(y)\ge p\}\), followed by the log1p transform.

## Functionals and target coordinates

The version-1 collection has \(J=3\): Gini, Theil T, and Atkinson with aversion parameter \(\varepsilon_A=0.5\). Their exact finite-grid formulas and numerical domain checks are specified in WP2, not invented here.

For an analysis-scale quantile function \(Q\), define the unscaled scientific and score vector

\[
U_K(Q)
=
\big(
Q(p_1),\ldots,Q(p_K),
T_1\{\exp(Q)-1\},\ldots,T_J\{\exp(Q)-1\}
\big).
\]

Training-only positive coordinate scales define the split operator

\[
S_{\mathrm{train}}
=
\operatorname{diag}(I_K,s_1,\ldots,s_J),
\]

where \(s_j>0\) are estimated on training data only. \(S_{\mathrm{train}}\) changes the split norm, not \(U_K\), the nuisance regressions, leaf means, treatment-effect targets, or reported coordinates. The curve block retains the deterministic quadrature weights \(w_k\).

## Conditional means and effects

For \(z\in\{0,1\}\), define the scientific conditional mean

\[
M_z(x)=E\{U_K(Q_i^{(z)})\mid X_i=x\}.
\]

The finite-grid scientific target is

\[
\Theta_K(x)=M_1(x)-M_0(x).
\]

Its first \(K\) coordinates are the mean quantile-effect curve on the grid,

\[
\tau_{Q,K}(x,p_k)=E\{Q_i^{(1)}(p_k)-Q_i^{(0)}(p_k)\mid X_i=x\},
\]

and its last \(J\) coordinates are the unscaled scientific effects

\[
\delta_{T_j}(x)
=
E\!\left[
T_j\{\exp(Q_i^{(1)})-1\}
-
T_j\{\exp(Q_i^{(0)})-1\}
\mid X_i=x
\right].
\]

These conditional quantities are defined only up to \(P_X\)-almost-everywhere equivalence unless a continuous version is selected under additional assumptions. The notation \(\Theta(x)\) without a subscript refers to the continuum target only when a later work package explicitly opens that theory. The confirmatory low-dimensional target is \(\Theta_V(v)=E\{\Theta_K(X)\mid V=v\}\), defined \(P_V\)-almost everywhere, and the frozen-subgroup target is \(\Theta_g=E\{\Theta_K(X)\mid X\in A_g\}\) for \(P(X\in A_g)>0\).

## Nuisances, score, and forest weights

The propensity and arm-specific outcome regressions are

\[
e(x)=P(Z=1\mid X=x),\qquad m_z(x)=E\{U_K(Q)\mid Z=z,X=x\}.
\]

For \(\eta=(e,m_0,m_1)\), the finite-vector oracle orthogonal score is

\[
\phi^\star(O^\star;\eta)
=
m_1(X)-m_0(X)
+\frac{Z}{e(X)}\{U_K(Q)-m_1(X)\}
-\frac{1-Z}{1-e(X)}\{U_K(Q)-m_0(X)\}.
\]

In the feasible implementation, \(U_K(Q)\) is replaced by \(U_K(\widehat Q)\) and all nuisance predictions are cross-fitted. This is a proxy score until A8 supplies an observation-recovery argument; cross-fitting alone does not remove inner-sample bias.

The honest forest prediction at query point \(x\) is a weighted score average,

\[
\widehat\Theta_K(x)=\sum_{i=1}^n\alpha_i(x)\widehat\phi_i,
\]

where \(\alpha_i(x)\ge 0\), \(\sum_i\alpha_i(x)=1\), and \(\alpha_i(x)\) are honest forest weights. Split-selection observations and leaf-estimation observations are distinct within each honest tree.

## Terminology guardrail

Use the following explicit terms:

| Term | Meaning |
|---|---|
| Distribution-valued outcome | The region-level potential outcome \(\mu_i^{(z)}\) or its quantile function \(Q_i^{(z)}\). |
| Conditional law of a scalar outcome | A law such as \(\mathcal L(Y\mid X=x)\), the object targeted by Distributional Random Forests. |
| Barycenter contrast | A contrast between arm-specific conditional mean quantile curves, such as \(M_1^Q(x)-M_0^Q(x)\). |
| Mean quantile-effect curve | \(E[Q^{(1)}-Q^{(0)}\mid X=x]\), represented by \(\tau_{Q,K}\) on the finite grid. |
| Nonlinear functional effect | \(E[T(Q^{(1)})-T(Q^{(0)})\mid X=x]\), represented by \(\delta_T\). |
| Distribution-valued causal effect | The full scientific vector \(\Theta_K(x)\), only when the finite-grid or continuum scope and oracle/feasible observation regime are stated. |

The identity \(T(EQ) = E[T(Q)]\) is not assumed. It generally fails for nonlinear \(T\).

## Collapse check

When \(J=0\) and \(K=1\), the oracle problem is an ordinary scalar CATE and \(\phi^\star\) is the usual scalar doubly robust score. When \(m_i=1\), only the empirical point-mass proxy collapses toward scalar HTE. The unrestricted latent distribution target need not collapse and is generally nonidentified from one inner draw.
