# WP2-T0: finite-grid estimands

**Input files:** `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md` Sections 2.1–2.8 and 3; `research/notation.md`; `research/assumption_ledger.md`.

**Source to adapt:** Lin, Kong, and Wang (2023), Theorems 1–2, for the distribution-valued causal identification template. The present file defines a finite implementation target and does not import their continuum asymptotics.

**Assumptions used:** A2–A7 for the scientific target, with A11 only for the separate training-time split scales.

**Status:** `PASS`, finite-grid scope only.

## 1. Frozen grids

The primary curve grid is the fixed interior grid from WP0:

\[
 p_k=0.05+\frac{0.90(k-1)}{48},\qquad k=1,\ldots,K,\qquad K=49.
\]

Let \(h=0.90/48\). The trapezoidal weights are

\[
 w_1=w_K=h/2,\qquad w_k=h\quad (2\leq k\leq K-1),
 \qquad \sum_{k=1}^K w_k=0.90.
\]

The curve coordinate is therefore the discretization of the analysis-scale quantile on \(\mathcal P=[0.05,0.95]\), with norm

\[
 \|q\|_{K,w}^2=\sum_{k=1}^K w_kq_k^2.
\]

The inequality functionals require integration over the full probability range. They use a separate fixed midpoint grid

\[
 r_\ell=\frac{\ell-1/2}{L},\qquad \omega_\ell=\frac1L,
 \qquad \ell=1,\ldots,L,\qquad L=400.
\]

The even value \(L=400\) makes the two-point test at probability \(1/2\) exact under midpoint quadrature. This is a numerical-definition choice, not a data-dependent tuning parameter. It is recorded in `research/decision_log.md` as WP2-D010.

## 2. Quantile representation and raw-income recovery

All population and empirical quantiles use the lower generalized inverse

\[
 Q(p)=\inf\{y:F(y)\geq p\}.
\]

For an empirical CDF this is the inverted-CDF convention, including selection of the lower support point when \(p\) equals a jump height. Interpolated sample-quantile conventions define a different finite-sample proxy and must not be substituted silently.

Let \(q_k\) denote the analysis quantile, on the fixed `log1p` scale, at \(p_k\). For scalar inequality coordinates, let \(q^{\mathrm{fun}}_\ell\) denote the same analysis quantile evaluated at \(r_\ell\). The raw-income quantile used by the inequality formulas is

\[
 y_\ell=\exp\{q^{\mathrm{fun}}_\ell\}-1.
\]

Thus the curve coordinate and the inequality coordinates have a declared scale distinction. A functional is evaluated on the recovered raw-income distribution before any outer conditional average is taken.

## 3. Finite-grid scalar functionals

Define the finite-grid raw mean

\[
 \mu_L(y)=\sum_{\ell=1}^L\omega_\ell y_\ell.
\]

The finite-grid Gini coordinate is the quantile formula

\[
 G_L(y)=1-\frac{2}{\mu_L(y)}
 \sum_{\ell=1}^L\omega_\ell(1-r_\ell)y_\ell.
\]

The finite-grid Theil T coordinate is

\[
 \operatorname{Theil}_L(y)=
 \sum_{\ell=1}^L\omega_\ell
 \frac{y_\ell}{\mu_L(y)}
 \log\left(\frac{y_\ell}{\mu_L(y)}\right),
\]

with the convention \(0\log 0=0\). The finite-grid Atkinson coordinate uses the frozen aversion parameter \(\varepsilon_A=0.5\):

\[
 \operatorname{Atkinson}_{L,0.5}(y)
 =1-\frac{\left\{\sum_{\ell=1}^L\omega_\ell y_\ell^{1-\varepsilon_A}\right\}^{1/(1-\varepsilon_A)}}{\mu_L(y)}
 =1-\frac{\left\{\sum_{\ell=1}^L\omega_\ell\sqrt{y_\ell}\right\}^{2}}{\mu_L(y)}.
\]

The domain check for all three coordinates is \(y_\ell\geq 0\) and \(\mu_L(y)>0\). Theorem-level work involving these functionals additionally uses A7, including the required bounded-away-from-zero mean condition. The finite-grid definitions are the estimands; they are not silently identified with their continuum counterparts at finite \(L\).

## 4. Target vector

Let

\[
 T_1=G_L,\qquad T_2=\operatorname{Theil}_L,\qquad
 T_3=\operatorname{Atkinson}_{L,0.5}.
\]

Define the unscaled scientific outcome vector

\[
 U_K(Q)=\left(q_1,\ldots,q_K,
 T_1\{y(Q)\},T_2\{y(Q)\},T_3\{y(Q)\}\right)\in\mathbb R^{K+3},
\]

where \(y(Q)_\ell=\exp\{q_\ell^{\mathrm{fun}}\}-1\), and the dependence on the fixed functional grid \(L=400\) is suppressed in the notation. This vector, its conditional means, its treatment contrast, and all reported errors are unscaled.

Training may use the separate diagonal split transform

\[
 S_{\mathrm{train}}
 =\operatorname{diag}(1,\ldots,1,s_1,s_2,s_3),
 \qquad s_j>0,
\]

or, equivalently, the squared-norm weights

\[
 (w_1,\ldots,w_K,s_1^2,s_2^2,s_3^2).
\]

The \(w_k\) are deterministic quadrature weights fixed by the grid. Only the \(s_j\) are learned from training data. \(S_{\mathrm{train}}\) changes split geometry but not \(U_K\), \(M_z^K\), \(\Theta_K\), nuisance targets, leaf predictions, or reported estimands.

For \(z\in\{0,1\}\) and \(P_X\)-almost every \(x\),

\[
 M_z^K(x)=E\{U_K(Q_i^{(z)})\mid X_i=x\},
 \qquad
 \Theta_K(x)=M_1^K(x)-M_0^K(x).
\]

The first \(K\) coordinates are

\[
 \tau_{Q,K}(x,p_k)=E\{Q_i^{(1)}(p_k)-Q_i^{(0)}(p_k)\mid X_i=x\},
\]

and the remaining coordinates are

\[
 \delta_{T_j}(x)
 =E[
 T_j\{y(Q_i^{(1)})\}-T_j\{y(Q_i^{(0)})\}
 \mid X_i=x
 ],
 \qquad j=1,2,3.
\]

The order of operations is part of the target definition:

\[
 E[T_j\{y(Q_i^{(z)})\}\mid X_i=x]
 \quad\text{is targeted, whereas}\quad
 T_j\!\left[
 \exp\{E(Q_i^{(z)}\mid X_i=x)\}-1
 \right]
 \quad\text{is not.}
\]

The observed finite implementation replaces \(Q_i\) by \(\widehat Q_i\) in \(U_K\). Its discrepancy from the latent target is an explicit WP5 issue under A8.

## 5. Collapse check

With \(J=0\) and \(K=1\), \(U_K(Q)\) is a scalar and \(\Theta_K\) is an ordinary scalar CATE. With randomized treatment and a known design propensity \(e(x)\in(0,1)\), which need not equal \(1/2\) and may depend on design strata in \(X\), the target remains the same and the propensity nuisance need not be fitted.

## Checks run

The exact arithmetic and score invariants are implemented in `research/wp2_invariants.py`. The script checks the finite-grid Gini values used by WP2-T4, the scalar collapse, the randomized-propensity case, nonlinear-before-averaging, and the projection guard.

## Observed failures

None at the finite-grid-definition level.

## Unresolved questions

The split-only choice of \(s_j\) is intentionally deferred to WP3-T3. Inner-sample bias and variance are intentionally deferred to WP5 under A8. No continuum or growing-grid claim is made.

`PASS`
