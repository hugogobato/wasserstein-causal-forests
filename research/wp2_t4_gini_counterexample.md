# WP2-T4: exact nonlinear-functional counterexample

**Input files:** `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md` Section 4, WP2-T4; `research/finite_grid_estimands.md`.

**Source to adapt:** None for the construction. The calculation is an original finite counterexample. The Gini quantile formula follows the functional definition in WP2-T0.

**Assumptions used:** A6 and A7 for the finite target. No causal identification assumption is needed to establish the algebraic separation between a mean quantile contrast and a nonlinear functional effect.

**Status:** `PASS`, exact arithmetic verified by `research/wp2_invariants.py`.

## Construction

Fix a covariate subgroup \(X\in\{A,B\}\). The values below are raw incomes. Quantiles use the lower generalized inverse

\[
 Q_{\mathrm{raw}}(p)=\inf\{y:F_{\mathrm{raw}}(y)\geq p\},
\]

so the lower support point is selected at \(p=1/2\). The curve coordinate is the corresponding analysis quantile

\[
 Q(p)=\log\{1+Q_{\mathrm{raw}}(p)\}.
\]

Let the raw control potential distribution be the same in both subgroups:

\[
 Q_{\mathrm{raw}}^{(0)}=\begin{cases}
 1,&0<p\leq 1/2,\\
 3,&1/2<p<1.
 \end{cases}
\]

In subgroup A, the treatment potential distribution is random across outer regions:

\[
 Q_{\mathrm{raw}}^{(1)}=
 \begin{cases}
 Q_{\mathrm{raw},\delta_1},&\text{with probability }1/2,\\
 Q_{\mathrm{raw},\{1,7\}},&\text{with probability }1/2,
 \end{cases}
\]

where \(Q_{\mathrm{raw},\delta_1}(p)=1\) and

\[
 Q_{\mathrm{raw},\{1,7\}}(p)
 =\begin{cases}1,&0<p\leq 1/2,\\7,&1/2<p<1.\end{cases}
\]

In subgroup B, set \(Q_{\mathrm{raw}}^{(1)}=Q_{\mathrm{raw}}^{(0)}\) for every region.

The probability-one-half mixture is a mixture over the latent region-level distribution-valued potential outcome. It is not a mixture of household incomes inside one region.

## Mean quantile effects

For subgroup A, the analysis-scale treatment mean at every \(p\) is

\[
 E\{Q^{(1)}(p)\mid X=A\}
 =\begin{cases}
 \log 2,&p\leq1/2,\\
 \{\log 2+\log 8\}/2=\log 4,&p>1/2,
 \end{cases}
 =Q^{(0)}(p).
\]

The upper-coordinate identity is exact because

\[
 (1+1)(1+7)=(1+3)^2.
\]

Therefore \(\tau_Q(A,p)=0\) for all \(p\). In subgroup B the two potential analysis quantile functions are equal, so \(\tau_Q(B,p)=0\) as well. The same conclusion holds on every finite curve grid, including the frozen \(K=49\) grid.

## Gini effects

For a nonnegative distribution with positive mean, use

\[
 G=\frac{E|Y-Y'|}{2E[Y]},
\]

where \(Y,Y'\) are iid copies.

The control distribution has mean \(2\), and

\[
 E|Y-Y'|=1,
 \qquad
 G(Q_{\mathrm{raw}}^{(0)})=\frac{1}{2\cdot2}=\frac14.
\]

The two treatment distributions in subgroup A have

\[
 G(\delta_1)=0,
 \qquad
 G(Q_{\mathrm{raw},\{1,7\}})=\frac{E|Y-Y'|}{2E[Y]}
 =\frac{3}{2\cdot4}=\frac38.
\]

Hence the treatment-arm mean of the unit-level Gini values is

\[
 E\{G(Q_{\mathrm{raw}}^{(1)})\mid X=A\}
 =\frac12\left(0+\frac38\right)=\frac3{16},
\]

and the subgroup-A Gini effect is

\[
 \delta_G(A)=\frac3{16}-\frac14=-\frac1{16}.
\]

In subgroup B, \(Q_{\mathrm{raw}}^{(1)}=Q_{\mathrm{raw}}^{(0)}\) region by region, so \(\delta_G(B)=0\).

## What the counterexample proves

The two subgroups have identical mean quantile-effect curves, both identically zero, but different nonlinear functional effects:

\[
 \tau_Q(A,\cdot)=\tau_Q(B,\cdot)=0,
 \qquad
 \delta_G(A)=-\frac1{16}\ne0=\delta_G(B).
\]

Therefore the mean quantile curve does not determine the conditional average effect of a nonlinear inequality functional. A curve-only forest can be blind to this heterogeneity at the population split criterion, while a composite target that includes the Gini coordinate has a population signal.

The correct target is the pre-averaging functional effect

\[
 E\{G(Q_{\mathrm{raw}}^{(1)})-G(Q_{\mathrm{raw}}^{(0)})\mid X\},
\]

not the Gini of an arm-specific mean quantile curve.

## Collapse sanity check

If each raw-income quantile \(Q_{\mathrm{raw}}^{(z)}\) is deterministic conditional on \(X\), then

\[
 E\{T(Q_{\mathrm{raw}}^{(z)})\mid X\}
 =T(Q_{\mathrm{raw}}^{(z)})
 =T\left(E\{Q_{\mathrm{raw}}^{(z)}\mid X\}\right).
\]

The counterexample requires conditional randomness in the treatment distribution-valued potential outcome in subgroup A. It disappears in the deterministic conditional limit.

## Checks run

The values \(0\), \(1/4\), \(3/8\), \(3/16\), and \(-1/16\) were computed with `fractions.Fraction` on the fixed midpoint functional grid. The raw-to-log identity was checked both by the exact integer equality \((1+1)(1+7)=(1+3)^2\) and by evaluating `log1p` on the frozen curve grid. The maximum absolute curve-effect error is below \(10^{-12}\).

## Observed failures

None. The curve-effect error is below the \(10^{-12}\) tolerance, and the Gini calculations are exact.

## Unresolved questions

The example establishes nonidentification from the mean quantile effect, not a lower bound on the predictive advantage of composite splitting. That empirical question remains in WP3 and WP9.

`PASS`
