# Foundational identification status

Date: 2026-07-27  
Scope: WP0 and WP1 observation model, before simulation  
Verdict: `CONDITIONAL PASS`

## Two different statistical experiments

The actual observed data for outer unit \(i\) are

\[
O_i=(X_i,Z_i,\mathcal S_i),
\]

where \(\mathcal S_i\) is an inner household sample of size \(m_i\). The oracle data are

\[
O_i^\star=(X_i,Z_i,Q_i),
\]

where \(Q_i\) is the latent transformed quantile function. Causal exchangeability and overlap identify the latent target from the law of \(O_i^\star\). They do not, by themselves, identify it from the law of \(O_i\).

## Bounded-inner-sample nonidentification witness

Take no covariates, randomized treatment, and \(m_i=1\). Keep the control arm identical across models. In the treated arm, compare:

1. Model A: \(\mu_i=\delta_1\) or \(\delta_3\), each with probability \(1/2\).
2. Model B: \(\mu_i=(\delta_1+\delta_3)/2\) deterministically.

Both models generate one observed inner draw equal to \(1\) or \(3\), each with probability \(1/2\). Their observed laws are therefore identical. Under the generalized-inverse convention, the raw-scale mean quantiles satisfy

\[
E\{Q_{A,\mathrm{raw}}(p)\}=2,
\qquad
E\{Q_{B,\mathrm{raw}}(p)\}
=
\begin{cases}
1,&p\le 1/2,\\
3,&p>1/2.
\end{cases}
\]

On the project’s log1p analysis scale,

\[
E\{Q_A(p)\}
=
\tfrac12(\log 2+\log 4),
\qquad
E\{Q_B(p)\}
=
\begin{cases}
\log 2,&p\le 1/2,\\
\log 4,&p>1/2.
\end{cases}
\]

The mean unit-level Gini is \(0\) in Model A and \(1/4\) in Model B. Thus the latent mean quantile and nonlinear-functional targets are not constant on observational equivalence classes when the inner sample is bounded and the latent mixing law is unrestricted.

## Binding repair

Every feasible claim about the latent target must use one of:

1. a triangular-array regime with \(\min_i m_i\to\infty\), together with uniform bias and \(L^2\) error rates strong enough for the claimed forest or inference result, or
2. an identified measurement, replicate, or validation model that recovers the relevant latent law.

A generic two-stage correction, precision weighting, cross-fitting, or bootstrap does not repair the witness above. If neither recovery regime is used, the feasible object is the empirical-proxy target based on \(U_K(\widehat Q_i)\), which must be named and interpreted separately.

## Oracle causal identification

Assume A1–A7, including consistency, no hidden treatment versions, no interference, conditional exchangeability given the full \(X\), overlap, and the stated coordinate moments and domains. Then, from the oracle law,

\[
M_z(x)
=
E\{U_K(Q_i^{(z)})\mid X_i=x\}
=
E\{U_K(Q_i)\mid Z_i=z,X_i=x\}
\]

for \(P_X\)-almost every \(x\). Low-dimensional modifier targets are identified for \(P_V\)-almost every \(v\), and subgroup targets require positive subgroup probability. Pointwise values outside the support or at a selected continuous version require additional smoothness and support assumptions.

## Simulation requirement

Every simulation row and report must identify the path as `oracle_latent`, `feasible_growing_inner`, `identified_measurement_model`, or `empirical_proxy`. Truth comparisons for the latent target are valid for the feasible estimator only in the first three regimes. The \(m_i=1\) case collapses to ordinary scalar HTE only for the empirical point-mass proxy or under an explicit assumption that the latent distributions are themselves degenerate.

## Remaining theory obligation

WP5 must verify the observation-recovery conditions for the chosen empirical-quantile or survey estimator. Lin, Kong, and Wang provide a source template, but their main estimated-distribution results use bounded-support and balanced/growing inner-sample conditions. They do not automatically cover unequal inner sizes, unbounded income tails, or the Gini, Theil, and Atkinson coordinates used here.
