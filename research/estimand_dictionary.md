# WP2-T8: estimand dictionary

**Input files:** `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md` Sections 2, 4, and 5; `research/finite_grid_estimands.md`; `research/simulation_results_schema.md`; `research/application_design_memo.md`.

**Source to adapt:** None. This is a project bookkeeping artifact based on the finite-grid target in WP2-T0.

**Assumptions used:** A2–A7 for causal target interpretation; A11 only for split scaling; A12 whenever a finite-dimensional claim is made.

**Status:** `PASS` for the pre-simulation dictionary.

## Target dictionary

| Planned object, metric, or plot | Exact target or estimand | Coordinates and weighting | Interpretation boundary |
|---|---|---|---|
| Quantile-effect curve | \(\tau_{Q,K}(x,p_k)=E[Q^{(1)}(p_k)-Q^{(0)}(p_k)\mid X=x]\) | First \(K=49\) coordinates; curve loss uses \(w_k\) | Mean quantile effect across regions, not the quantile treatment effect of scalar regional incomes and not an individual percentile effect |
| Gini effect | \(\delta_G(x)=E[G_L\{y(Q^{(1)})\}-G_L\{y(Q^{(0)})\}\mid X=x]\) | Unscaled coordinate \(K+1\); split loss uses weight \(s_1^2\) | Gini is computed per latent raw-income distribution before averaging; it is not the Gini of the arm mean quantile curve |
| Theil T effect | \(\delta_{\mathrm{Theil}}(x)=E[\mathrm{Theil}_L\{y(Q^{(1)})\}-\mathrm{Theil}_L\{y(Q^{(0)})\}\mid X=x]\) | Unscaled coordinate \(K+2\); split loss uses weight \(s_2^2\) | The raw-income functional uses the fixed \(L=400\) grid and the positive-mean domain condition |
| Atkinson effect | \(\delta_{\mathrm{Atkinson}}(x)=E[\mathrm{Atkinson}_{L,0.5}\{y(Q^{(1)})\}-\mathrm{Atkinson}_{L,0.5}\{y(Q^{(0)})\}\mid X=x]\) | Unscaled coordinate \(K+3\); split loss uses weight \(s_3^2\) | Aversion parameter \(0.5\) is fixed before outcome-effect comparisons |
| Full finite target vector | \(\Theta_K(x)=M_1^K(x)-M_0^K(x)\) | Direct sum of the curve and three scalar coordinates | Exploratory unless evaluated at a frozen subgroup or a selected low-dimensional modifier |
| Frozen subgroup effect | \(\Theta_g=E[\Theta_K(X)\mid X\in A_g]\) | Same \(K+3\) coordinates | The subgroup \(A_g\) must be frozen independently of confirmation outcomes |
| Low-dimensional modifier effect | \(\Theta_V(v)=E[\Theta_K(X)\mid V=v]\) | Same coordinates, outer expectation over \(X\mid V=v\) | Conditioning or adjustment only on \(V\) is not a substitute for the full-X causal adjustment used to identify \(\Theta_K(X)\) |
| Integrated squared curve error | \(\operatorname{ISE}_{Q,r}\) defined below on the frozen weighted evaluation set | Quadrature-weighted first \(K\) coordinates, then \(\nu_b\)-weighted over evaluation points | A simulation loss, not a new causal estimand |
| Functional RMSE | \(\operatorname{RMSE}_{T_j}\) defined below over replicates and the frozen weighted evaluation set | One unscaled scalar coordinate at a time | Split-only scales are not used in evaluation |
| Worst-coordinate standardized error | \(W_r\) defined below using frozen positive standardizers \(a_c\) | Maximum of evaluation-set RMSE over all unscaled coordinates | Diagnostic for imbalance across targets, not a causal target |
| Pointwise coverage | \(P\{\Theta_{K,k}(x)\in C_k(x)\}\) for a fixed coordinate or subgroup | One declared coordinate at a time | Exploratory local coverage unless the forest/inference branch supplies a theorem |
| Simultaneous coverage | Probability of the frozen finite-family intersection event defined below | Declared pairs of evaluation points and finite target coordinates | No continuum, changing family, or arbitrary post-hoc functional coverage claim |
| Arm quantile-curve monotonicity | Discrete validity of \(\widehat Q_z(x,p_1)\leq\cdots\leq\widehat Q_z(x,p_K)\) | Arm-specific estimates only | Projection may be applied to arm curves, never to the effect curve \(\widehat\tau_{Q,K}\) |
| Projection distance | \(\|\Pi_{\mathcal Q_K}(\widehat Q_z)-\widehat Q_z\|_{K,w}\) | Arm curve diagnostic | Measures an algorithmic correction, not a treatment-effect estimand |
| Composite split signal | Difference in child-node means of a coordinate of the score \(\widehat\phi\) | Coordinatewise forest-training diagnostic | It is not evidence of causal heterogeneity without the target and identification assumptions |

## Frozen simulation-evaluation convention

Before fitting any competing method, each DGP must store a deterministic evaluation set

\[
 \mathcal X_{\mathrm{eval}}=\{x_1,\ldots,x_B\},
 \qquad
 \nu_1,\ldots,\nu_B\geq0,
 \qquad
 \sum_{b=1}^B\nu_b=1.
\]

The set is either an analytic grid declared by the DGP or one fixed draw from its evaluation distribution using a truth-only seed. The same set and weights are used for every method, sample size, and replicate within that DGP. They may not be regenerated after comparative results are inspected.

For each unscaled coordinate \(c=1,\ldots,K+J\), also store a positive standardizer \(a_c\). The default is the pooled oracle standard deviation of the unit-level potential-outcome coordinate \(U_{K,c}(Q^{(0)})\) and \(U_{K,c}(Q^{(1)})\) under the DGP evaluation law, computed analytically when possible or from a fixed independent truth sample. A coordinate with zero oracle dispersion uses a DGP-declared positive scientific reference scale. Neither \(a_c\) nor its fallback may depend on a fitted method or simulation result, and the split-only \(s_j\) are never used as evaluation standardizers.

With replicate index \(r\), the integrated curve error is

\[
 \operatorname{ISE}_{Q,r}
 =\sum_{b=1}^B\nu_b\sum_{k=1}^K
 w_k\{\widehat\tau_{Q,r}(x_b,p_k)-\tau_Q(x_b,p_k)\}^2.
\]

The functional RMSE for coordinate \(j\) is

\[
 \operatorname{RMSE}_{T_j}
 =\left[
 \frac1R\sum_{r=1}^R\sum_{b=1}^B\nu_b
 \{\widehat\delta_{T_j,r}(x_b)-\delta_{T_j}(x_b)\}^2
 \right]^{1/2}.
\]

The replicate-level worst-coordinate standardized error is

\[
 W_r
 =\max_{1\leq c\leq K+J}
 \frac{
 \left[\sum_{b=1}^B\nu_b
 \{\widehat\Theta_{K,r,c}(x_b)-\Theta_{K,c}(x_b)\}^2\right]^{1/2}
 }{a_c},
\]

and simulation summaries report the empirical mean and quantiles of \(W_r\).

For a simultaneously reported family \(\mathcal I\subseteq\{1,\ldots,B\}\times\{1,\ldots,K+J\}\), frozen before fitting, simultaneous coverage is the probability of the explicit intersection event

\[
 P\left\{
 \bigcap_{(b,c)\in\mathcal I}
 [\,\Theta_{K,c}(x_b)\in C_c(x_b)\,]
 \right\}.
\]

Pointwise coverage uses one declared pair \((b,c)\). A subgroup analysis treats each frozen subgroup as an evaluation point. These conventions make the metric rows above operational and prevent post-result changes to the evaluation distribution, standardization, or coverage family.

Each DGP manifest must persist the evaluation points, weights, truth-only seed, standardizers, fallback reference scales, and simultaneous-coverage family. Simulation output must carry the corresponding immutable manifest identifier.

## Application plot mapping

The planned application plots map as follows. The quantile-effect plot reports \(\widehat\tau_{Q,K}\) on the frozen \([0.05,0.95]\) grid. The three inequality-effect plots report the unscaled estimates of \(\widehat\delta_G\), \(\widehat\delta_{\mathrm{Theil}}\), and \(\widehat\delta_{\mathrm{Atkinson}}\). Arm-specific distribution plots report \(\widehat M_z^Q\) or its projected arm curve, while an effect plot reports the difference of arm estimates and is not projected. Subgroup or modifier panels report the corresponding coordinates of \(\widehat\Theta_g\) or \(\widehat\Theta_V(v)\), with the subgroup or modifier declared before confirmation outcomes are used.

## Target-ordering rules

First, calculate each \(T_j\) on each unit's estimated or latent distribution. Second, form conditional means or treatment contrasts. Third, apply forest localization and any confirmatory aggregation. The reverse order, applying \(T_j\) to an arm mean quantile curve and calling it \(\delta_{T_j}\), is outside the target dictionary.

## Checks run

Every planned primary simulation metric in WP0-D006 has an exact row above. The finite-vector and collapse conventions agree with `research/finite_grid_estimands.md` and `research/notation.md`.

## Observed failures

None.

## Unresolved questions

The confirmation target is not yet selected between frozen subgroups and a low-dimensional modifier. That choice belongs to the application pilot and WP2-T3.

`PASS`
