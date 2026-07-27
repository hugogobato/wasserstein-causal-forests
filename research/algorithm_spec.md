# WP3-T1: finite-grid ODCF algorithm specification

**Input files:** `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md` Sections 2.3, 2.6, 2.7, and WP3; `research/finite_grid_estimands.md`; `research/notation.md`; `research/assumption_ledger.md`.

**Source to adapt:** Athey, Tibshirani, and Wager (2019), Sections 2–3, for forest localization and local moment splitting; Oprescu, Syrgkanis, and Wu (2019), Sections 2–3, for nuisance-orthogonal forest scores; Nie and Wager (2021), Sections 2–3, for the residualized CATE formulation; and Ćevid et al. (2022), Section 3, for MMD split inspiration. The local MMD comparator acts on causal score vectors and is therefore not target-agnostic in the official DRF sense. The prototype uses only finite-vector algebra and does not import a rate or limit theorem from these sources.

**Assumptions used:** A1–A4 for the causal score, A8 for the optional inner-sample interface, A9 for cross-fitting, A11 for training-only split scaling, and A12 for fixed `K` and `J`. A10 is a target for later forest theory, not a property discharged by this finite prototype.

**Status:** `PARTIAL`, finite-grid pre-simulation prototype. The score algebra and finite candidate-gain identity are available. Full raw-outcome honesty, nuisance oracle equivalence, asymptotic regularity, and inner-sample validity are not established here.

## 1. Frozen defaults

The implementation uses the `log1p` analysis quantile on the fixed interval \([0.05,0.95]\), `K=49` trapezoid-weighted curve coordinates, and `J=3` raw-income functional coordinates, namely Gini, Theil T, and Atkinson(0.5), as defined in `research/finite_grid_estimands.md`. Trapezoidal weights have total mass \(0.90\), with half-weighted endpoints. They must be passed through the same frozen constructor in fitting, scaling calibration, projection, and tests.

The split-only functional-coordinate rule is `robust_sd`,
\[
s_j=\frac{d_Q}{1.4826\operatorname{MAD}(R_{\cdot,K+j})}.
\]
It is selected analytically because it is Gaussian-consistent, robust to outliers, and does not introduce the explicit \(\sqrt n\) divergence of `null_score_se`. The calibration in `research/wp3_scaling.py` is an illustrative diagnostic, not a proof or the source of this choice.

The reproducible forest defaults are `n_trees=200`, `subsample_fraction=0.70`, `honesty_fraction=0.50`, `min_leaf=5`, `min_child_fraction=0.10`, `max_depth=8`, at most 32 candidate thresholds per feature, `mtry=floor(sqrt(d))` unless supplied, `min_gain=1e-12`, and `random_state=20260727`. Both split and populate children must contain at least `min_leaf` observations and at least ten percent of their parent sample. Empty populate leaves and global-populate fallbacks are forbidden.

The feasible nuisance defaults are five region-level stratified folds; a 100-tree random-forest classifier for an unknown propensity; separate 100-tree random-forest regressors for the two arm means; `n_jobs=1`; and
`min_samples_leaf=max(2,min(10,n_train//20))`. Fold \(f\) uses seed `random_state + f` for the classifier, one plus that seed for the control regression, and two plus that seed for the treated regression. Estimated propensities are clipped to \([0.02,0.98]\). A supplied known propensity bypasses classifier fitting but not cross-fitted arm regressions. Every nuisance-training fold must contain both treatment arms, otherwise fitting fails explicitly. If scikit-learn is unavailable, the explicit fallback uses the fold treatment fraction and arm-specific fold means; other fitting errors are not silently swallowed.

The inner path uses empirical `inverted_cdf` quantiles, the frozen `L=400` functional grid, and 100 bootstrap replicates of the original within-region sample size. Region \(i\) uses seed `random_state + i`. The DRF-inspired MMD-on-score comparator uses a Gaussian kernel, direct-sum metric weights, and the median positive pairwise distance from at most 128 split observations as bandwidth, truncated below at \(10^{-8}\). These are prototype defaults, not recommendations inherited from the official DRF implementation.

## 2. Inputs and outputs

The primary interface takes an outer-unit covariate matrix \(X\in\mathbb R^{n\times d}\) and an **unscaled scientific score matrix** \(R\in\mathbb R^{n\times(K+J)}\). In the oracle path, \(R_i=\phi(O_i;\eta_0)\). In the feasible path, \(R_i\) is constructed from the unscaled finite vector \(\widehat U_K(\widehat Q_i)\) and region-level cross-fitted \((\widehat e,\widehat m_0,\widehat m_1)\).

`ODCFEstimator.fit(X, scores, treatment=None, propensity=None, noise_variances=None)` returns a fitted model. Its `predict(X_new)` method returns the unscaled finite target-vector prediction, `weights_at(x)` returns normalized forest weights over regions, and `honesty_report()` verifies finite index separation and populate-leaf constraints. The report does not certify that nuisance training or global preprocessing is independent of all leaf-estimation outcomes.

The common prediction interface is also used by `SpecializedForest`, which combines separate target-specific forests. The `mmd_score` variant is a Gaussian-kernel MMD forest on cross-fitted score vectors. It is DRF-inspired, not an implementation of official DRF or Causal-DRF. An official Causal-DRF implementation remains a mandatory WP9 baseline.

## 3. Finite-vector construction

For each region, compute the unscaled scientific vector before any conditional mean is taken:

\[
U_{K,i}=\left(Q_i(p_1),\ldots,Q_i(p_K),T_1(Q_i),T_2(Q_i),T_3(Q_i)\right).
\]

The functional coordinates use the full-probability midpoint grid fixed by WP2. The observed finite score is

\[
\widehat\phi_i=\widehat m_1(X_i)-\widehat m_0(X_i)
 +\frac{Z_i}{\widehat e(X_i)}\{\widehat U_i-\widehat m_1(X_i)\}
 -\frac{1-Z_i}{1-\widehat e(X_i)}\{\widehat U_i-\widehat m_0(X_i)\}.
\]

Each validation score is generated by nuisance models trained on complementary regions. Folds are stratified by treatment and checked for arm support. If treatment is randomized with known design propensity, `known_propensity=e_design` bypasses propensity fitting while retaining outcome-regression cross-fitting; \(e_{\mathrm{design}}\) need not equal \(1/2\). This is nuisance-fold separation only; it does not itself establish full forest honesty after the scores are reused by every tree.

## 4. Finite score-honest composite tree

Let \(D_s=\operatorname{diag}(1,\ldots,1,s_1,\ldots,s_J)\) be the split-only scaling transformation. The scientific scores \(r_i\) remain unscaled for leaf averaging and returned predictions. For split selection, define the positive direct-sum weights

\[
\omega=(w_1,\ldots,w_K,1,\ldots,1),
\]

where the \(w_k\) are the frozen trapezoidal weights. Equivalently, the split criterion is evaluated on \(D_s r_i\). This separation prevents training-only scale estimates from becoming part of the causal estimand.

Each tree draws a subsample without replacement and partitions it into a split sample and a populate sample. Conditional on the fixed cross-fitted score matrix and fixed scales, a node is split using only split-sample scores. For a considered candidate partition \(A=A_L\mathbin{\dot\cup}A_R\), the empirical gain is

\[
\mathcal G(A_L,A_R;A)=
\sum_{i\in A}\|D_s(r_i-\bar r_A)\|_\omega^2
 -\sum_{i\in A_L}\|D_s(r_i-\bar r_{A_L})\|_\omega^2
 -\sum_{i\in A_R}\|D_s(r_i-\bar r_{A_R})\|_\omega^2.
\]

The equivalent two-child expression is

\[
\mathcal G(A_L,A_R;A)=
\frac{|A_L||A_R|}{|A|}
\sum_{c=1}^{K+J}\omega_c s_c^2
\left(\bar r_{A_L,c}-\bar r_{A_R,c}\right)^2.
\]

Here \(s_c=1\) for a curve coordinate. This is C3.1's exact scope: a **considered finite candidate partition** has positive empirical gain if at least one active scaled child mean differs. It does not prove that the greedy algorithm considers or selects the scientifically best partition, and it is not a population forest theorem.

The split is accepted only when both split and populate children satisfy `min_leaf`, both sides satisfy `min_child_fraction=0.10`, and the gain exceeds `1e-12`. Populate outcomes do not enter the gain. A leaf prediction averages the unscaled scientific scores of its routed populate observations. There is no full-populate fallback.

The forest prediction averages the tree leaf predictions. The equivalent forest-weight representation is

\[
\widehat\Theta_K(x)=\sum_{i=1}^n\alpha_i(x)\widehat\phi_i,
\qquad \alpha_i(x)\geq0,\qquad \sum_i\alpha_i(x)=1.
\]

## 5. Variants exposed by the prototype

`curve_only` activates coordinates \(1,\ldots,K\), providing ODCF-v0. `composite` activates all \(K+J\) coordinates and is ODCF-v1. `fit_specialized_forests` fits separate forests for the curve and each scalar functional, using the frozen default seed for its first component and consecutive seeds thereafter unless a base seed is supplied. `mmd_score` activates all coordinates but ranks candidate partitions with a direct-sum-weighted Gaussian-kernel child MMD score. It should be reported as “MMD-on-cross-fitted-score forest (DRF-inspired).” It is not a substitute for the official Causal-DRF benchmark.

`fit_odcf_from_inner_samples` is the explicitly provisional WP3-T10 path. For each region, it computes \(\widehat U_i\), resamples that region's households, applies the finite-vector map to each bootstrap sample, and uses \(2\widehat U_i-\overline U_i^{\mathrm{boot}}\) as a bootstrap bias-corrected vector. A known randomized-design propensity can be passed through this helper and bypasses classifier fitting. When explicitly enabled for an SSE-based variant, the direct observation-level bootstrap variance is propagated to the DR-score scale through the squared coefficient
\[
a_i^2=\left\{\frac{Z_i}{\widehat e_i}+\frac{1-Z_i}{1-\widehat e_i}\right\}^2
\]
before the diagonal plug-in noise-gain heuristic is subtracted. The option is off by default and requires supplied score-scale variances. It is not defined for `mmd_score`, and that combination is rejected instead of silently doing nothing. This heuristic does not account for nuisance-training propagation, cross-region covariance, the sampling variance of the bias-corrected estimator, or survey designs. It is not a correction theorem and remains subject to WP5.

## 6. Arm curves and projection

`arm_dr_scores` constructs cross-fitted AIPW pseudo-outcomes for \(M_0\) and \(M_1\). `fit_arm_curve_forests` fits a separate finite score-honest curve forest for each arm, and `ArmCurveForest.predict_arms(..., project=True)` applies weighted PAVA to each predicted arm curve under the declared trapezoidal geometry. `predict_effect` subtracts those arm estimates and never projects the resulting effect curve, because a difference of quantile functions need not be monotone. This is a finite diagnostic interface, not an arm-forest consistency or inference theorem.

## 7. Collapse checks and guardrails

When `J=0`, the estimator is a multi-output quantile-curve causal forest. When `K=1,J=0`, the score and split objective are scalar. With randomized treatment and any known design propensity in \((0,1)\), classifier fitting is bypassed. Cross-fitting and finite score honesty are separate: folds protect each score evaluation from its own nuisance-training data, while tree-level index separation keeps split scores out of that tree's leaf average. Neither statement proves independence between every populate outcome and the fitted nuisance functions used by split observations.

The main implementation guardrail is that a target coordinate can enter a split only through its declared finite score and positive norm weight. No nonlinear functional is applied to an outer mean curve. No projection is applied to an effect curve. No tree-level separation is inferred from cross-fitting.

## 8. Checks run

`research/wp3_invariants.py` checks the finite candidate-gain identity, the pure-functional score construction, explicit trapezoidal weights, grid-duplication invariance, treatment-stratified fold support, the known-propensity classifier bypass, split/populate index separation, nonempty balanced populate leaves, weighted PAVA, the unconstrained effect curve, common prediction interfaces, and the experimental bootstrap interface. These are finite implementation checks conditional on the generated score matrix.

Observed finite implementation gate: `PASS`. Overall WP3 status remains `PARTIAL` because the inner path is experimental and official Causal-DRF remains a WP9 baseline rather than a local implementation. No consistency, rate, asymptotic normality, full raw-outcome honesty, bootstrap validity, or inner-sample validity claim is made by this file.
