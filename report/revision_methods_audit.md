# Methods audit for the WCF revision

## Scope and verdict

This is a specialist audit of the current method description in
[`report/wcf_main.tex`](./wcf_main.tex), the implementation in
`src/wasserstein_causal_forests/cwdb/{model.py,arm_shared_tree.py,dr_calibration.py,cross_fitted.py}`,
and the adapters and metric layer in
`src/wasserstein_causal_forests/g3/{phase6_methods.py,phase65_methods.py,evaluation.py}`.
The frozen estimand contract in `research/estimand_contract.md` was also read.
The audit does not edit the manuscript. The proposed replacement is in
[`revision_methods_section.tex`](./revision_methods_section.tex).

Overall status: **material method-description corrections required; implementation is coherent after scope is narrowed**. The current code supplies a particle estimate of each arm-specific finite-grid conditional law and an AIPW calibration for a finite set of scalar functional contrasts. It does not supply a doubly robust estimator of the full conditional law. The main mathematical identities used by the proposed replacement are valid, subject to the overlap and nuisance-estimation conditions stated there.

The user explicitly requested no reviewer subagents for this specialist pass. Consequently, the independent blind-review and adjudication stages prescribed by the general mathematical-audit workflow were not run. This artifact is therefore a code-to-manuscript audit, not a full independent theory audit.

## Source reconstruction

The current Section 2 fixes `X` to `[-1,1]^6`, `K=25`, and `M=10` in the setup, defines a particle law and energy score, describes a shared tree, gives an AIPW score, and adds a two-part law. The implementation makes the following final-path choices:

| Object | Code path | Implemented behavior |
|---|---|---|
| Arm law | `CWDBRegressor` with `architecture="v1"`, `sharing="partial"` | One shared partition, arm-specific gradient updates, projection to the monotone cone |
| Final Track B WCF-DR | `g3/phase6_methods.py:48-143` | Pooled initialization, contrast ridge, two-fold held-out selection over `(0,50,500)`, 10 particles in the current manifest |
| Particle loss | `cwdb/energy.py` | Collision-smoothed attraction minus pairwise repulsion |
| Scalar calibration | `cwdb/dr_calibration.py:103-227` | Cross-fitted logistic propensity with `[.02,.98]` clipping and AIPW scores for five declared scalar functionals |
| Reported calibrated contrasts | `g3/phase6_methods.py:111-131` | Four functional contrasts and the reference contrast are replaced by binwise AIPW contrasts; the law and law-level metrics remain particle-law outputs |
| Two-part law | `g3/phase65_methods.py:323-432` | Per-arm penalized logistic classifier for an all-zero quantile vector, positive-row WCF fit, explicit mixture mass |
| TATE and TCATE metrics | `g3/evaluation.py:269-337` | `TATE-K-*`, `TCATE-K-*`, `REF-ATE-K`, and `REF-TCATE-K` are already emitted by the evaluation layer |

## Adoption, propensity, and identification

The propensity score is the function (e(x)=\Pr(A=1\mid X=x)). Endogenous adoption is a property of the joint assignment and outcome process, not a synonym for the propensity score. A nonconstant propensity means treatment probabilities differ across covariate values. It becomes observed confounding when those covariates also predict (Y^0), (Y^1), or a target functional of them. A constant value such as (e(x)=0.7) is unequal allocation but can still be randomized. If adoption depends on unobserved potential outcomes after conditioning on (X), conditional exchangeability fails, and neither the particle law nor the AIPW layer repairs that failure.

The income-track DGPs use the observed-confounding case. Their logistic assignment index is a function of covariates that also enter the location, scale, and shape surfaces. It is therefore accurate to say that the simulation studies examine nonconstant assignment related to outcome-relevant covariates. It is too broad to say that the method works whenever the propensity is not (0.5), and it is also too narrow to define endogeneity only as the literal functional relationship (e(X)=f(\mu(X))). The relevant condition is shared prognostic information in (X), together with positivity.

The law identification statement should be written with consistency, (Y^a\perp A\mid X), and (0<e(X)<1). The finite-grid target is (operatorname{Law}{q_K(Y^a)\mid X=x}), not automatically the continuum outcome law. This distinction is already respected by `research/estimand_contract.md` and should be brought into Section 2.

## Claim and dependency matrix

| ID | Current claim or object | Code check | Status |
|---|---|---|---|
| M1 | Fixed-grid Wasserstein geometry and particle empirical law | `LawPrediction.from_particles`, `validate_weights`, `project_quantiles` | No defect found within finite-grid scope |
| M2 | Shared partition and contrast shrinkage | `ArmSharedTreeRegressor._leaf_values` | Current manuscript formula is false for unbalanced leaves; corrected below |
| M3 | AIPW score is doubly robust | `aipw_scores`, exact algebraic identity | Correct for scalar means under standard nuisance and overlap conditions; current exposition overstates scope and efficiency |
| M4 | Cross-fitted propensity is logistic and clipped | `FunctionalAIPW.fit` | Correct implementation description; logistic is a working choice, not a method requirement |
| M5 | WCF-DR is doubly robust for the conditional law | `DRAdapter` and `DRCalibratedCWDB` | False if read this way. DR applies only to declared scalar functional contrasts |
| M6 | TATE and TCATE are evaluated | `evaluation.py:269-337` | Code supports both, but current manuscript foregrounds only moderator contrasts and does not report the full requested dictionary |
| M7 | Reference effect is a modified TATE/TCATE | `h_ref`, `reference_contrast` | Correct as a finite-grid expected-distance contrast; it is not distance of the arm mean |
| M8 | Two-part extension for zero-inflated outcomes | `ZIPTAdapter` | Correct for a unit-level atom at the zero quantile vector; current “structurally blind” wording is too absolute |
| M9 | `d`, `K`, and `M` are method restrictions | `DGPSpec.n_features`, `GridSpec.n_grid`, manifest fields | False. They are general inputs and tuning/design parameters; only the current experiment fixes them |
| M10 | Existing DGPs cover nonlinear behavior | `phase6_dgps.py`, `phase65_dgps.py`, earlier `dgps.py` | Outcome quantiles are nonlinear through exponentiation and Hermite shape, and earlier regimes include sine/square surfaces. The income-track covariate surfaces and propensity index are mostly additive linear; no nonlinear propensity experiment is currently present |
| M11 | Causal-DRF is a doubly robust competitor | Causal-DRF source and adapter | Unsupported. The cited method is a causal distributional forest with a CKTE target; the paper should simply state the scope difference and avoid a first-known claim unless a systematic review is added |
| M12 | WCF-DR is fully honest cross-fitting | `DRCalibratedCWDB.fit` | Qualification required: the OOF arm means are generated after selecting λ with the same fold outcomes, so strict nested sample splitting is not implemented |

## Material findings

### M2. The published leaf formula is wrong on unbalanced leaves

The manuscript currently states

\[
\widehat g_1=\bar g+\tfrac c2\delta,\qquad
\widehat g_0=\bar g-\tfrac c2\delta.
\]

That preserves the unweighted average only when (n_0=n_1). The ridge branch in
`arm_shared_tree.py:215-252,254-285` uses (s=n_1/(n_0+n_1)) and returns

\[
\widehat g_0=\bar g-s c\delta,
\qquad
\widehat g_1=\bar g+(1-s)c\delta,
\]

which preserves ((n_0\widehat g_0+n_1\widehat g_1)/(n_0+n_1)=\bar g). The exact rational check in `report/check_revision_identities.py` gives a pooled value (9/5) for (n_0=8,n_1=2,g_0=1,g_1=5,lambda=3), while the manuscript's symmetric formula gives (159/115). This is a substantive implementation-description error, not a cosmetic notation issue. The corrected formula is included in the proposed section.

### M3 and M5. The doubly robust claim must be restricted to scalar functional contrasts

For (H=h(q_K(Y))), nuisance limits (m_a(x)), and a propensity limit (g(x)), the population AIPW score used by the code has conditional bias

\[
\{g(X)-e(X)\}
\left[\frac{m_1(X)-\mu_{1,h}(X)}{g(X)}
      +\frac{m_0(X)-\mu_{0,h}(X)}{1-g(X)}\right].
\]

This verifies the usual product-error identity: the scalar contrast is unbiased if the propensity is correct or if both arm-specific conditional means are correct, under positivity and integrability. This identity does not make (widehat P_{a,M}^K(x)) doubly robust. In code, the law is still the particle booster, and only the scalar columns for `grid_mean`, `grid_sd`, `grid_skewness`, `grid_upper_tail_mean`, and `reference` are overwritten (`phase6_methods.py:111-131`). The arm entries are set to `{contrast, 0}` as an output interface, so the meaningful object is their difference, not an independently calibrated arm-specific conditional mean.

The current phrase “efficient when both are” is unsupported by this implementation and by the manuscript's supplied theory. Efficiency requires a specified semiparametric target, the efficient influence function, regularity, and an inference argument. The revised wording should say “standard doubly robust score for scalar functional means under the usual conditions” and explicitly state that no theorem for the finite-particle law is claimed.

### M4 and M12. Logistic propensity, clipping, and sample splitting

`FunctionalAIPW.fit` fits a five-fold cross-fitted `sklearn.linear_model.LogisticRegression` and clips predictions to `[0.02,0.98]`. This is a simple working model for the simulation, not a structural requirement. A flexible binary classifier can replace it, provided it is cross-fitted and overlap is controlled.

The IC3 DGP clips its true propensity to `[0.01,0.99]`, so the fixed estimator clip excludes part of the true support. If the outcome nuisance is correct, the DR route can still be the relevant robustness route. If the outcome nuisance is misspecified and the estimator relies on propensity correctness, the clipped propensity is not the true propensity in those tails. The exact check in `report/check_revision_identities.py` constructs a true propensity (0.01), an estimate (0.02), and misspecified outcome means, producing (1/2) instead of the true effect (1). The paper should not call the clipped propensity exactly correctly specified in IC3.

There is a second qualification. `DRCalibratedCWDB.fit` first chooses the contrast strength using held-out energy risk on the same fold partition and then uses that selected strength to produce OOF arm means. Thus each OOF prediction is excluded from the fit of its tree model, but its outcome can influence the selected hyperparameter. This is not the strict nested cross-fitting used in a clean DML proof. The paper can describe the implementation as cross-fitted nuisance predictions with cross-validated tuning, but should not imply a fully nested sample-splitting theorem.

### M6 and M7. TATE, TCATE, and reference-distance contrasts

The evaluation layer already emits both marginal and moderator-stratified functional targets. For each declared (h_j), `tate_functional_rmse` compares the marginal contrast and `tcate_functional_rmse` compares the four moderator-bin contrasts. The reference columns likewise compare `REF-ATE-K` and `REF-TCATE-K`. The current manuscript's setup defines (\theta_j(v)) and (\theta_{\mathrm{ref}}(v)) but does not define the TATE counterparts or give the reference-distance estimands the requested contribution status.

The correct new contribution is the finite-grid reference-distance effect

\[
\Delta_{\star}^K=E\{d_{W,K}(q_K(Y^1),q_\star)
              -d_{W,K}(q_K(Y^0),q_\star)\},
\]

and its moderator-specific analogue

\[
\Delta_{\star}^K(v)=E\{d_{W,K}(q_K(Y^1),q_\star)
              -d_{W,K}(q_K(Y^0),q_\star)\mid V=v\}.
\]

Because expectation and distance do not commute, these are not
`d_W(E[q_K(Y^a)|X],q_star)` contrasts. The exact check records the simple witness (q=+1) or (-1) with equal probability and (q_\star=0): expected distance is (1), while distance of the mean is (0).

### M8. Scope of the two-part extension

`ZIPTAdapter` identifies a degenerate observation by `max(abs(q)) <= 1e-12`, fits a separate arm-specific logistic classifier on that indicator, and creates one explicit zero atom. Its classifier is not cross-fitted. The positive WCF fit uses the cross-validated contrast selector, but the mixture classifier itself is fit on all training rows.

The phrase “particle methods ignore degenerate components entirely” should be removed. A finite particle law can contain a zero particle and can represent zero mass in increments of (1/M), although it lacks a separate, smoothly estimated component probability. The defensible claim is that the explicit two-part construction represents a unit-level structural atom directly and avoids forcing that mass to be carried by the positive-part particle cloud. This extension is optional and should be motivated by structural-zero settings such as benefit take-up, insurance claims, expenditures, or regional activity with a genuine all-zero outcome law. It is not a claim that every scalar distribution with observations equal to zero requires a separate law component.

### M9 and M10. General dimensions, sensitivity, and nonlinear designs

The method code validates an arbitrary number of covariate columns and an arbitrary grid length (K\geq2); the particle count is a positive integer. The manuscript should introduce (d), (K), and (M) generically and state (d=6,K=25,M=10) only in implementation details. The sensitivity plan should vary (K) and (M) while holding the DGP, seeds, sample sizes, and evaluation targets fixed, and should report both law metrics and TATE, TCATE, REF-ATE, and REF-TCATE.

Nonlinearity is partly present already. The quantile law is nonlinear in covariates through the exponential scale and the Hermite shape transformation, and the earlier DGP family includes sine and squared surfaces. The income-track covariate surfaces and propensity index are predominantly additive linear before the logistic link. A new nonlinear-assignment DGP is therefore still useful: keep the same arm outcome surfaces and set the propensity logit to include interactions or smooth terms, for example (x_4x_5), (x_4^2), or \sin(\pi x_5)), with an explicit positivity clip. The propensity nuisance should then be compared under logistic and a flexible classifier. This isolates assignment-model misspecification from outcome-law geometry.

## Citation audit

The online checks used primary or publisher sources where available.

| Citation | Identity | Content and local applicability |
|---|---|---|
| Näf, Susmann, “Causal-DRF” | Verified at [arXiv:2411.08778](https://arxiv.org/abs/2411.08778) | The abstract describes a treatment-aware distributional forest for the conditional kernel treatment effect and consistency/asymptotic normality under its assumptions. It does not establish that Causal-DRF is doubly robust. The bibliography entry should be updated to match the accessible version, whose listed authors are Näf and Susmann, while the manuscript entry currently lists Näf, Park, and Susmann as a 2026 AISTATS item. |
| Distributional Random Forest | Verified at [JMLR 23 (2022)](https://jmlr.org/papers/v23/21-0585.html) | Supports the distributional-forest comparator and weighted empirical conditional-law representation. It does not support a claim about the WCF AIPW layer. |
| Meinshausen, Quantile Regression Forests | Verified at [JMLR 7 (2006)](https://jmlr.csail.mit.edu/papers/v7/meinshausen06a.html) | Supports the forest interpretation as a weighted conditional distribution/quantile estimator. |
| Chernozhukov et al., DML | Verified by publisher DOI [10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097) | Supports the general relationship between orthogonal scores, cross-fitting, and nuisance estimation. It does not automatically certify this implementation because hyperparameter selection is not nested and no rate conditions are proved here. |
| Kennedy et al., nonparametric doubly robust treatment effects | Verified by DOI [10.1111/rssb.12213](https://doi.org/10.1111/rssb.12213) | Supports the scalar doubly robust treatment-effect framework. Its results should be treated as an external leaf; local applicability still requires the finite-grid functional, overlap, nuisance, and cross-fitting conditions. |
| Cragg two-part model | Identity verified at DOI [10.2307/1909582](https://doi.org/10.2307/1909582) | Recommended citation for two-part limited-dependent-variable motivation; not currently in the manuscript bibliography. |
| Mullahy modified count models | Identity verified at DOI [10.1016/0304-4076(86)90002-3](https://doi.org/10.1016/0304-4076(86)90002-3) | Recommended citation for hurdle/zero-modified motivation; not currently in the manuscript bibliography. |
| Székely and Rizzo energy-distance citation | Identity/content not checked beyond the manuscript bibliography in this pass | The current citation is not enough to support the exact claim that the collision-smoothed fixed-(M) objective is strictly proper. Remove that claim or add a primary energy-score/propriety reference and state that the implemented objective is a restricted empirical-score optimization. |

## Exact and executable checks

`report/check_revision_identities.py` uses exact rational arithmetic and writes
`report/revision_identity_checks.json`. It verifies the following: the AIPW
score recovers the contrast when either nuisance branch is exact, clipping can
break the propensity branch, the count-weighted leaf update preserves the
pooled gradient whereas the symmetric formula does not, a finite particle law
can contain an exact zero atom, and expected reference distance differs from
distance of the mean. The reported status is `all exact checks passed`.

The targeted Python test command

```text
rtk python3 -m pytest -q tests/test_cwdb_arm_shared_tree.py tests/test_cwdb_model.py tests/test_g3_dgps.py
```

started successfully and passed the initial shared-tree and model checks, but
was interrupted after the DGP portion continued beyond the available review
window. It should be rerun to completion after manuscript integration. No
numerical result from that partial run is used as a proof.

## Recommended integration decisions

Use the supplied replacement section as the method-section basis. In the main paper, present only the final WCF-DR method and the optional WCF-ZIPT extension. Remove WCF-v1, WCF-R3, PTA-S, and repair-iteration labels from the scientific method and results narrative. Keep any comparison of candidate variants in a private development record or a clearly labeled supplementary ablation only if it is needed for reproducibility.

Define and report the complete finite-grid target dictionary: `TATE-K-j`, `TCATE-K-j`, `REF-ATE-K`, and `REF-TCATE-K`, alongside the law target and mean-quantile contrast. Name the reference-distance targets as a contribution, while avoiding a claim of causal identification beyond the stated exchangeability and positivity assumptions.

Replace “works when the propensity is not 0.5” with “is designed for nonconstant, outcome-relevant treatment assignment under conditional exchangeability and positivity.” Replace “first doubly robust distributional learner” or any similar novelty statement with the narrower and supportable claim that WCF combines a particle conditional-law estimator with AIPW calibration of declared scalar distributional functionals. A literature-wide first claim would require a systematic search and a precise comparison class.

## Limitations and final status

The code-to-text mapping is complete for the final WCF-DR path, the reference estimands, the optional two-part extension, and the current metric layer. The audit did not prove asymptotic consistency of the particle booster, prove efficiency of the AIPW implementation, authenticate every bibliography entry, or independently review the result tables. The fixed clip and nonnested hyperparameter selection are material qualifications for any theoretical DR statement. Final status: **method description requires the listed corrections; after those corrections, no additional finite-grid algebraic defect was found in the audited implementation path.**
