# Manuscript revision response

This companion records the response to the manuscript review. It is not part of the paper.

## Assignment, propensity, and robustness (points 1 and 8)

The propensity score is the function `e(x) = P(A=1 | X=x)`. It describes an assignment mechanism; it is not synonymous with endogeneity. A constant propensity of 0.3 can describe a randomized experiment with unequal allocation. Even a nonconstant propensity need not confound an outcome if it depends only on covariates unrelated to that outcome. Confounding by observed covariates arises when assignment and potential outcomes depend on overlapping information in X. A dependence on the untreated conditional mean is one example, not a requirement. Assignment can instead depend on covariates affecting dispersion or other aspects of the potential outcome distribution.

Pretreatment outcomes can be included in X. Dependence on the realized post-treatment outcome, or on unobserved potential outcomes beyond X, is a different identification problem. The proposed estimator assumes conditional exchangeability; it does not remove unmeasured confounding.

The current implementation calibrates marginal and moderator-group averages of outcome functionals using an augmented inverse-propensity score. Its double-robustness identity concerns those functionals, not the particle law or the pointwise quantile-vector predictions. It does not require `e(x)` to be a function of the baseline mean, and it does not guarantee better finite-sample performance whenever `e(x) != 0.5`. Classifiers other than logistic regression can supply estimated probabilities in the statistical procedure, although the present implementation fixes logistic regression.

Fixed clipping matters: the hard-overlap design allows true propensities below 0.02 and above 0.98, while the estimator clips to those bounds. Correctly estimating the unclipped propensity does not make the clipped propensity correct in that design. Double robustness also does not mean that consistency of both nuisance models alone establishes efficiency; further rate and regularity conditions are needed.

## Reference effects (point 15)

For a fixed reference quantile curve `q*`, use the observed-outcome functional `h_ref(q) = d_W,K(q,q*)`. The conditional contrast is the difference between expected treated and untreated distances, and the population contrast averages that difference over the target population. A negative contrast means treatment brings the outcome distribution closer to the reference on average. It does not say every unit benefits or that the reference is objectively ideal.

This is an expectation of distances. It is not the distance from the mean quantile curve to the reference, nor the distance between the two treatment-arm laws. These quantities differ for nonlinear functionals. The manuscript introduces the reference-based estimands as a contribution without claiming that all reference-distance effects or doubly robust distributional methods originate here.

## Checks and supporting artifacts

`check_revision_identities.py` performs exact rational checks of both branches of the AIPW identity, a clipping counterexample, the count-weighted shrinkage formula, and the distinction between expected distance and distance of the mean. It also verifies examples showing that a particle cloud can carry exact zero mass and that nonconstant propensity alone need not create confounding. Results are saved in `revision_identity_checks.json`. These calculations do not prove asymptotic properties of the fitted learner.

## Changes corresponding to the remaining review points

| Review point | Action |
|---|---|
| 2, 3, 4, 10, 13 | Rewrote the abstract, introduction, methods, results, discussion, and conclusion as an external academic paper. Removed the discarded skewness/scale hypothesis narrative, preregistration and gate scoreboards, repair history, and “not a theorem” language. |
| 5, 6 | Introduced arbitrary covariate dimension, quantile grid, and particle count in the method. Numerical settings appear in the simulation implementation subsection. `sensitivity_and_dgp_experiments.md` gives a staged K/M plan, actual API examples, common-grid evaluation, classifier changes, resource pilots, and self-contained Colab requirements. |
| 7: DGP interpretation | Gave the DGP equations directly. Regional income is an illustrative setting, not a claim that the simulation is an applied study. Removed the solicitation for collaborators and the empty Applied Study section. |
| 7: model explanation | Expanded the fitting update, preconditioned gradient, shared partition and leaf rule, monotone projection, initialization, calibration, estimands, and structural mixture. Added citations for the mathematical and statistical ingredients. |
| 8: comparison with Causal-DRF | Focused the contribution on conditional-law estimation plus AIPW functional calibration. Removed universal claims about separate-arm interpolation and superiority under any nonconstant propensity. Preserved observed WCF losses in the nonlinear suite. |
| 9 | Explained the optional two-part extension and cited related economic two-part models. Made explicit that its zero component is an entire all-zero unit-level quantile vector, not ordinary individual zeros within every unit's distribution. |
| 11, 12 | Added existing final-WCF comparisons for symmetric and nonlinear DGPs. Created and ran a cheap symmetric nonlinear-assignment DGP construction probe. Full estimator runs with nonlinear propensity and alternative classifiers are specified as future experiments, not fabricated results. |
| 14 | Used WCF as the display name of the calibrated `cwdb_dr` estimator. Removed PTA-S and intermediate WCF variants from paper tables and prose. The optional two-part estimator is separately identified as a law/component estimator without AIPW calibration. |
| 15 | Added TATE, TCATE, reference-distance TATE, and reference-distance TCATE results, with metric semantics and aggregation explained. Added per-functional tables to supplement aggregate functional errors. |
| 16 | Moved implementation settings into the simulation section. The appendix now contains substantive additional functional results rather than a tiny implementation note. |

## Material corrections found during integration

The old shared-leaf formula used symmetric `+/- c/2` adjustments around a count-weighted mean. That preserves the mean only for equal arm counts. The final formula uses the opposite arm's sample share and agrees with the code.

The old mixture formula called its classifier output the probability of degeneracy while assigning that output to the nondegenerate component. The final notation defines the probability of the nondegenerate component consistently. A finite particle law can contain exact zero atoms, so the manuscript no longer claims structural impossibility of zero-mass representation.

Some proposed tables labeled uncalibrated zero-inflation rows as final WCF. Those rows were removed during parent review. The zero-inflation table compares only the optional two-part estimator and the established forest baselines. The main calibrated WCF tables use `cwdb_dr` results from the merged phase-6 artifact. No intermediate method was renamed to imply a calibration it did not perform.

The current source uses two outcome folds by default, while the original IC configuration used three. The paper reports the study-specific settings and separately describes the five-fold propensity fits. The penalty is selected before the out-of-fold calibration fits and reused across them, so the present tuning is not fully nested; the paper makes no asymptotic inference claim for it.

The reference vector in the old implementation appendix omitted its cubic Hermite term. The revised DGP equations include both quadratic and cubic terms, matching the code.

The code's `mean_quantile_rmse` evaluates all quantile coordinates, whereas the old prose described only the grid mean. The paper now distinguishes these estimands. TATE values are test-standardized averages of training-bin calibrated estimates in the stored adapter, not the direct training-sample AIPW average; the evaluation section explains this detail.

The individual-curve shapes are symmetric in several existing DGPs, but that is different from symmetry of the random law over curves. The multimodal D6 design illustrates this distinction and WCF's observed limitations. No causal claim that assignment alone explains all performance differences is retained.

## What remains experimental work

The full K/M sensitivity study has not been executed. The nonlinear-propensity helper verifies construction and balance only; it does not fit or compare estimators. Flexible-propensity plumbing and self-contained sensitivity notebooks are specified in the requested plan, not claimed as implemented. An applied study and coverage/efficiency theory are also not supplied by this revision.

## Final validation

The final paper compiles to 16 pages without LaTeX warnings, unresolved references, unresolved citations, or overfull boxes. All 684 displayed numerical summaries were independently reconstructed from the merged result files: 348 aggregate rows and 336 functional-specific rows matched at tolerance $5\times10^{-10}$. The exact algebraic checks passed, and 37 targeted implementation tests covering the energy objective, shared treatment-arm tree updates, and contrast regularization passed. The final PDF was inspected page by page.

Three delegated specialist reviews were completed and reconciled. A requested additional complete-paper review could not run because the `gpt-5.6-luna` service reported that its usage limit had been reached. The final manuscript therefore has parent review and completed specialist input, but it is not represented as having passed that additional blind review.
