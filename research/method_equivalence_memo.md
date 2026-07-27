# WP3-T9: method-equivalence kill test

**Input files:** `research/algorithm_spec.md`, `research/wp3_odcf.py`, and WP3 in the theory plan.

**Source to adapt:** Athey, Tibshirani, and Wager (2019), Sections 2–3; Oprescu, Syrgkanis, and Wu (2019), Sections 2–3; Nie and Wager (2021), Sections 2–3; and Ćevid et al. (2022), Section 3.

**Assumptions used:** finite fixed `K` and `J`, a fixed unscaled scientific score matrix, fixed split-only coordinate scales, a diagonal direct-sum quadratic loss, identical candidate splits and randomization, and disjoint split/populate indices. This is a conditional finite-algorithm statement, not a claim of raw-outcome honesty or A10 asymptotic regularity.

**Status:** `PASS`, with an important novelty warning.

## Equivalence statement

Conditional on the already constructed unscaled score matrix \(\widehat\phi_i\) and fixed split scales, ODCF-v1 is exactly a multi-output regression forest whenever the comparator uses the same finite weighted squared-error impurity, trapezoidal quadrature weights, split-only coordinate scales, candidate partitions, child-balance rules, split/populate indices, and randomization. The identity in `research/wp3_t2_split_gain.md` is the reason: the ODCF composite criterion is a diagonal weighted multi-output sum of squared errors. Leaf predictions remain averages of the unscaled scientific scores.

Thus the finite-grid shared-partition algorithm alone is not a methodological novelty claim. Calling the method a “Wasserstein forest” does not change this equivalence. The implemented curve term is a trapezoidal discretization of a **trimmed log-scale \(L^2\) quantile loss on \([0.05,0.95]\)**. It is inspired by the one-dimensional quantile representation of \(W_2^2\), but it is not full \(W_2^2\) on \([0,1]\) and is only a finite-grid pseudometric on unrestricted distributions.

## Where equivalence ends

The following components are not supplied automatically by a generic multi-output regressor: region-level cross-fitted doubly robust score construction; the distinction between a latent distribution-valued outcome and a scalar outcome's conditional law; pre-averaging nonlinear functional coordinates; an explicit inner-sample observation model; the provisional direct-score noise heuristic; confirmatory subgroup or low-dimensional target handling; and the project-specific joint inferential layer. Even these components are not automatically novel. Causal-DRF, FOCaL, distribution-valued causal-effect work, and DRF inference remain mandatory prior-art and baseline checks. The local `mmd_score` variant is only an MMD-on-cross-fitted-score forest inspired by DRF; it is not the mandatory official Causal-DRF benchmark.

The conditional equivalence also clarifies the honesty boundary. Disjoint split/populate indices prove score-level index separation after the scores and scales are treated as fixed. They do not show that a populate outcome was absent from every nuisance fit or global preprocessing operation that influenced a split score. Any full raw-data honesty or inference claim requires a separate argument or a stronger sample-splitting construction.

## Kill-test conclusion

ODCF-v1 should not be presented as new merely because it combines several output coordinates in one tree. The defensible contribution must be one of the following, selected only after G2A: a nontrivial inner-sample oracle-equivalence result, a composite-splitting result that explains a demonstrated D4/D5 advantage, or a strong applied contribution with correctly separated nonlinear effects and joint uncertainty. If no such component survives, the method-equivalence result is a reason to abandon the claimed algorithmic novelty rather than a reason to add more theory.

## Checks run

The duplicate-grid test in `research/wp3_invariants.py` verifies the finite weighted-loss equivalence numerically under explicit trapezoidal geometry. The pure-functional score test verifies that the distinction between curve-only and composite variants is which coordinates are allowed into the same weighted loss. Neither test establishes population split selection, forest consistency, or an empirical advantage.

Observed failures: none.

Unresolved question: whether the feasible inner-sample path produces a material empirical or theoretical distinction remains a WP5/WP9 question.

`PASS`
