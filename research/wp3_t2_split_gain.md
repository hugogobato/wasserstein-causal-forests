# WP3-T2: finite-grid composite split gain

**Input files:** `research/finite_grid_estimands.md`, `research/algorithm_spec.md`, and WP3 in `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md`.

**Source to adapt:** Athey, Tibshirani, and Wager (2019), Sections 2–3, for honest adaptive forest splitting; Ćevid et al. (2022), Section 3, for the distinction between a target-specific quadratic split and a distribution-agnostic MMD split. The identity below is proved directly and is not imported as a theorem.

**Assumptions used:** A10 and A11, plus fixed finite `K` and `J` from A12. The calculation is conditional on the score vectors, so it does not use A3 or a nuisance convergence condition.

**Status:** `PASS`, finite-node algebra only.

## 1. Definition

Let (A) be a split node, with (A=A_L\mathbin{\dot\cup}A_R), (n_L=|A_L|>0), (n_R=|A_R|>0), and (n=n_L+n_R). Let (r_i\in\mathbb R^{K+J}) be the scaled finite score and let (omega_c>0) be the direct-sum coordinate weights. The node impurity is

\[
I(A)=\sum_{i\in A}\sum_c\omega_c(r_{ic}-\bar r_{A,c})^2.
\]

The gain is (G=I(A)-I(A_L)-I(A_R)).

## 2. Direct algebra

For each coordinate (c), the one-dimensional ANOVA identity gives

\[
\sum_{i\in A}(r_{ic}-\bar r_{A,c})^2
=\sum_{i\in A_L}(r_{ic}-\bar r_{A_L,c})^2
+\sum_{i\in A_R}(r_{ic}-\bar r_{A_R,c})^2
+\frac{n_Ln_R}{n}(\bar r_{A_L,c}-\bar r_{A_R,c})^2.
\]

Multiplying by (omega_c) and summing over the finite coordinates yields

\[
G=\frac{n_Ln_R}{n}\sum_{c=1}^{K+J}
\omega_c(\bar r_{A_L,c}-\bar r_{A_R,c})^2.
\]

Every factor outside the squared mean difference is positive. Therefore, if at least one standardized target coordinate has different child means, (G>0). This proves C3.1 for a finite node. It does not say that a greedy forest will select the scientifically best split, nor does it establish population localization.

The curve-only version is the same identity with the sum restricted to (c\leq K). Consequently, a pure-functional signal with zero curve-coordinate child differences has zero curve-only population gain but a positive composite gain whenever its functional coordinate differs.

## 3. Exhaustive calculation

`exhaustive_gain_identity` enumerates every nontrivial subset and complement for a seven-row, four-coordinate node. With weights `(0.15, 0.10, 0.20, 0.55)`, the maximum absolute difference between the impurity definition and the child-mean formula is below (10^{-12}). The separate one-coordinate signal check gives a strictly positive gain equal to

\[
\frac{3\cdot2}{5}(0.15+0.20)=0.42
\]

when the two differing coordinates have unit mean gaps.

## 4. Numerical hooks

N3a is the two-group pure-functional construction in `research/wp3_invariants.py`: the curve-only root remains a leaf while the composite root has positive gain. N3b duplicates every curve grid point and halves its quadrature weight. The resulting predictions differ by less than (10^{-8}) and the complete tree structures are identical, demonstrating that the criterion depends on the normalized quadrature measure rather than the raw count of duplicated curve coordinates.

## Checks run

Command: `python3 research/wp3_invariants.py`.

Observed failures: none.

Unresolved questions: C3.2, the limiting behavior of training-only scaling under a forest asymptotic regime, is intentionally not claimed here.

`PASS`
