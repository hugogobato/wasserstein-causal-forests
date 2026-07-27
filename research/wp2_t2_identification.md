# WP2-T2: finite-grid identification derivation

**Input files:** `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md` Sections 2.3–2.5; `research/finite_grid_estimands.md`; `research/assumption_ledger.md`.

**Source to adapt:** Lin, Kong, and Wang (2023), Theorems 1–2, especially their identification of the average causal effect map from conditional observed-outcome means. The source establishes a global distribution-valued result; this file applies the same four-step argument to the finite-grid conditional target.

**Assumptions used:** A2 consistency, A3 distribution-level conditional exchangeability, A4 overlap, A6 finite-vector square integrability, and A7 functional measurability and integrability.

**Status:** `PASS`.

## Claim

For each \(z\in\{0,1\}\), the following equality holds for \(P_X\)-almost every \(x\):

\[
 M_z^K(x)=E\{U_K(Q)\mid Z=z,X=x\}.
\]

Consequently, \(\Theta_K(x)=M_1^K(x)-M_0^K(x)\) is identified from the oracle law of \((X,Z,Q)\), up to \(P_X\)-almost-everywhere equivalence. The argument is coordinatewise in \(\mathbb R^{K+3}\), so no Bochner-integral result is needed at this stage. Identification from inner samples additionally requires A8 and is not supplied by this four-line calculation.

## Four equality steps

Start from the latent arm-specific conditional mean:

\[
\begin{aligned}
 M_z^K(x)
 &=E\{U_K(Q^{(z)})\mid X=x\} \\
 &=E\{U_K(Q^{(z)})\mid Z=z,X=x\} \\
 &=E\{U_K(Q^{(Z)})\mid Z=z,X=x\} \\
 &=E\{U_K(Q)\mid Z=z,X=x\}.
\end{aligned}
\]

The first line is the definition of \(M_z^K\). The second line uses A3 because \(U_K\) is a measurable deterministic transformation of the potential distribution, including the unit-level nonlinear coordinates. The third line uses A2: conditional on \(Z=z\), the observed potential outcome is \(Q^{(Z)}=Q^{(z)}\). The fourth line uses the observed-outcome definition \(Q=Q^{(Z)}\).

The denominators needed for the observed conditional mean are well-defined by A4. The finite-vector expectations exist by A6 and A7.

## G-formula form

Equivalently, with \(e(x)=P(Z=1\mid X=x)\),

\[
 M_z^K(x)=E\left[
 \frac{\mathbf 1\{Z=z\}U_K(Q)}{P(Z=z\mid X)}
 \middle| X=x\right].
\]

For \(z=1\), the conditional expectation of the weighted term is

\[
 E\left[\frac{Z U_K(Q)}{e(X)}\middle|X=x\right]
 =\frac{e(x)}{e(x)}E\{U_K(Q^{(1)})\mid X=x\}=M_1^K(x),
\]

and the control expression is analogous. This is the finite-grid analogue of the identification step in Theorem 2 of Lin et al. (2023), whose theorem is stated for the Wasserstein causal effect map and a global average.

## Functional-coordinate consequence

For each fixed functional \(T_j\), applied to recovered raw income, the same calculation gives

\[
 E[T_j\{\exp(Q^{(z)})-1\}\mid X=x]
 =E[T_j\{\exp(Q)-1\}\mid Z=z,X=x].
\]

This is not the identity

\[
 T_j\!\left[\exp\{E(Q^{(z)}\mid X=x)\}-1\right]
 =E[T_j\{\exp(Q^{(z)})-1\}\mid X=x],
\]

which fails for nonlinear \(T_j\). The exact failure is established in WP2-T4.

## Source-transfer boundary

Lin et al. identify an average causal effect map through arm-specific conditional expectations of transformed quantile functions, and their Theorem 1 relates the map to an average of individual maps. The present result is narrower and more local: it is a direct finite-dimensional conditional g-formula for the oracle \(U_K\). No forest consistency, nuisance-rate, inner-sample identification, or continuum inference statement is transferred from Lin et al.

## Checks run

The four equality steps were checked symbolically and are exercised by the score and target invariants in `research/wp2_invariants.py`.

## Observed failures

None for the oracle target under A2–A7.

## Unresolved questions

Identification of the post-G2A confirmation target \(\Theta_V(v)\) or a selected subgroup target is deferred to WP2-T3. Its outer averaging over \(X\mid V=v\) must not be replaced by adjustment on \(V\) alone.

`PASS`

**Source:** [Lin, Kong, and Wang, “Causal Inference on Distribution Functions,” arXiv:2101.01599](https://arxiv.org/abs/2101.01599), Theorems 1–2.
