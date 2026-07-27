# WP2-T6: finite-vector conditional double robustness

**Input files:** `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md` Sections 2.6 and WP2-T6; `research/finite_grid_estimands.md`; `research/assumption_ledger.md`.

**Source to adapt:** Standard AIPW algebra, with the distribution-valued identification template in Lin, Kong, and Wang (2023), Theorem 2.

**Assumptions used:** A2–A7 for the target and identification, plus a fitted-propensity range restriction \(e(X)\in[\eta,1-\eta]\) so the score denominators are defined.

**Status:** `PASS`, finite-vector version.

## Score

Write

\[
 Y=U_K(Q)\in\mathbb R^{K+3},
 \qquad
 e_0(x)=P(Z=1\mid X=x),
 \qquad
 m_z^0(x)=E(Y\mid Z=z,X=x).
\]

For generic nuisance functions \(e,m_0,m_1\), define

\[
 \phi(O;e,m_0,m_1)
 =m_1(X)-m_0(X)
 +\frac{Z}{e(X)}\{Y-m_1(X)\}
 -\frac{1-Z}{1-e(X)}\{Y-m_0(X)\}.
\]

The vector \(Y\) is the unscaled scientific oracle outcome. All operations are coordinatewise. In the feasible implementation \(Y\) is replaced by \(U_K(\widehat Q)\), which is a proxy score rather than a latent-target score until A8 supplies an observation-recovery argument.

## Conditional expectation at generic nuisances

For \(P_X\)-almost every \(x\), let \(e_0=e_0(x)\), \(m_z^0=m_z^0(x)\), and evaluate the conditional expectation given \(X=x\). Since

\[
 E\{Z(Y-m_1)\mid X=x\}=e_0(m_1^0-m_1)
\]

and

\[
 E\{(1-Z)(Y-m_0)\mid X=x\}
 =(1-e_0)(m_0^0-m_0),
\]

we obtain

\[
\begin{aligned}
 E\{\phi(O;e,m_0,m_1)\mid X=x\}
 &=m_1-m_0
 +\frac{e_0}{e}(m_1^0-m_1)
 -\frac{1-e_0}{1-e}(m_0^0-m_0).
\end{aligned}
\]

This identity holds as a vector identity, hence also coordinate by coordinate.

## Case 1: correct propensity

If \(e=e_0\), then

\[
\begin{aligned}
 E\{\phi\mid X=x\}
 &=m_1-m_0+(m_1^0-m_1)-(m_0^0-m_0)\\
 &=m_1^0-m_0^0\\
 &=\Theta_K(x),
\end{aligned}
\]

even if both outcome regressions are misspecified.

## Case 2: correct arm regressions

If \(m_1=m_1^0\) and \(m_0=m_0^0\), the two residual conditional means vanish for any propensity function with valid denominators, so

\[
 E\{\phi\mid X=x\}=m_1^0-m_0^0=\Theta_K(x).
\]

Thus the score is conditionally doubly robust for the finite target vector at \(P_X\)-almost every \(x\). The phrase “both outcome regressions” is essential: correctness of only one arm regression is not sufficient for the generic double-robust identity.

## Relation to forest localization

The oracle score has conditional mean \(\Theta_K(x)\) for \(P_X\)-almost every \(x\). An honest forest may therefore use it as a finite-vector pseudo-outcome for localization. This algebra does not prove that adaptive forest weights are independent of score noise, that an empirical-quantile score targets the latent vector, forest consistency, or a central limit theorem. Those are A8 and later-work-package questions. Cross-fitting nuisance estimates and within-tree honesty are separate requirements.

## Checks run

The generic conditional-expectation identity, the two double-robust cases, the scalar \(K=1,J=0\) collapse, and the known randomized-propensity case are represented in `research/wp2_invariants.py`.

## Observed failures

None under the stated denominator and moment conditions.

## Unresolved questions

The rate at which replacing \(U_K(Q)\) by \(U_K(\widehat Q)\) perturbs the score is not covered by this result. It remains under A8 and WP5.

`PASS`

**Source:** [Lin, Kong, and Wang, “Causal Inference on Distribution Functions,” arXiv:2101.01599](https://arxiv.org/abs/2101.01599), Theorem 2.
