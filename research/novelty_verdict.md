# WP1 novelty verdict

Task IDs: `WP1-T0` through `WP1-T3` (PRE-SIM)  
Date and search cutoff: 2026-07-27  
Status: `CONDITIONAL PASS`, G0 decision `CONTINUE` only for a narrowed empirical and inner-sampling test

Source theorem/result references screened: Causal-DRF Theorems 4.1 and 4.3,
FOCaL Lemma 4.2, Theorem 4.4, and Proposition 4.5, plus the main theorem or
result statements exposed in the planned source landing pages and papers,
including R3D (arXiv:2504.03992), Geodesic Causal Inference
(arXiv:2406.19604), and DR-FoS (arXiv:2501.06024;
DOI 10.1515/jci-2025-0045).
No post-G2A appendix extraction is claimed.

## Inputs, scope, and operational test

Inputs were `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md`,
`wasserstein-distributional-causal-forests.md`, the WP0 control files under
`research/`, and the verified source links in
`Wasserstein_Distributional_Causal_Forests_Literature_Reading_List.md`.
The repaired target is the finite vector consisting of the conditional mean
quantile curve and fixed nonlinear unit-level distributional effects,

\[
E\{Q^{(1)}-Q^{(0)}\mid X=x\},
\qquad
E\!\left[T_j\{\exp(Q^{(1)})-1\}-T_j\{\exp(Q^{(0)})-1\}\mid X=x\right],
\]

with binary treatment, honest forest localization, nuisance correction, latent
region-level distributions estimated from inner household samples, and joint
confirmation inference. The relevant scope is the frozen ledger A1--A12,
especially distribution-level exchangeability and overlap (A3--A4),
square-integrability and functional domains (A6--A7), explicit inner-sample
recovery rather than bounded-sample identification (A8), cross-fitted nuisance functions (A9), honest forest construction
(A10), and fixed finite output dimension (A12).

For this audit, `direct_hit=yes` would require one source to supply all of the
following in one estimand and method: distribution-valued potential outcomes
at the outer-unit level, binary heterogeneous causal localization, an
orthogonal or doubly robust forest construction, nonlinear functionals applied
before conditional averaging, explicit inner-sample error treatment, and joint
inference. A method that supplies only a conditional law of a scalar outcome,
a functional curve, or a distributional effect through a kernel embedding is
marked as an adjacent or partial hit, not silently upgraded to a direct hit.
The absence of one paper containing every element is not treated as affirmative
novelty when the proposed estimator is an off-the-shelf composition.

## WP1-T0 search record

The exact-title and concept searches required by the plan were run for:
“distribution-valued outcome heterogeneous treatment”, “random object CATE
forest”, “functional outcome causal forest”, “Hilbert-valued R-learner”,
“Fréchet causal forest”, “Wasserstein causal forest distribution outcome”,
“estimated distribution response causal inference”, “regression discontinuity
distribution-valued outcomes”, and “geodesic causal inference”. The searches
were expanded to the exact titles and author combinations of the planned
screening groups and to `Causal-DRF`, `FOCaL`, `R3D`, `Geodesic Causal
Inference`, and `DR-FoS`.

The search found no source satisfying the complete direct-hit definition. It
did find a materially closer boundary than the original queue anticipated:
`Causal-DRF` (Näf, Park, and Susmann, AISTATS 2026) uses one shared causal
forest to estimate a conditional kernel treatment effect, proves fixed-
covariate Hilbert-space consistency and asymptotic normality, and supplies
resampling-based inference. `FOCaL` (Salmaso, Testa, and Chiaromonte, 2026)
uses a doubly robust meta-learner with a generic final regression stage for
functional CATEs and simultaneous bands.
These are included in the matrix rather than treated as afterthoughts.
R3D is also a material boundary: it explicitly studies random distribution
outcomes, the two sampling layers, empirical quantiles, and uniform inference,
although under an RD design rather than an orthogonal forest. Geodesic Causal
Inference supplies doubly robust average causal inference for general
random-object outcomes. DR-FoS supplies doubly robust functional average
treatment effects and simultaneous bands. None is an exact forest-CATE hit,
but all three remove broader novelty language.

## WP1-T1 screen of the planned sources

Lin, Kong, and Wang provide the closest treatment of causal distribution-valued
outcomes and estimated unit distributions, including doubly robust estimation,
but their target is not a heterogeneous causal forest. Athey, Tibshirani, and
Wager, Oprescu, Syrgkanis, and Wu, and Nie and Wager provide the honest,
orthogonal, and quasi-oracle HTE components, but their outcomes are scalar or
finite-dimensional. Qiu, Yu, and Zhu provide forest-weighted local Fréchet
regression for random objects, while Ćevid et al. and Näf et al. provide
distribution-sensitive forests and Hilbert-space inference. Those works are
predictive or conditional-law methods, not forests for latent distribution-
valued potential outcomes with causal nuisance correction.

Du, Biau, Petit, and Porcher are the closest title-level Wasserstein forest
threat. Their construction concerns conditional distributions of scalar
potential outcomes and does not observe an income distribution for each outer
region. Van Dijcke and Wüthrich make quantile projection and uniform inference
non-novel in a global IV setting. Bhattacharjee et al. cover doubly robust
causal effects for random objects with continuous treatment, and Huang et al.
cover doubly robust policy learning for distribution-valued outcomes. Neither
uses binary heterogeneous forest localization for the repaired target.

The newly found adjacent papers change the wording required for any future
paper. It must not claim the first distributional causal forest, the first
functional CATE, the first doubly robust random-object causal method, or the
first joint inference for a distributional treatment effect.

R3D additionally makes “the first causal method with distribution-valued
outcomes observed through inner samples” unavailable. Geodesic Causal
Inference and DR-FoS make broad average random-object and functional-effect
claims unavailable. The remaining question is forest-specific: whether inner
sampling changes honest heterogeneous localization or inference in a way not
covered by these sources, or whether shared localization of the curve and
pre-averaged nonlinear coordinates yields a reproducible gain.

## WP1-T2 matrix result

`research/prior_art_matrix.csv` contains 17 screened papers, including the
planned core/boundary sources and the current R3D, Geodesic Causal Inference,
DR-FoS, FOCaL, and Causal-DRF search hits. The matrix uses `partial` when a
paper covers a nearby component
but changes the outcome object, treatment target, or observation scheme. Every
row has `direct_hit=no`. The most serious row is Causal-DRF, not Du et al.,
because it already combines binary treatment, causal forest localization,
distributional effects, one shared forest, and inference. The missing pieces
are precisely the ones that could still generate the contribution: latent
distribution-valued outer-unit outcomes from inner samples, nonlinear
unit-level functional effects, and their interaction with nuisance-corrected
forest localization.

FOCaL is learner-generic and can use a forest as its final regression step.
Appending the fixed nonlinear coordinates to its functional outcome is
algebraically immediate. Therefore the FOCaL direct-sum construction is a
mandatory baseline, not evidence of a new method. Causal-DRF is the mandatory
shared distribution-sensitive forest baseline.

## Guardrail and collapse check

Surface similarity is not a direct hit, but composition is not automatically a
contribution. In particular, “multi-output causal forest plus plug-in Gini”
would be a composition unless the paper proves or demonstrates why
\(E[T\{\exp(Q^{(1)})-1\}-T\{\exp(Q^{(0)})-1\}\mid X]\) is identified and
estimated separately from a functional of the two arm mean quantile curves,
and why the inner-sample error
does not invalidate the claimed inference.

If every region has one household, its empirical distribution is a point mass
and the empirical-proxy problem collapses toward scalar or low-rank
multivariate HTE. The latent distribution-valued target does not collapse
unless the latent distributions themselves are assumed degenerate. With an
unrestricted latent mixing law it is instead nonidentified from one inner
draw. The claimed novelty cannot be advertised as a generic multi-output
forest improvement.

## Verdict and required narrowing

`CONTINUE` at G0 only as a conditional empirical test. The screened corpus
contains no exact direct hit, but absence of the exact conjunction is
insufficient because much of ODCF-v1 is composable from existing methods.
Before substantial theory, the project must benchmark against Causal-DRF and
a FOCaL-style doubly robust learner applied to the same unscaled augmented
vector. R3D is a mandatory source for any inner-sampling theorem. The primary
contribution must be either 1. a nontrivial forest-by-inner-sampling result
under an identified observation regime, or 2. a reproducible Pareto advantage
from shared localization of the curve and nonlinear coordinates. If neither
survives, the decision is `PIVOT-INNER-SAMPLING` or `ABANDON`, not a relabeling
of the same finite-vector forest.

Post-G2A work remains pending: no WP1-T4, WP1-T5, or WP1-T6 theorem extraction
is claimed here. The complete source-by-source feature matrix is the audit
artifact; novelty is a live conjecture, not a proof of universal absence.

**CONTINUE**
