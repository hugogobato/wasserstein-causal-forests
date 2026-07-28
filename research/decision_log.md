# Decision log

Project: Orthogonal Distributional Causal Forest (ODCF)

This file is the WP0 source of truth for consequential choices. A choice may be changed only before Gate G2, with a dated reason that does not use comparative outcome results. Every change must preserve the stable decision ID and append a new entry; prior entries are not deleted.

## WP0-D001, control-file interpretation

Date: 2026-07-27

The original plan said that Section 0.5 contained seven control files, but its code block enumerated eight: `decision_log.md`, `evidence_log.md`, `claims_register.csv`, `assumption_ledger.md`, `prior_art_matrix.csv`, `application_design_memo.md`, `cited_results_cheatsheet.md`, and `falsification_verdicts.csv`. All eight enumerated files were created, and the plan's prose count has now been corrected to eight.

Status: frozen for WP0.

## WP0-D002, observation regime and weighting

Date: 2026-07-27

The primary theory and simulation regime is Regime A from the concept note. The outer unit is a region, treatment is binary and assigned at region level, and the latent outcome is the region's income distribution. Inner household samples estimate the observed regional distribution. The primary estimand weights regions equally. Inner-sample precision weights are not part of the primary estimand because they can change the target to a size- or precision-weighted region population. They remain a WP8 sensitivity branch.

Status: frozen for WP0 and the pre-simulation implementation.

## WP1-D010, current-literature boundary expansion

Date: 2026-07-27

The exact-title and concept searches found two relevant sources absent from the
initial queue: Näf, Park, and Susmann (2026), `Causal-DRF`, and Salmaso, Testa,
and Chiaromonte (2026), `FOCaL`. Both are added to the prior-art matrix and the
novelty boundary. Causal-DRF is the closest forest comparator because it uses a
single shared causal forest for conditional kernel treatment effects with
Hilbert-space inference. FOCaL is the closest functional-CATE comparator
because it uses a doubly robust meta-learner and simultaneous bands. Neither is
classified as a direct hit because neither handles all of latent
region-level distributions estimated from inner samples, unit-level nonlinear
functional effects, and the repaired target vector.

Status: frozen for G0 and the pre-simulation benchmark set.

## WP1-D011, G0 novelty verdict

Date: 2026-07-27

The rapid audit ends in `CONTINUE`. The exact repaired conjunction is absent
from the screened corpus, but the broad claims “distributional causal forest,”
“functional CATE,” “doubly robust random-object causal inference,” and “joint
distributional treatment-effect inference” are already occupied in adjacent
forms. The viable contribution is the interaction between honest causal
localization, estimated distribution-valued outer-unit outcomes, and
pre-averaging nonlinear functional effects. Causal-DRF and FOCaL are mandatory
baselines before the contribution is treated as methodologically valuable.

Alternatives considered: `PIVOT` immediately to inner-sampling-only, or
`ABANDON` if Causal-DRF covers the practical target. Evidence does not force
either choice yet because the matrix still identifies two unfilled interfaces,
but those alternatives become the next gate if the benchmark shows no added
value.

Status: G0 `CONTINUE`, with the narrowing above.

## WP0-D003, outcome transformation and quantile domain

Date: 2026-07-27

The primary transformed income scale is `log1p(real household income)`. The latent raw-income quantile is denoted `Q_raw`, and the analysis quantile is `Q_log1p = log1p(Q_raw)`. The quantile domain is

\[
\mathcal P=[0.05,0.95].
\]

The primary pilot grid has `K=49` equally spaced points,

\[
p_k=0.05+0.90(k-1)/48,\qquad k=1,\ldots,49,
\]

with trapezoidal quadrature weights normalized to approximate integration over `\mathcal P`. Raw income, alternative transformations, and alternative trimming are robustness branches, not adaptive primary choices.

Rationale: the plan requires a fixed interior interval and gives `K=49` as the pilot grid. The concept note identifies log-income as the default response scale because of upper-tail sensitivity of raw-income Wasserstein loss.

Status: frozen until the pre-G2 change rule is invoked.

## WP0-D004, functional coordinates

Date: 2026-07-27

The initial fixed collection is `J=3`:

1. Gini coefficient;
2. Theil T index;
3. Atkinson index with inequality-aversion parameter `epsilon_A=0.5`.

These functionals are computed at the unit level before conditional averaging. They are evaluated on the raw-income quantile recovered from the transformed quantile,

\[
Q^{\mathrm{raw}}(p)=\exp\{Q^{\mathrm{log1p}}(p)\}-1,
\]

so the quantile curve and inequality coordinates have declared, distinct scales. The mean-income domain condition and the functional-specific treatment of zero income must be checked in WP2 and WP5. A quantile ratio or poverty functional is deferred from version 1 because its scale and threshold conventions are not yet frozen.

Status: frozen until the pre-G2 change rule is invoked.

## WP0-D005, confirmatory scope

Date: 2026-07-27

Confirmatory claims are restricted to frozen subgroups or a low-dimensional modifier `V=g(X)` with dimension at most two. Fully nonparametric pointwise inference for high-dimensional `X` is exploratory only. The primary forest target remains the finite vector consisting of the quantile coordinates and the three functional coordinates.

Status: frozen by the approved repaired formulation.

## WP0-D006, simulation metrics

Date: 2026-07-27

The primary simulation metric vector is:

1. quadrature-weighted integrated squared error for the quantile-effect curve;
2. RMSE for each of the three functional effects;
3. worst-coordinate standardized error;
4. nominal 95 percent pointwise and simultaneous coverage when an inference method is being evaluated.

Monotonicity violations of arm curves, average band width, calibration slope and intercept, runtime, and peak memory are mandatory diagnostics. No single composite leaderboard is introduced at WP0. G2 comparisons must report the primary metric vector and the prespecified Pareto rule in the plan.

Status: frozen for the pilot.

## WP0-D007, simulation-result schema

Date: 2026-07-27

Every long-format simulation result row must contain the fields `claim_id`, `dgp_id`, `observation_regime`, `evaluation_manifest_id`, `n_regions`, `inner_n`, `seed`, `method`, `metric`, and `value`. `observation_regime` distinguishes oracle latent truth, feasible growing-inner recovery, an identified measurement model, and an empirical-proxy estimand. `evaluation_manifest_id` locks the evaluation distribution, weights, truth standardizers, and coverage family. Additional fields may be appended, but these names and meanings are stable. For heterogeneous inner samples, `inner_n` records the design label and the optional fields `inner_n_min` and `inner_n_max` record the range.

The schema-level check is recorded in `research/simulation_results_schema.md`. No simulation result rows exist yet, so this is a structural pass rather than an empirical result.

Status: frozen for WP0 and WP9.

## WP0-D008, compute guardrail

Date: 2026-07-27

The current machine has 20 logical CPUs and approximately 15 GiB RAM. At the snapshot time, approximately 8.4 GiB was available and an existing job had eight visible loky workers. No WP9 sweep is to be launched from this task. Later parallel runs must use at most eight workers when the machine is otherwise idle, fewer when competing work is active or available RAM is below 10 GiB, and must keep expected memory below 12 GiB.

Status: frozen as an execution guardrail; recheck before experiments.

## WP0-D009, raw versus transformed quantile notation

Date: 2026-07-27

The latent raw-income quantile is `Q_raw`; the analysis quantile used in the finite-grid forest is `Q_log1p = log1p(Q_raw)`. Functional coordinates are applied to the raw distribution recovered from `Q_log1p`. This removes the ambiguity in the source plan where `Q` can refer to either scale.

Status: frozen for WP0 and the pre-simulation implementation.

## WP2-D010, functional quadrature grid

Date: 2026-07-27

The finite implementation uses a separate full-probability midpoint grid with
`L=400` points for Gini, Theil T, and Atkinson(0.5). The curve coordinate still
uses the frozen interior `K=49` grid on `[0.05, 0.95]`. The full-probability
grid is necessary because the inequality functionals integrate over `(0, 1)`;
the even grid also makes the prespecified two-point counterexample exact at the
one-half mass split. This is a fixed numerical definition, not a
data-dependent tuning choice. It may be changed only before G2 under the
existing decision-change rule.

Status: frozen for WP2 and the pre-simulation implementation.

## WP3-D001, finite-grid scaling rule

Date: 2026-07-27

The frozen rule is `robust_sd`, \(s_j=d_Q/\{1.4826\operatorname{MAD}(R_j)\}\), with a declared unit reference if the training curve has zero dispersion. It was chosen analytically because it is Gaussian-consistent, robust, and has a sample-size-stable population limit when the relevant dispersions are positive. The `null_score_se` alternative is excluded because its scale grows at order \(\sqrt n\), contrary to A11.

The deterministic script is an illustrative regression diagnostic, not a selection experiment. With the corrected trapezoidal weights it reports legacy diagnostic scores 1.09361 for `robust_sd`, 1.18511 for `mad`, and 1.16837 for `null_score_se`. A single seed and half-sample cannot prove downstream superiority.

Status: frozen for WP3 and the pre-simulation implementation.

## WP0-D012, oracle versus feasible observation regime

Date: 2026-07-27

The latent distribution target is identified by the causal g-formula only in
the oracle experiment that observes \(Q_i\). It is not nonparametrically
identified from bounded-size inner samples under an unrestricted latent
random-distribution law. The binding witness uses \(m_i=1\): a latent
distribution that is \(\delta_1\) or \(\delta_3\) with equal probability and a
latent distribution that is deterministically
\((\delta_1+\delta_3)/2\) induce the same inner-draw law but different mean
quantiles and mean unit-level Gini coefficients.

Feasible latent-target claims must therefore use either 1. a triangular array
with \(\min_i m_i\to\infty\) and explicit uniform bias and \(L^2\) error rates,
or 2. an identified measurement, replicate, or validation model. A generic
two-stage correction, cross-fitting, precision weighting, or bootstrap is not
an identification argument. Oracle, feasible-recovery, and empirical-proxy
simulation paths must be labeled separately.

This entry supersedes any reading of WP0-D002 under which the latent target is
automatically feasible merely because inner samples are observed.

Status: frozen foundational repair before simulation.

## WP0-D013, raw support and quantile convention

Date: 2026-07-27

Raw income has support in \([0,\infty)\), making the frozen log1p transform
well-defined. Gini, Theil, and Atkinson retain their positive-mean and
functional-specific tail domains. Raw and empirical quantiles use the
generalized inverse

\[
Q(p)=\inf\{y:F(y)\ge p\}.
\]

Linear interpolation between sample order statistics is not the declared
empirical-quantile target. The fixed finite score vector has an explicit
\(2+\delta\) moment; an \(L^2\) function moment alone is not used to justify
point evaluations \(Q(p_k)\).

Status: frozen foundational repair before simulation.

## WP1-D014, expanded prior-art boundary and conditional G0

Date: 2026-07-27

The rapid screen now includes R3D (arXiv:2504.03992), Geodesic Causal
Inference (arXiv:2406.19604), and DR-FoS (arXiv:2501.06024;
DOI 10.1515/jci-2025-0045), in addition to Causal-DRF and FOCaL. These sources
do not constitute an exact direct hit. They do remove broad claims concerning
two-layer distribution-valued causal inference, doubly robust geodesic or
functional average effects, functional CATE learning, and shared
distribution-sensitive causal forests.

G0 remains `CONTINUE` only as a conditional empirical test. Causal-DRF and a
FOCaL-style learner applied to the same unscaled augmented vector are
mandatory baselines, and R3D is a mandatory source for an inner-sampling
theorem. Continuation beyond the empirical gate requires either a nontrivial
forest-by-inner-sampling result under an identified observation regime or a
reproducible advantage from shared localization. Otherwise the decision is
`PIVOT-INNER-SAMPLING` or `ABANDON`.

Status: supersedes the broader reading of WP1-D011; G0 is conditional.

## WP9-D015, second pilot design repair

Date: 2026-07-28

The first pilot (`eval-v2`, claim `WP9-T3-colab`) is complete and fails Gate G2
on every decidable criterion. Before that failure can be read as evidence about
the method, three defects in the pilot itself must be repaired.

First, D0 through D5 ran only under `oracle_latent`, which supplies the exact
latent quantile functions together with oracle nuisances. The AIPW score is then
noiseless, and measured errors collapsed to between 1e-7 and 1e-16, with
`rmse_functional_0` on D1 equal to 1.1e-16 for six methods. Those cells ranked
floating-point behavior, not statistical performance, so they are uninformative
rather than negative. Six of the seven pilot DGPs carried no statistical
content, and D8 was the only cell with genuine sampling noise. The second pilot
runs every DGP under `feasible_growing_inner`.

Second, no `D8 x oracle_latent` cell was run, so the feasible-oracle gap in
criterion 4 was undefined and `g2_checks.py` returned `null`. The second pilot
adds that reference cell.

Third, `worst_standardized_error` divided by the empirical standard deviation of
the truth across units. On D4 most coordinates are constant, that scale collapsed
to about zero, was floored at 1e-8, and inflated the metric to between 1e6 and
1e7. It is the declared primary metric for D5 and D8, so it is now standardized
by frozen constants declared in `sim.config.frozen_coordinate_scales`, identical
across every DGP, regime, method, seed, and sample size. The evaluation manifest
is tagged `eval-v3` and `merge_results.py` refuses to merge contracts.

Separately, the tournament never included the closest published competitors
required by WP9.2 and by the conditional G0 decision in WP1-D014. Ports of
Causal-DRF, a FOCaL-style doubly robust functional meta-learner, and Du et al.
Wasserstein Random Forests are added in `research/sim/incumbents.py`. They are
reimplementations from published descriptions, not the authors' code, and must
be reported with that provenance. `specialized_forest`, the separate-per-block
ablation that matched or beat `odcf_composite` in essentially every first-pilot
cell, remains the primary adversary.

Two cost repairs make the noisy grid runnable: cross-fitted nuisances are now
computed once per cell rather than refitted by each method, and the MMD split
criterion scores candidate thresholds on a fixed per-node subsample of at most
96 points, since it otherwise rebuilds a full kernel per threshold and makes
tree growth quadratic in node size.

This entry does not revise the G2 verdict. It records that the first pilot
cannot support a verdict over D0 through D5 in either direction, and that the
second pilot is the one diagnostic iteration permitted under the `UNCLEAR`
label in WP9-T10.

Status: frozen simulation-design repair before the second pilot.
