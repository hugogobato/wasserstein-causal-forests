# Phase 6 preregistration: reference-effect repair and the realism audit

Status: frozen before the first decisive Phase 6 seed.
Parent contracts: `G3-MAIN-v1`, `G3-REPAIR-v1`, `G3-PHASE55-v1`.
Estimand contract: `G0-WP0-A-v1` (unchanged). Evaluation manifest `G3-EVAL-v1`
(unchanged, including the tail event, mode radius, mass floor, and the 200-row
law subset).

## Motivation

Three facts from the frozen record motivate this phase. First, on the
reference-distribution targets the user cares most about (`REF-ATE-K`,
`REF-TCATE-K`), C-WDB R3 loses to both forest baselines on D5 by roughly a
factor of four to five and to Causal-DRF on D6. A pre-registered diagnostic
(five fits at seeds 0-1, recorded in the phase report) shows the mechanism: the
boosted particle cloud under-disperses the conditional law, arm-specifically,
and any convex spread-sensitive functional inherits the attenuation. Second,
the D7 pure-shape transfer that motivated the whole law representation is
damaged by contrast regularisation, so accuracy gains on named functionals must
come from somewhere other than more shrinkage. Third, every synthetic regime so
far shares one Hermite inner shape and a standard-normal reference; the applied
study under discussion is state-by-year income distributions with an explicit
benchmark-economy reference, and no regime in the suite looks like it.

Phase 6 therefore tests four mechanisms, each isolated in one variant, plus a
realism track. Nothing here replaces the frozen tournament; every new cell sits
on the frozen main coordinates so existing rows pair seed by seed.

## The variants

* `cwdb_dr` (produces a law): the cross-fitted R3 law is unchanged; a
  doubly-robust calibration layer is added for declared functional contrasts.
  Out-of-fold law predictions (both arms) come from the selection folds of the
  cross-fitted booster itself; the propensity is a five-fold cross-fitted
  logistic regression clipped to [0.02, 0.98]; each functional's AIPW scores
  are averaged for the marginal target and Hájek-averaged within moderator
  bins for the conditional target. Functionals evaluated post hoc (skewness,
  upper tail, reference distance) stay eligible: the correction needs only the
  realised-arm value h(q(Y)), which is observed whatever h is.
* `cwdb_smooth` (produces a law): dispersion repair of the particle cloud.
  Candidate transforms are radial scaling around the cloud barycenter and
  Gaussian jitter in rescaled coordinates followed by monotone projection;
  the transform and its strength are selected on held-out energy score over an
  arm-stratified calibration split, then the model is refitted on the full
  sample. Declared candidate grid, identical in every cell: scales
  {1.0, 1.15, 1.3, 1.45}, jitter sigmas {0.0, 0.08, 0.16, 0.28} with four
  replicates per particle.
* `cwdb_krr` (produces a law): independent-arm particle boosting whose weak
  learner is kernel ridge regression (Gaussian kernel on X, median-distance
  bandwidth, 150 Nyström landmarks, ridge lambda 1.0 relative). The pilot on
  D5 at manifest coordinates but seeds outside the manifest showed the ridge
  must be strong: at lambda = 1e-2 the learner interpolates the idiosyncratic
  per-row component of the energy gradient instead of averaging it, and the
  replayed direction field is noise at test points. This is a weak-learner
  probe, not a claimant: sharing is off by construction, so any gain is
  attributable to smoother base learners and any loss is bounded evidence
  about what the weak learner must average.
* `cwdb_frl` (no law): the functional R-learner. For each functional h,
  nuisances m_h(x) = E{h(q(Y)) | X = x} (cross-fitted boosted regression) and
  e(x) as above feed the scalar R-loss; the contrast surface t_h(x) is fitted
  by the same R-loss trees as `cwdb_rmean`, with shrinkage chosen on held-out
  R-loss over two folds from {0, 50, 500}. Arm-level means are reconstructed
  from the pooled prognostic identity, exactly as `cwdb_rmean` does.

## The realism track (IC regimes)

Four regimes written to resemble the applied study discussed with collaborators:
state-year panels, income-like right-skewed inner distributions built from the
same monotone Hermite reshaping at larger shape parameters, treatment adoption
endogenous to economic conditions, and a reference distribution that is an
explicit benchmark economy rather than the standard normal.

* `IC0`: stochastic baseline, no policy effect (placebo null).
* `IC1`: EITC-like credit: raises lower quantiles, compresses left-tail shape,
  effect stronger where non-employment is high.
* `IC2`: minimum-wage-like floor: smaller location shift, strong shape
  compression concentrated at the bottom, moderate confounding.
* `IC3`: IC1's outcome surfaces under deteriorating overlap, the panel analogue
  of D9.

The moderator remains the four-bin discretisation of X_0. All oracle truth is
the same quadrature machinery, because the generative form is unchanged; only
the surfaces, the outer law, the propensities, and the reference vector move.
Methods entering the IC grids: `cwdb_v1`, `cwdb_r3_cvridge`, `cwdb_dr`,
`causal_drf`, `drf`, `pta_s`. Ten seeds, n in {500, 1000}, K = 25, M = 10.

## Grids

Track A (mechanisms): frozen main coordinates, DGPs {D0, D2, D5, D6, D7, D8},
n in {500, 1000}, K = 25, M = 10, methods {cwdb_dr, cwdb_smooth, cwdb_krr,
cwdb_frl}, ten seeds. 480 cells. Every cell pairs seed by seed with the frozen
roster rows already in `results/merged*`.

Track B (realism): IC grids, same shapes, six methods listed above, ten seeds.
4 x 6 x 2 x 10 = 480 cells.

Budgets: the C-WDB boosting budget is the frozen G3 budget; the DR and FRL
nuisance stacks use the Phase 5.5 nuisance budget and clip; the smoothing
candidate grid above; KRR bandwidth/ridge as stated. No parameter may depend
on the regime, and no candidate outside the declared grids may be tried after
seeing a decisive result.

## Frozen decision rules

A win is a seed-paired difference beyond two paired standard errors, as before.

* R1 (reference repair): `cwdb_dr` beats `cwdb_r3_cvridge` on REF-ATE-K and on
  REF-TCATE-K in D5, and does not lose either reference target to R3 in D2 or
  D8 by more than the decision multiple.
* R2 (null safety inherited): `cwdb_dr` holds mean_quantile_rmse <= 0.15 on D0
  and D2, and its D2 ratio against the best frozen baseline stays <= 1.25.
* R3 (law integrity under smoothing): `cwdb_smooth` improves
  `kernel_law_error` against R3 in at least three of the six Track A regimes,
  keeps D6 mode coverage >= 0.90, and keeps effective support >= 6 of 10.
* R4 (weak learner probe, descriptive): `cwdb_krr` against R3 on
  `kernel_law_error` and mean_quantile_rmse per regime, reported with paired
  standard errors; no gate attaches to the direction.
* R5 (functional R, descriptive): `cwdb_frl` against `cwdb_rmean`'s published
  Stage 1 numbers on the shared cells (D2, D8), and against R3 on the
  functional and reference TCATE/TATE targets of D7 and D5.
* R6 (realism audit, descriptive): whether the R3-versus-Causal-DRF ordering
  on `kernel_law_error`, mean_quantile_rmse, and both reference targets
  reproduces on the IC regimes, and whether any conclusion of Tracks A flips
  there.

Nothing in this file authorises re-running a failed cell under another seed or
adding candidates after seeing results. Negative findings are results.

## Repair log (implementation defects, disclosed)

Two defects were found and repaired, neither informed by any decisive
comparison between methods.

First, the initial income-track run failed on all 480 Track B cells before any
result existed: `_income_log_scale` had the wrong arity, the benchmark
reference vector was built with a broadcasting error, and IC2's wage-floor
mechanism was parameterised so that it lowered bottom quantiles, which is the
opposite of what a wage floor does. The surfaces were repaired (IC2 compresses
through the log-scale surface, which raises the bottom and lowers the top)
and every Track B cell ran afterwards. No failed cell produced a number that
informed the repair or any comparison.

Second, the first Track A screen showed `cwdb_frl` failing D5 catastrophically.
The recorded diagnostics traced it to a design defect, not a mechanism result:
the joint scalar shrinkage selection coupled twenty-five zero-contrast
coordinate columns with the reference column carrying all of D5's signal, and
selected lambda around 340 for the block. The repair gives each functional
column its own held-out-selected ridge strength from the same fold runs; split
gains remain unshrunk. The repair was rerun on `cwdb_frl`'s Track A cells only,
and both the defect and the repair are reported in the phase document.

Third, a cell-level audit of the first Track A screen found the cross-fitted
variants (`cwdb_dr`, `cwdb_smooth`) had been run at three selection folds
while the published R3 estimator they must sit on uses two, and their adapters
failed to forward the declared particle count, so both variants fitted laws of
M = 5 against the roster's M = 10. Neither defect touches the DR or smoothing
layers themselves; both change only the underlying law the layers sit on. Both
were repaired and the affected cells rerun before any Track A claim was
written; the screen's earlier dr/smooth rows are superseded, not pooled.

## Computational provenance note

The Track B Causal-DRF cells run the authors' original driver, which requires
the causal-clean branch of `herbps10/drf` at the pinned commit `0a1a508`. That
package was installed from source into the project-local library
`results/Rlib/causal_drf/` and is selected by the driver through
`WCF_CAUSAL_DRF_R_LIB`; the CRAN `drf` 1.3.1 used by the paper-DRF and W-DRF-T
drivers is untouched. The launcher scripts that ran Track B export the
variable; a rerun must do the same.
