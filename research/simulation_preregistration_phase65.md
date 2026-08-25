# Phase 6.5 preregistration: adversarial controls, ingredient ablation, zero inflation

Status: frozen before the first decisive Phase 6.5 seed.
Parent contracts: `G3-PHASE6-v1` (Phase 6 record untouched and reused).
Estimand contract: `G0-WP0-A-v1` (unchanged). Evaluation manifest `G3-EVAL-v1`
extended with one field, `zero_mass_tolerance = 0.05`; every other entry is
inherited unchanged.
Manifest: `results/manifests/phase65_manifest.json`, 900 cells, checksum
`4e28d308ca99cde4c81379524fc4492a15b38f029b449899b0a307b6c0ace110`.

## Motivation

The Phase 6 Track B collapse (forests manufacturing reference effects on
income-shaped panels, including on the placebo) currently supports a weaker
sentence than the one the project wants to defend. Three confounds are open:
the incumbents' bandwidth details were frozen on normal-scale outcomes and no
transform control was run; five realism ingredients switch on together so no
ingredient owns the reversal; and nothing measures whether the deficit shrinks
with n as the forest atom bank grows. Separately, zero inflation is the
field-standard complication for exactly the applied outcomes targeted, and it
is loaded against the proposed family: a point mass is trivial for an
empirical-law forest and hard for a ten-particle cloud.

## The method variants

* `causal_drf_log`, `drf_log` (produce laws): the incumbent driver runs
  unmodified on `log(Q - floor)`, with the frozen floor rule
  `floor = min(Q_train) - 0.05 sd(Q_train)` because income panels contain
  non-positive coordinates. The monotone composite map sends the atom bank back
  onto byte-for-byte the original-scale training sample, so the control moves
  only the kernel and splitting geometry; evaluation is entirely on the
  original scale against original-scale truth.
* `causal_drf_retn` (produces a law): the incumbent driver runs unmodified on
  `Q / m` for a frozen multiplier m from `{0.25, 0.5, 1, 2, 4}`. With
  `response.scaling = FALSE` and the data-driven median-heuristic bandwidth,
  this is algebraically identical to running at bandwidth m times the default:
  every kernel quantity is homogeneous of degree one in the outcome scale.
* `cwdb_zipt` (produces a law): per-arm logistic classifiers (`C = 1.0`,
  lbfgs, 2000 iterations) for P(degenerate component | X, arm); the cross-fitted
  R3 booster (candidates `{0, 50, 500}`, two selection folds, pooled
  initialisation, arm shrinkage 5.0) refitted on positive-component rows only;
  assembled law `(1 - phat_a(x)) delta_0 + phat_a(x) * cloud` with uniform
  particle mass. Constant-rate fallbacks when an arm has no or all degenerate
  rows, recorded as diagnostics. This is a mixture assembly in law space, not a
  relabelling of posterior draws.

## The bandwidth-selection rule

For each (IC regime, n) cell, pilot fits at seeds **100 and 101** (outside
every decisive range) split each sample in half by a uniform draw from
`default_rng(seed + 500000)`; the fitting half feeds one causal-drf fit per
candidate multiplier, and the candidate's score is the mean energy risk of its
arm-law estimate against the held-out half's realised outcome distributions, a
proper score. The argmin over the grid is frozen into
`results/manifests/phase65_bandwidth_selection.json` before any decisive retune
cell runs. Oracle truth never enters selection. No candidate outside the grid
may be tried after seeing a decisive result.

## New regimes

Track D, one IC1 ingredient off at a time, base point the frozen IC1 record:
`DAskew` (inner shape at zero), `DArand` (propensity identically one half),
`DAunit` (all surfaces divided by the frozen divisor 1.274, the measured IC1
population standard deviation), `DAref` (standard-normal reference), `DAdim`
(two covariates carrying the remapped surfaces).

Track E, unit-level two-part form: with probability `p_a(x)` a panel
observation's whole outcome distribution is strictly positive income-like
(Hermite form at location base 4.80, verified to clear zero over the cube and
the twelve-node latent range); otherwise it is degenerate at zero. The
conditional law mixes the two components with covariate-dependent weight, so
truth node weights become `(n, J)` matrices; the metric layer's shared-weight
paths are unchanged. Regimes: `ZI0` placebo null; `ZI1` participation effect;
`ZI2` intensity effect; `ZI3` both under deteriorating overlap.

## Grids

| Track | Grid id | DGPs | Methods | n | Seeds | Cells |
|---|---|---|---|---|---|---|
| C controls | `c_controls` | IC0-IC3 | `causal_drf_log`, `drf_log`, `causal_drf_retn` | 500, 1000 | 0-9 | 240 |
| C scaling | `c_scaling` | IC0, IC1 | R3, `causal_drf`, DRF | 2000, 4000 | 0-4 | 60 |
| D ablation | `d_ablation` | DA x5 | R3, `cwdb_dr`, `causal_drf`, DRF | 1000 | 0-9 | 200 |
| E zero-inflation | `e_zi` | ZI0-ZI3 | R3, `cwdb_dr`, `cwdb_zipt`, `causal_drf`, DRF | 500, 1000 | 0-9 | 400 |

PTA-S is excluded by design on every track: it produces no law, cannot carry a
zero-mass statement, and its pre-declared advantage is already documented.
Adding it later requires a new frozen manifest, not an afterthought. R3 and
`cwdb_dr` numbers on the IC regimes are reused from the frozen Phase 6 merge,
never rerun.

## New metrics

`zero_mass_abs_error` per arm: RMSE of the fitted law's degenerate-component
mass against truth. An atom represents the degenerate law when its largest grid
coordinate is at most 0.05 (primary, loose, fair to boosted averages); the
exact-atom fraction at tolerance 1e-9 travels as a row detail diagnostic.
`mass_contrast_rmse`: moderator-binned RMSE of the arm contrast in implied
degenerate probability. On regimes without a degenerate component both rows are
recorded `not_applicable`.

## Frozen decision rules

A win is a seed-paired difference beyond two paired standard errors, as before.

* **RC1 (transform control).** Primary comparison `causal_drf_log` versus
  `cwdb_r3_cvridge` on IC1 and IC3, metrics `mean_quantile_rmse` and
  `REF-TCATE-K`. If the paired difference is within two paired SEs, or the
  error ratio falls below 1.25, the Phase 6 collapse is attributed primarily to
  outcome parameterisation and downgraded in writing. If Causal-DRF-on-log
  remains worse beyond two paired SEs by a factor of two or more, the collapse
  is recorded as robust to the transform.
* **RC2 (bandwidth control).** Same thresholds for `causal_drf_retn`.
* **RC3 (scaling, descriptive).** Sign and rough magnitude of the forest
  deficit trend from n = 1000 through 4000; no gate.
* **RD1 (ablation).** For each DA regime, the seed-paired R3-minus-Causal-DRF
  difference on `kernel_law_error`, `mean_quantile_rmse`, and both reference
  targets against the frozen IC1 base point. The write-up's generalisation
  claim must name whichever single ingredient reproduces at least half of the
  IC1 gap; if none does, the attribution is an interaction and must be written
  as one.
* **RE-P1 (registered prediction, falsifiable).** R3 loses to plain DRF on
  `zero_mass_abs_error` in at least three of four ZI regimes. If forests lose
  the mass comparison too, the mechanism story extends; either direction is a
  result.
* **RE-P2 (registered prediction).** `cwdb_zipt` beats R3 on
  `zero_mass_abs_error` and `mass_contrast_rmse` everywhere it runs.
* **RE1 (two-part gate).** PASS requires, in ZI1-ZI3, wins against R3 on both
  zero-mass metrics beyond two paired SEs and `kernel_law_error` no more than
  ten percent worse than R3. FAIL means the two-part idea is reported as a
  negative finding with its mechanism and the contribution claim is scoped to
  strictly-positive outcomes.
* **RE2 (descriptive).** Forest standing on ZI recorded with paired SEs.

Nothing here authorises re-running a failed cell under another seed or adding
candidates after seeing results. Negative findings are results.

## Execution

Seventeen self-contained Colab notebooks under `colab/phase65_shards/`
(twelve `core65`, five `forest65`), balanced at roughly 85 single-threaded
reference minutes each, generated by
`research/checks/phase65_make_colab_notebooks.py`. Thread pinning precedes any
import; shards checkpoint their execution log after every cell; failed cells
are preserved with their reason. Local execution uses
`python research/run_phase65.py freeze|run|merge` with the same discipline; the
decisive retune cells refuse to run without the frozen selection document.
