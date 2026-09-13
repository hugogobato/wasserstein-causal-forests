# WCF sensitivity study: final decision on K and M

This document closes the work packages in `report/sensitivity_and_dgp_experiments.md`.
Every number below comes from the isolated study under `results/wcf_sensitivity/`
with contract `WCF-SENSITIVITY-v1`; the historical Phase 6 and 6.5 results were
never touched. The final estimator is the `cwdb_dr` adapter (labelled WCF in the
paper): three selection folds, contrast candidates `[0, 50, 500]`, logistic
propensity factory, common-grid evaluation attached.

**Primary setting: K = 25, M = 10 (retained).** The pre-fixed stability rule
failed, the corrected common-grid and interior evidence gives no decisive reason
to switch, and K=49/M=10 is reported as a tail-fidelity robustness alternative
with the disclosures listed under Decision below. The machine-readable summary
is `final_decision.json`.

## What the pre-fixed rule found

The stability rule failed for the incumbent primary `(K, M) = (25, 10)`. The
rule required every one-factor change to stay within 10 percent of the primary
WCF reference-TCATE and law error in IC1 and IC3 and no larger grid or particle
count to improve both targets by more than two between-seed standard errors in
both regimes. On seeds 0--9 the native reference-TCATE in IC1 improves 15.5
percent at K=5 and 14.1 percent at K=49 relative to K=25, while K=5 is 19.1
percent worse on native law error; no larger pair improved both targets in both
regimes, so the rule returned `retain_primary = false` with no automatic
selection. The complete table is preserved in `analysis/decision.json`,
`analysis/one_factor_k.csv`, and `analysis/one_factor_m.csv`.

## Why the native table is not decisive, and what the common grid adds

Native metrics change the estimand when K changes, so the one-factor K table
cannot separate grid approximation from estimator error. The study therefore
evaluated every fit against a common 199-level target, and separately against an
interior 199-level target restricted to levels 0.1--0.9, which avoids endpoint
extrapolation for every proposed K. The interior diagnostic is the sharper
test, and it changes the reading of the result.

On seeds 0--9, K=49 minus K=25 at M=10:

| Regime | Full-range LAW | Interior LAW | Full-range REF-TCATE | Interior REF-TCATE |
|---|---|---|---|---|
| IC0 | -46.4% | -3.3% | +4.2% | +8.2% |
| IC1 | -43.4% | +0.5% | -16.4% | -15.0% |
| IC2 | -43.8% | -3.7% | -3.5% | +6.4% |
| IC3 | -41.8% | +1.1% | +5.7% | +3.9% |

On the fresh-seed confirmation (seeds 20--39):

| Regime | Full-range LAW | Interior LAW | Full-range REF-TCATE | Interior REF-TCATE |
|---|---|---|---|---|
| IC0 | -46.5% | -1.7% | -1.2% | -0.8% |
| IC1 | -43.7% | -3.4% | -7.9% | -2.6% |
| IC2 | -44.1% | -1.3% | -10.5% | -0.6% |
| IC3 | -42.8% | -1.3% | +0.3% | +0.8% |

The full-range law gain of roughly 42 to 46 percent is therefore a tail-coverage
effect: it comes from the levels below 0.1 and above 0.9, where a K=25 fit must
be continued as a constant outside its fitted support while a K=49 fit reaches
further into the tail. On the interior levels the two grids are statistically
indistinguishable (all paired |t| below 1.4 on both samples). The reference-TCATE
gain is confined to IC1 and is borderline (t = -2.4 on seeds 0--9, -2.2 on the
confirmation); IC0, IC2, and IC3 show no significant difference on the interior
target. The mean-quantile gains are regime-dependent: robust in the IC0 placebo
(about -42 percent on both the full-range and interior targets, t below -3) and
smaller and mostly not significant in IC1--IC3 (full-range -16, -13, and -3
percent; interior -8, -5, and -0.3 percent).

## Interaction, cost, and the one adverse target

The factorial interaction tables for IC1 and IC3 are in
`analysis/factorial_ic1_ic3.csv`; no pair dominates the native table, and the
non-monotone interaction is consistent with an estimand that moves with K. The
cost of K=49 is about 1.6 times K=25 at the same M (median 524 s against 330 s
per cell), and M=25 costs roughly four to eight times M=10 for at most 1.2
percent common-grid law error at K=25 and 1.5 percent across the tested grids,
so M=10 remains the right resolution. One target moves the other way: native `REF-ATE-K` in IC3 is 20.8
percent worse at K=49 (t = 2.85 on the confirmation sample), and the full-range
common `REF-ATE-COMMON199` is 12.1 percent worse (t = 1.98). If the marginal
reference effect under poor overlap is a primary target, K=25 is preferable.

## Decision

The final primary setting is **K = 25, M = 10**, retained after the corrected
sensitivity review. The pre-fixed stability rule failed for it, so the complete
sensitivity table is reported; the corrected common-grid and interior evidence
does not establish that switching to K=49 would improve the scientific targets,
while K=49 costs 1.6 times more and loses on the IC3 marginal reference effect.
M=10 is retained because M=25 costs four to eight times more for at most 1.2
percent common-grid law error at K=25 and 1.5 percent across the tested grids,
and M=5 does not improve the interior targets.

K = 49, M = 10 is reported as a tail-fidelity robustness alternative: it
improves the full-range law error by roughly 42 to 46 percent through better
coverage of the extreme quantile levels, and improves the IC1 reference-TCATE.
It must be presented with the disclosures that the interior law and
reference-TCATE targets are statistically indistinguishable from K=25, that the
IC3 marginal reference effect is 20.8 percent worse, and that it costs 1.6
times more per cell. If the paper later emphasises full-range law fidelity,
including the tails, K=49 is the better choice; the interior evidence gives no
reason to prefer it otherwise.

In every table, `(25, 10)` remains the pre-registered comparator and the study is
reported as a sensitivity assessment on ten plus twenty replications, not as an
equivalence test.

## Assignment mechanisms and propensity sensitivity

WCF against the two forest baselines on `REF-TCATE-K` at n = 1000 (mean over
ten seeds, lower is better):

| Regime | WCF | Causal-DRF | DRF |
|---|---|---|---|
| SYM-RANDOM | 0.0286 | 0.0313 | 0.0275 |
| SYM-LIN | 0.0350 | 0.0392 | 0.0340 |
| SYM-NL | 0.0417 | 0.0814 | 0.0736 |
| SYM-MU | 0.0287 | 0.0461 | 0.0406 |

WCF wins clearly under the two nonlinear-propensity regimes (`SYM-NL`,
`SYM-MU`), is at parity under the linear propensity, and is at parity with DRF
under the calibration control. The `SYM-ALIGN` against `SYM-IRREL` contrast
shows no significant alignment effect at ten seeds (reference-TCATE p = 0.48 at
n = 500 and p = 0.88 at n = 1000; law-error p = 0.11 and p = 0.72), so prognosis
alignment is not detectable at this sample size.

The propensity-model sensitivity ran on `SYM-NL` and `SYM-MU` with the logistic
default, the random-forest factory, the default gradient-boosting factory, and
the oracle diagnostic. `REF-TCATE-K`, n = 1000:

| Regime | Logistic | Random forest | Gradient boosting | Oracle |
|---|---|---|---|---|
| SYM-NL | 0.0417 | 0.0383 | 0.1471 | 0.0339 |
| SYM-MU | 0.0287 | 0.0291 | 0.1057 | 0.0261 |

The random-forest factory is statistically indistinguishable from logistic
(paired differences -8.1 to +1.9 percent, all |t| < 1.4), so WCF is robust to a
competent flexible propensity. The default gradient-boosting factory is three to
four times worse; a dedicated calibration diagnostic showed its out-of-fold
predictions reach 0.001 and 0.998 with 2 to 12 percent clipped, and its RMSE
against the true propensity is 0.22 to 0.23, against 0.05 to 0.16 for logistic
and 0.09 to 0.10 for the random forest. That result is therefore a statement
about an overconfident factory, not evidence against flexible propensities, and
is reported as a documented negative finding (persisted in
`analysis/propensity_calibration_diagnostic.json`). The oracle propensity improves on
logistic by up to 18.7 percent in SYM-NL, which bounds the efficiency left on
the table by the misspecified logistic model.

## Placebo checks on the null companions

The null companion regimes have exactly zero effects, so every entry is an
absolute false-effect error (mean over ten seeds, lower is better):

| Target | n | WCF | Causal-DRF | DRF |
|---|---|---|---|---|
| TATE `grid_mean` | 500 | 0.0329 | 0.1466 | 0.1064 |
| TATE `grid_mean` | 1000 | 0.0295 | 0.1026 | 0.0736 |
| TCATE `grid_mean` | 500 | 0.0615 | 0.1542 | 0.1183 |
| TCATE `grid_mean` | 1000 | 0.0466 | 0.1098 | 0.0851 |
| `REF-ATE-K` | 500 | 0.0179 | 0.0215 | 0.0197 |
| `REF-ATE-K` | 1000 | 0.0135 | 0.0140 | 0.0138 |
| `REF-TCATE-K` | 500 | 0.0449 | 0.0504 | 0.0556 |
| `REF-TCATE-K` | 1000 | 0.0352 | 0.0476 | 0.0519 |

The placebo errors are small relative to the effect-regime errors and WCF is
the best-calibrated method on every target, which supports the placebo claims
the paper can make.

## Audit and reproducibility

The primary stage declares 880 cells, all observed, all successful, no
duplicate keys, no test-seed violations, and no missing law-metric families
(`merged/merge_audit.json`). The full study contains 1,520 distinct cells:
primary 880, null companions 360, factorial corners 80 (74 local and 6 Colab),
fresh-seed confirmation 160, and random-forest propensity 40. Every stage was
merged with its own audit; the combined parquet is
`merged/wcf_sensitivity_combined.parquet` and its per-stage inventory is
`merged/combined_audit.json`.

The artifacts are: the frozen primary manifest (`manifest.json`), the stage
manifests under `confirm/`, `factorial/`, `flex_rf/`, and `nulls/`, the shard
sidecars that pin the manifest checksum and estimator source hash
`c105c0aa...`, the execution logs under each stage's `logs/`, the merged
parquets under each stage's `merged/`, the analysis tables and figures under
`analysis/`, the design checks under `design_checks.json`, and the Colab
notebooks under `colab/wcf_sensitivity_shards/`,
`colab/wcf_sensitivity_confirm_shards/`, `colab/wcf_sensitivity_factorial_shards/`,
and `colab/wcf_sensitivity_flex_rf_shards/`. The four pure-Python Colab batches
embed the exact revised source, so they remain runnable without the working
tree.

## Limitations

The one-factor and SYM results rest on ten replications; the fresh-seed
confirmation uses twenty. All uncertainty statements are paired Monte Carlo
standard errors over seeds and are not standard errors of a test-row RMSE. The
factorial covers only IC1 and IC3, as the pre-registration required. The
gradient-boosting propensity result is factory-specific. No theoretical
consistency or efficiency claim follows from this study.
