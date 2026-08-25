# Preregistration: Phase 5.5 Stage 2, the `cwdb_mutau` frozen comparison

**Written before the first Stage 2 cell was executed.** Every threshold below is
either inherited verbatim from an earlier frozen document or is a scope decision
that does not depend on a Stage 2 number, because no Stage 2 number exists yet.

**Frozen manifest:** `results/manifests/phase55_stage2_manifest.json`,
checksum `29d7ca0928aa542023a091fd2ffa3943f88858fd3f193762fee7fe87ca9c4a18`
**Parent Stage 1 manifest checksum:** `4f4c789b312512ade0817ba82d1d1fd4bc9342f878b357f2f1ea5f960a0b15b7`
**Estimand contract:** `G0-WP0-A-v1`
**Manifest contract:** `G3-PHASE55-v1` (unchanged; the stage field moves from 1 to 2)

## 1. Why this stage exists

Phase 5.5 was declared as a two-stage design. Stage 1 (`research_phases/Phase 5.5
- Orthogonalized C-WDB Variants.md`, section WP5.5-E) is a mechanism screen on
four regimes, D0, D2, D7 and D8, chosen because each isolates one mechanism
(null bias, false effect, pure shape transfer, confounding). Stage 2 is the
frozen comparison against the full G3 record, and it was gated on a Stage 1 pass.
`cwdb_mutau` passed, so this stage runs.

The screen left the phase's own numerical hook for the mu/tau variant
unsatisfied. That hook requires evaluation on "D2, D3, D5, D6, D7, D8, and D9",
and the phase's decision table contains the row "`cwdb_mutau` fixes D2 but loses
D3 or D7: Reject as the main claimant". D3, D5, D6 and D9 have never been run
for this method, so the outstanding evidence is exactly the evidence that could
overturn its Stage 1 verdict. That asymmetry, not a compute limit, is the reason
for the stage: the Stage 1 screen cost 4.4 CPU-hours in total.

## 2. Scope, and what is deliberately not in it

Only `cwdb_mutau` continues. `cwdb_rmean` is contract-limited to `MEANQ-A-K` and
was retained as a benchmark rather than a claimant, so widening its regime set
cannot change a verdict. `cwdb_xmean` failed its Stage 1 mechanism screen and is
closed.

The manifest enumerates 320 cells on the frozen `main` coordinates
(n in {500, 1000}, K = 25, M = 10): the six regimes Stage 1 omitted (D1, D3, D4,
D5, D6, D9) at twenty seeds, and a seed top-up (seeds 10 through 19) on the four
screen regimes. Nothing Stage 1 already computed is recomputed. Cell keys are
content-addressed over coordinates alone, the two key sets were verified
disjoint at freeze time, and the claimant's analysis surface is the union of
this stage's merged table with the Stage 1 `cwdb_mutau` rows.

The seed top-up is in scope because every incumbent carries twenty seeds on the
`main` grid. Without it, a pooled comparison would have to restrict the
incumbents to ten seeds, which is the convention that produced a spurious
sample-size artefact in the report's Phase 5.5 section (an R3 rule-1 ratio of
1.26 at n = 500 that is 0.98 on R3's own twenty seeds).

The imbalance suite is not extended. It exists to test the X-learner's declared
advantage, and the X-learner is closed.

## 3. Comparators

PTA-S, C-WDB R3 (`cwdb_r3_cvridge`), C-WDB V1, Causal-DRF, and W-DRF-T, each on
its own frozen rows at the same coordinates and the same seeds. Causal-DRF and
DRF are the original-code reruns in `results/merged_original_causal_drf` and
`results/merged_original_drf`; the retired project-local drivers in
`results/merged` are not comparators and are not a cost denominator.

A mean-only comparator is compared only on `MEANQ-A-K` and its declared
functionals. A target a comparator cannot supply is reported as absent, never as
a loss, and a win on such a target is labelled a capability win and never
aggregated with accuracy wins.

## 4. Decision rules, inherited verbatim

Rule 1, correctness and nulls: D0 `mean_quantile_rmse` at or below 0.15; D2
`mean_quantile_rmse` at or below 0.15; D2 false-effect ratio at or below 1.25
times the best frozen baseline at the matching sample size. The ratio convention
is the one already in `research/checks/phase55_stage1_report.py` (`rule1_block`),
per sample size, not hand-pooled.

Rule 2, primary law metric: beat Causal-DRF on `kernel_law_error` in at least two
scientifically relevant mechanisms.

Rule 3, transfer: carry the advantage to at least one predeclared functional or
reference target.

Rule 4, beats the direct learner: beat PTA-S on the transferred target of rule 3.

Rule 5, no collapse: effective particle support at or above 0.6 M, and D6 mode
coverage reported alongside the squared-W2 comparator.

Rule 6, cost: median runtime at or below 60 times Causal-DRF. **Runtime is read
from Stage 2 rows only**, for the reason given in section 6.1. **The denominator
is declared here because the project contains two Causal-DRF records and the
verdict depends on which is used:** the denominator is the original-code
Causal-DRF median on the same cells. Against the retired local driver the same
`cwdb_mutau` median gives a ratio near 91, and that number is reported as a
sensitivity, not as the verdict.

## 5. The stage's own clauses, from the phase document

First, the accuracy clause frozen before decisive execution: a full-law candidate
must obtain at least one accuracy win against PTA-S on a target both methods
estimate. Passing only because PTA-S lacks a target does not count.

Second, the degradation clause from the decision table: the candidate cannot pass
the full-law gate if it improves D2 by materially degrading D3 sharing or D7
shape transfer. D3 is the separate-head-favourable regime, which is the regime
the mu/tau reparameterisation puts at risk, because it shares one tree between
the prognostic and contrast fields. "Materially" is fixed here, before the
result, as a paired loss against C-WDB R3 that crosses the frozen decision
multiple on `kernel_law_error` at either sample size.

Third, the two collapse checks the phase names for this variant and that Stage 1
already verified are carried forward unchanged: the leaf-share collapse to the
reparameterised shared tree, and the requirement that a contrast penalty must not
erase a pure shape effect on D7 when held-out energy risk selects no penalty.

## 6. Analysis conventions

The decision multiple is 2.0 paired standard errors, unchanged. Comparisons are
seed-paired; a comparison with fewer than three shared seeds is reported as
absent. A cell is (grid, dgp, n, K, M, method, seed) with arm rows averaged
inside the cell before any pooling, and regime means pool n = 500 and n = 1000
only after that. Failed cells stay in the merged table as failures; the merge
audit must return `PASS` before any number is read.

### 6.1 The `mutau` implementation changed between the stages, and only in speed

`src/wasserstein_causal_forests/cwdb/mutau.py` was modified after the Stage 1
shards were written, as part of the adversarial audit of the Stage 1 memo. The
`src` tree is not under version control, so the change cannot be diffed, and it
was therefore tested directly before this stage was launched, on three seed
top-up cells at coordinates Stage 1 also ran (D0, n = 500, seeds 10 to 12).

Every accuracy quantity reproduced inside the Stage 1 spread:
`mean_quantile_rmse` 0.0980, 0.1044, 0.1089 against a Stage 1 range of 0.1003 to
0.1163; `kernel_law_error` inside the Stage 1 range; `diagnostic_train_risk`
0.0220, 0.0248, 0.0230 against a Stage 1 range of 0.0188 to 0.0258; selected
contrast shrinkage 0 and boosting steps 100 in both, exactly as Stage 1. Runtime
did not: 34.6 to 35.1 seconds against a Stage 1 median of 165.7 at the same
coordinates, a factor near five.

The conclusion drawn, and frozen here, is that the change is a performance
change and the two stages' accuracy rows are poolable. Cost rows are not. Every
runtime, and therefore the rule 6 ratio, is computed from Stage 2 rows alone, so
that the numerator and the denominator are measured under one implementation and
one machine load. The Stage 1 cost table stands as the record of what the Stage 1
memo reported and is not restated.

This calibration was run before the stage and is reported whatever it showed. Had
an accuracy metric moved outside the Stage 1 spread, the declared response was to
rerun the four screen regimes at all twenty seeds under current code rather than
pool across implementations.

## 7. What this stage cannot establish

It cannot establish anything about `cwdb_rmean` or `cwdb_xmean` on the new
regimes, because they are not run. It cannot establish uncertainty coverage for
any C-WDB variant, because none has an interval construction and the estimand
contract forbids substituting a posterior-draw quantity. It cannot establish
anything about grid resolutions other than K = 25 or particle counts other than
M = 10, both of which are fixed by the manifest. It cannot license any claim
about the joint law of individual treatment effects, which is not identified.

## 8. Prohibited moves

No threshold in section 4, 5 or 6 may be restated after a Stage 2 number is
seen. If a convention has to change, the change is recorded here with the reason,
both conventions are reported, and the verdict under each is stated. A regime may
not be dropped from a summary because it is unfavourable; the manifest's regime
set is the reporting set.
