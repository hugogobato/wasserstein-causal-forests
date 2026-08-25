# Phase G3 gate memo: preregistered simulation tournament

**Verdict:** `NOT-GO`
**Rules passed:** 5 of 6
**Merged results checksum:** `a871fd7b6ab72544e1f0f317c16cb18551759b575899188c0bfb1d048f9a4a69`
**Code revision:** `63651f1b650ba00f6a69f98bd79b8b8708ada257`
**Manifest checksum:** `5a672a60c382091c1fbc994a859c2a2c06ed60f3be42e73df67180b06a924113`
**Preregistration:** `research/simulation_preregistration.md`

Every threshold applied below was frozen in the preregistration before the first decisive seed. Nothing in this memo re-chooses a metric, a regime, or a cutoff.

## 1. Execution and reconciliation

The manifest declares 4110 cells; 4110 produced rows, for a total of 97650 result rows. Failed cells: 0. Merge audit status: `PASS`.

Failure rate by method:

| Method | Cells | Failed | Rate |
|---|---|---|---|
| `causal_drf` | 670 | 0 | 0.0% |
| `cwdb_v0` | 670 | 0 | 0.0% |
| `cwdb_v1` | 870 | 0 | 0.0% |
| `cwdb_v1_noshrink` | 40 | 0 | 0.0% |
| `pta_f` | 200 | 0 | 0.0% |
| `pta_s` | 630 | 0 | 0.0% |
| `sqw2_booster` | 360 | 0 | 0.0% |
| `wdrft` | 670 | 0 | 0.0% |

## 2. The adversarial reading, stated first

C-WDB-v1 is beaten by more than two paired standard errors in the following regimes and targets. These are the cells a sceptical reader should weigh first, and averaging them into an overall score would hide them.

| Regime | Target | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D6 | `kernel_law_error` | `causal_drf` | 0.0191 | 0.00945 | +0.009648 | 0.000452 | 0% | **comparator** |
| D5 | `TCATE-K-grid_sd` | `causal_drf` | 0.04216 | 0.02754 | +0.01462 | 0.00241 | 10% | **comparator** |
| D5 | `TCATE-K-grid_skewness` | `causal_drf` | 0.01507 | 1.871e-17 | +0.01507 | 0.0032 | 0% | **comparator** |
| D6 | `TCATE-K-grid_sd` | `causal_drf` | 0.02165 | 0.01455 | +0.007097 | 0.00129 | 10% | **comparator** |
| D6 | `TCATE-K-grid_skewness` | `causal_drf` | 1.535e-16 | 3.476e-17 | +1.187e-16 | 2.42e-17 | 0% | **comparator** |
| D7 | `TCATE-K-grid_sd` | `causal_drf` | 0.02575 | 0.01755 | +0.008197 | 0.00181 | 20% | **comparator** |
| D5 | `REF-TCATE-K` | `causal_drf` | 0.09606 | 0.02302 | +0.07303 | 0.00455 | 0% | **comparator** |
| D6 | `REF-TCATE-K` | `causal_drf` | 0.07463 | 0.04116 | +0.03347 | 0.00609 | 5% | **comparator** |
| D7 | `REF-TCATE-K` | `causal_drf` | 0.02893 | 0.02303 | +0.005893 | 0.00257 | 30% | **comparator** |
| D5 | `REF-ATE-K` | `causal_drf` | 0.09373 | 0.01792 | +0.07581 | 0.00482 | 0% | **comparator** |
| D6 | `REF-ATE-K` | `causal_drf` | 0.0485 | 0.02207 | +0.02644 | 0.00741 | 20% | **comparator** |
| D7 | `TCATE-K-grid_mean` | `pta_s` | 0.0341 | 0.02231 | +0.01179 | 0.00407 | 25% | **comparator** |

Structural limits of this tournament, restated from the preregistration:

1. PTA-F runs only at $K=5$, because its cost accelerates in the target dimension; no conclusion about it at the working resolution is available. 2. The `Uncertainty usable` claim row is **not evaluated**: C-WDB has no interval construction, and contract Section 4 forbids substituting a posterior-draw quantity. 3. No claim about Causal-DRF's band coverage relative to the published two-forest benchmark is made, per the Phase 4 limitation. 4. Grid-resolution conclusions hold for $K\in\{5,25,49\}$ only.

## 3. Gate rules

| Rule | Statement | Result |
|---|---|---|
| `rule_1_correctness` | passes D0 through D2 correctness and null checks | **FAIL** |
| `rule_2_law_advantage` | beats Causal-DRF on the primary law metric in at least two scientifically relevant mechanisms | PASS |
| `rule_3_transfer` | transfers the advantage to at least one predeclared functional or reference target | PASS |
| `rule_4_beats_direct_learner` | beats PTA-S on the transferred target of rule 3 | PASS |
| `rule_5_no_collapse` | no systematic particle collapse or excessive projection | PASS |
| `rule_6_cost` | compute cost commensurate with the gain | PASS |

### Rule 1, correctness and nulls

D0 `mean_quantile_rmse` = 0.1091; D2 = 0.1466 against a best baseline of 0.0545, a false-effect ratio of 2.69.

### Rule 2, primary law metric against Causal-DRF

Wins: 3 of 2 required, in ['D1', 'D5', 'D7'].

| Regime | Target | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D1 | `kernel_law_error` | `causal_drf` | 0.01051 | 0.01883 | -0.00832 | 0.000574 | 100% | claimant |
| D5 | `kernel_law_error` | `causal_drf` | 0.01995 | 0.05546 | -0.03551 | 0.00304 | 100% | claimant |
| D6 | `kernel_law_error` | `causal_drf` | 0.0191 | 0.00945 | +0.009648 | 0.000452 | 0% | **comparator** |
| D7 | `kernel_law_error` | `causal_drf` | 0.01007 | 0.0251 | -0.01503 | 0.00105 | 100% | claimant |

### Rules 3 and 4, transfer to a causal functional

Against Causal-DRF: 5 winning targets (['TCATE-K-grid_mean@D6', 'TCATE-K-grid_upper_tail_mean@D6', 'TCATE-K-grid_mean@D7', 'TCATE-K-grid_skewness@D7', 'TCATE-K-grid_upper_tail_mean@D7']). Against PTA-S on those same targets: 3 wins, of which 0 are accuracy wins on a target PTA-S also estimates and 3 are capability wins on a target it cannot estimate at all.

| Regime | Target | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D5 | `TCATE-K-grid_mean` | `causal_drf` | 0.03798 | 0.04763 | -0.009652 | 0.00566 | 60% | tie |
| D5 | `TCATE-K-grid_sd` | `causal_drf` | 0.04216 | 0.02754 | +0.01462 | 0.00241 | 10% | **comparator** |
| D5 | `TCATE-K-grid_skewness` | `causal_drf` | 0.01507 | 1.871e-17 | +0.01507 | 0.0032 | 0% | **comparator** |
| D5 | `TCATE-K-grid_upper_tail_mean` | `causal_drf` | 0.05536 | 0.05611 | -0.0007553 | 0.0061 | 45% | tie |
| D6 | `TCATE-K-grid_mean` | `causal_drf` | 0.1674 | 0.1962 | -0.02886 | 0.011 | 70% | claimant |
| D6 | `TCATE-K-grid_sd` | `causal_drf` | 0.02165 | 0.01455 | +0.007097 | 0.00129 | 10% | **comparator** |
| D6 | `TCATE-K-grid_skewness` | `causal_drf` | 1.535e-16 | 3.476e-17 | +1.187e-16 | 2.42e-17 | 0% | **comparator** |
| D6 | `TCATE-K-grid_upper_tail_mean` | `causal_drf` | 0.1634 | 0.1941 | -0.03069 | 0.0108 | 70% | claimant |
| D7 | `TCATE-K-grid_mean` | `causal_drf` | 0.0341 | 0.06924 | -0.03514 | 0.00594 | 95% | claimant |
| D7 | `TCATE-K-grid_sd` | `causal_drf` | 0.02575 | 0.01755 | +0.008197 | 0.00181 | 20% | **comparator** |
| D7 | `TCATE-K-grid_skewness` | `causal_drf` | 0.01207 | 0.07324 | -0.06117 | 0.00364 | 100% | claimant |
| D7 | `TCATE-K-grid_upper_tail_mean` | `causal_drf` | 0.04147 | 0.07035 | -0.02888 | 0.00617 | 90% | claimant |
| D5 | `REF-TCATE-K` | `causal_drf` | 0.09606 | 0.02302 | +0.07303 | 0.00455 | 0% | **comparator** |
| D6 | `REF-TCATE-K` | `causal_drf` | 0.07463 | 0.04116 | +0.03347 | 0.00609 | 5% | **comparator** |
| D7 | `REF-TCATE-K` | `causal_drf` | 0.02893 | 0.02303 | +0.005893 | 0.00257 | 30% | **comparator** |
| D5 | `REF-ATE-K` | `causal_drf` | 0.09373 | 0.01792 | +0.07581 | 0.00482 | 0% | **comparator** |
| D6 | `REF-ATE-K` | `causal_drf` | 0.0485 | 0.02207 | +0.02644 | 0.00741 | 20% | **comparator** |
| D7 | `REF-ATE-K` | `causal_drf` | 0.02016 | 0.01652 | +0.00364 | 0.00269 | 35% | tie |

| Regime | Target | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D6 | `TCATE-K-grid_mean` | `pta_s` | 0.1674 | 0.1603 | +0.007053 | 0.00929 | 45% | tie |
| D6 | `TCATE-K-grid_upper_tail_mean` | `pta_s` | 0.1995 | _no estimate_ | | | | claimant, on capability |
| D7 | `TCATE-K-grid_mean` | `pta_s` | 0.0341 | 0.02231 | +0.01179 | 0.00407 | 25% | **comparator** |
| D7 | `TCATE-K-grid_skewness` | `pta_s` | 0.01523 | _no estimate_ | | | | claimant, on capability |
| D7 | `TCATE-K-grid_upper_tail_mean` | `pta_s` | 0.05252 | _no estimate_ | | | | claimant, on capability |

### Rule 5, particle collapse

D6 mode coverage 0.99; effective particle support 10 of $M$, a fraction of 1. The squared-$W_2$ comparator, which removes the repulsion term, reaches mode coverage 0.2303125.

### Rule 6, cost

Median runtime 13.3 s against Causal-DRF's 1.172 s, a ratio of 11.3 against a ceiling of 60.

| Method | Median runtime (s) | Max runtime (s) | Median peak RSS (MB) |
|---|---|---|---|
| `causal_drf` | 1.17 | 35.4 | 75.2 |
| `cwdb_v0` | 8.93 | 42.2 | 0 |
| `cwdb_v1` | 13.3 | 149 | 0 |
| `cwdb_v1_noshrink` | 26.2 | 32.2 | 0 |
| `pta_f` | 96.1 | 113 | 0 |
| `pta_s` | 28.1 | 77.4 | 0 |
| `sqw2_booster` | 0.996 | 2.3 | 0 |
| `wdrft` | 1.33 | 36.2 | 214 |

## 4. Mechanism ablations

### Repulsion, against the squared-$W_2$ booster

| Regime | Target | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D1 | `kernel_law_error` | `sqw2_booster` | 0.0164 | 0.01387 | +0.002523 | 0.000221 | 0% | **comparator** |
| D1 | `mode_coverage` | `sqw2_booster` | 1 | 1 | +0 | 0 | 0% | tie |
| D1 | `arm_energy_risk` | `sqw2_booster` | 0.07175 | 0.1413 | -0.06954 | 0.000636 | 100% | claimant |
| D6 | `kernel_law_error` | `sqw2_booster` | 0.03681 | 0.197 | -0.1602 | 0.00122 | 100% | claimant |
| D6 | `mode_coverage` | `sqw2_booster` | 0.9853 | 0.2303 | -0.755 | 0.00499 | 100% | claimant |
| D6 | `arm_energy_risk` | `sqw2_booster` | 0.1128 | 0.6818 | -0.569 | 0.00227 | 100% | claimant |

### Arm-shared localisation, against C-WDB-v0

| Regime | Target | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D3 | `mean_quantile_rmse` | `cwdb_v0` | 0.259 | 0.291 | -0.03199 | 0.00356 | 100% | claimant |
| D3 | `kernel_law_error` | `cwdb_v0` | 0.01325 | 0.01333 | -7.574e-05 | 0.000319 | 60% | tie |
| D4 | `mean_quantile_rmse` | `cwdb_v0` | 0.1765 | 0.2777 | -0.1012 | 0.00322 | 100% | claimant |
| D4 | `kernel_law_error` | `cwdb_v0` | 0.008894 | 0.01146 | -0.002571 | 0.000158 | 100% | claimant |

### Causal regularisation, against `arm_shrinkage = 0`

| Regime | Target | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D2 | `mean_quantile_rmse` | `cwdb_v1_noshrink` | 0.1266 | 0.1605 | -0.03384 | 0.00145 | 100% | claimant |
| D2 | `kernel_law_error` | `cwdb_v1_noshrink` | 0.009998 | 0.01096 | -0.0009589 | 0.000136 | 100% | claimant |
| D8 | `mean_quantile_rmse` | `cwdb_v1_noshrink` | 0.1741 | 0.2061 | -0.032 | 0.00271 | 100% | claimant |
| D8 | `kernel_law_error` | `cwdb_v1_noshrink` | 0.01517 | 0.01625 | -0.001086 | 0.000298 | 70% | claimant |

### Finite-particle approximation

| Regime | $M$ | Excess energy risk | SE | Replications |
|---|---|---|---|---|
| D1 | 2 | 0.08968 | 0.00114 | 20 |
| D1 | 5 | 0.05837 | 0.000872 | 20 |
| D1 | 10 | 0.05076 | 0.000775 | 20 |
| D1 | 25 | 0.04681 | 0.000586 | 20 |
| D6 | 2 | 0.1063 | 0.00222 | 20 |
| D6 | 5 | 0.08514 | 0.001 | 20 |
| D6 | 10 | 0.06965 | 0.00093 | 20 |
| D6 | 25 | 0.06511 | 0.000988 | 20 |

## 5. The favourable reading

Where C-WDB-v1 does win on a preregistered comparison, it wins here:

| Regime | Target | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D1 | `kernel_law_error` | `causal_drf` | 0.01051 | 0.01883 | -0.00832 | 0.000574 | 100% | claimant |
| D5 | `kernel_law_error` | `causal_drf` | 0.01995 | 0.05546 | -0.03551 | 0.00304 | 100% | claimant |
| D7 | `kernel_law_error` | `causal_drf` | 0.01007 | 0.0251 | -0.01503 | 0.00105 | 100% | claimant |
| D6 | `TCATE-K-grid_mean` | `causal_drf` | 0.1674 | 0.1962 | -0.02886 | 0.011 | 70% | claimant |
| D6 | `TCATE-K-grid_upper_tail_mean` | `causal_drf` | 0.1634 | 0.1941 | -0.03069 | 0.0108 | 70% | claimant |
| D7 | `TCATE-K-grid_mean` | `causal_drf` | 0.0341 | 0.06924 | -0.03514 | 0.00594 | 95% | claimant |
| D7 | `TCATE-K-grid_skewness` | `causal_drf` | 0.01207 | 0.07324 | -0.06117 | 0.00364 | 100% | claimant |
| D7 | `TCATE-K-grid_upper_tail_mean` | `causal_drf` | 0.04147 | 0.07035 | -0.02888 | 0.00617 | 90% | claimant |

## 6. Verdict

**`NOT-GO`**

Rules not met:

- `rule_1_correctness`: passes D0 through D2 correctness and null checks

Under the Phase G3 decision list, C-WDB does not return `GO`. The choice among `PIVOT`, `INCREMENTAL-ONLY`, and `KILL` is argued in Section 7.

The independent recomputation in `research/checks/g3_gate_flags.py` returns `NOT-GO` with 0 disagreements against the analysis code.

## 7. Interpretation and next step

_This section is argued by hand against the tables above. Every number in it
appears in those tables or in `results/merged/analysis_payload.json`._

**Verdict: `NOT-GO`, and the recommendation is `PIVOT`, not `KILL`.**

### What the tournament established

Two mechanisms behaved exactly as the theory predicts, and both are the
project's own claims rather than borrowed ones.

The proper-score repulsion works. On D6, whose outer law is a two-component
mixture, C-WDB-v1 reaches mode coverage 0.990 with the particle weights'
participation ratio at 10.00 of $M=10$, meaning no collapse whatever. The
squared-$W_2$ booster, which is the same tree machinery with the repulsion term
deleted and nothing else changed, reaches 0.230, below the 0.5 that sitting on
one mode of two would give. The paired differences on D6 are large against
their standard errors: mode coverage $-0.755$ (SE 0.005), kernel law error
$-0.160$ (SE 0.001), excess energy risk $-0.569$ (SE 0.002). Because the
comparator is an ablation rather than a different method, this isolates the
repulsion term and nothing else.

The full-law representation transfers to functionals declared after training.
On D7, where the location and scale surfaces agree across arms so the entire
treatment effect sits in the shape of the inner law, C-WDB-v1's error on
`TCATE-K-grid_skewness` is 0.0121 against Causal-DRF's 0.0732, a factor of six,
and PTA-S cannot produce the quantity at all. This is the clearest single
result in the tournament and it is the mechanism the contribution claims.

The finite-particle approximation is controlled and cheap. Excess energy risk
falls monotonically in $M$ and is nearly flat past $M=10$: on D1,
$0.0897 \to 0.0584 \to 0.0508 \to 0.0468$ for $M \in \{2,5,10,25\}$; on D6,
$0.1063 \to 0.0851 \to 0.0696 \to 0.0651$. Ten particles buy most of what
twenty-five buy, which is what makes the method affordable at 11.3 times
Causal-DRF's runtime rather than far more.

### Why it still fails the gate

Rule 1 fails, and it fails on the regime that matters most for credibility. D2
has an exactly null treatment effect. C-WDB-v1's `mean_quantile_rmse` there is
0.147 against a best baseline of 0.0545, a false-effect ratio of 2.69 against a
preregistered cap of 1.25. In absolute terms 0.147 is inside the 0.15 tolerance,
so the method is not wildly wrong; but it manufactures roughly two and a half
times as much apparent heterogeneity as the best comparator on data containing
none. A distributional causal method whose selling point is detecting effects
that scalar methods miss cannot also be the method most likely to report effects
that are not there. This is disqualifying on its own terms, not a threshold
technicality.

The mechanism is visible in the design. C-WDB is a boosting procedure with a
frozen budget of 100 trees and no early stopping tied to a null; `arm_shrinkage`
shrinks arm-specific leaf vectors toward the pooled vector but does not shrink
the pooled vector toward zero contrast. The BCF-based baselines carry a
half-Cauchy prior on the treatment-effect scale that pulls the contrast to zero
when the data support none, and the forest baselines average over honest splits.
C-WDB has no equivalent regulariser on the contrast itself.

### The rule 4 result must not be oversold

Rule 4 passes, but on **zero accuracy wins and three capability wins**. On every
target where both methods produce an estimate, PTA-S is at least as accurate:
it beats C-WDB on `TCATE-K-grid_mean` at D7 ($+0.0118$, SE 0.0041) and at D6 the
two are indistinguishable. C-WDB passes only because on three targets PTA-S
produces nothing at all.

That capability advantage is real and it is the point of a transfer test: the
functional is named after training, and a method holding a conditional law
answers immediately while a method holding fixed target coordinates cannot. But
a reviewer will ask the obvious question, and the honest answer weakens it:
PTA-S's remedy is to add the functional to its manifest and fit one more scalar
head, measured at 1.14 s per head. The capability gap is therefore a
convenience advantage of seconds, not a barrier. It is worth stating; it is not
worth building a paper's central claim on.

Set against that, C-WDB does hold a genuine accuracy advantage over
**Causal-DRF**, the published incumbent, on three of four eligible regimes for
the primary law metric and on five per-target functional comparisons. The
incumbent comparison is the stronger of the two.

### The one regime where the headline mechanism loses

C-WDB loses the primary law metric on D6 (0.0191 against 0.0945 for Causal-DRF,
paired difference $+0.0097$, SE 0.0005) while simultaneously dominating it on
mode coverage. The two are consistent: C-WDB places particles on both modes and
so wins any metric sensitive to missing a mode, but with only ten atoms it
represents the shape within each mode more coarsely than a forest's weighted
empirical law over a thousand training points. This is a representation-size
effect, and the $M$-sensitivity table suggests raising $M$ narrows it slowly.
It should be reported, not explained away: the regime built to showcase the
repulsion mechanism is also the regime where the aggregate law metric prefers
the incumbent.

### Why `PIVOT` rather than `INCREMENTAL-ONLY` or `KILL`

`KILL` is wrong. Two preregistered mechanisms produced large, clean, correctly
signed effects against ablations that isolate them, with zero failed cells
across 4110 and paired standard errors an order of magnitude below the effects.

`INCREMENTAL-ONLY` is wrong. The D7 skewness result is not an increment on an
existing method; no comparator in the roster can produce that estimate at all.

`PIVOT` is right, and the pivot is specific: the contribution should be
restated around the full-law representation and the repulsion mechanism, with
the null-regime behaviour repaired before any further claim. Concretely, in
order:

1. Add a contrast-level regulariser and re-test D2 and D8 as the first order of
   business. The natural candidate is shrinkage of the arm contrast toward zero,
   with strength selected by cross-fitting rather than frozen, which is what the
   BCF baselines do and what C-WDB lacks. Rule 1 is a repairable defect, not a
   property of the estimand.
2. Build the interval construction. The `Uncertainty usable` claim row was
   **not evaluated at all** in this tournament, because C-WDB has no interval
   machinery and the estimand contract forbids substituting posterior draws of a
   mean surface. A distributional causal method without uncertainty is not
   publishable, and this is the largest single gap in the work.
3. Re-run this frozen manifest afterwards. It is preregistered, it executes in
   2.5 hours on ten cores, and rules 2, 3, 5 and 6 already pass, so the
   re-run is a genuine test of the repair rather than a search for a better
   number.

Until rule 1 is repaired, no claim of superiority over the scalar baselines
should be made in writing, because the regime where they are safest is the
regime where C-WDB is worst.

## 8. Artefacts

- `results/merged/main_results.parquet` (merged rows)
- `results/merged/merge_audit.json` (reconciliation)
- `results/merged/analysis_payload.json` (every number above)
- `results/manifests/main_manifest.json` (frozen manifest)
- `results/manifests/cost_pilot.json` (measured cost basis)
- `tables/simulation/`, `figures/simulation/`

