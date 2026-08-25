# Phase G3 repair memo: the rule-1 contrast regulariser

**Verdict:** `GO` for `cwdb_r3_cvridge`; `GO` for `cwdb_r2_threshold3` with one mechanism ablation lost
**Rules passed:** 6 of 6, for both
**Frozen results checksum:** `a871fd7b6ab72544e1f0f317c16cb18551759b575899188c0bfb1d048f9a4a69` (unchanged)
**Repair results checksum:** `238a64aff5171c999a80431459dc72eeee08e5b0cb050455ca89e5afe68a23e2`
**Repair manifest checksum:** `dc754113a4d086ceff74fabc0ad0b49803e250b601f8f1642a1f796ec369a96c`
**Preregistration:** `research/simulation_preregistration_repair.md`
**Parent memo:** `research/gates/G3_simulation_memo.md`

Every threshold below was frozen before the first decisive repair seed, and
every one of them is the same threshold the parent tournament used. No rule was
restated, relaxed, or re-chosen.

## 1. Execution

2600 repair cells, 68240 rows, 0 failures, merge audit `PASS`. Failure rate is
zero for all five variants. The 97650 frozen rows were not recomputed: every
baseline, ablation, and C-WDB-v1 number below is the number the parent memo
reports, read from `results/merged/main_results.parquet` unchanged.

Repair cells reuse the frozen coordinates and seeds, so every comparison is
paired seed by seed against the existing rows.

## 2. The defect, located

Two mechanisms produced C-WDB-v1's false effect on D2, and separating them
changed which repair was worth building.

**The arm-specific initialisation is confounded.** `compute_init_base` was
applied per arm, so each arm started from the empirical law of its own treated
or control sample. That is a marginal quantity, and under a covariate-dependent
propensity the two marginals differ even when the two conditional laws are
identical. Measured over 20 samples at $n = 1000$, the initial arm gap has root
mean square 0.215 on D2, where the truth is exactly zero, and 0.824 on D8.

**The contrast accumulates along the boosting path.** Replaying the fitted path
on a held-out design, D2's contrast error reaches a minimum near iteration 20
and then rises monotonically to the frozen budget of 100: 0.077, 0.093, 0.104,
0.114, 0.122 at iterations 20, 40, 60, 80, 100. `arm_shrinkage` pulls each arm's
leaf vector toward the pooled vector, but the contrast it leaves at a balanced
leaf is the raw gap times $n_a/(n_a+\lambda)$, which is 0.86 at a typical
main-grid leaf. The contrast was barely regularised at all, and a hundred
boosting steps compounded what survived.

The ablation settles the attribution. `cwdb_v1_pooledinit` is frozen v1 with the
initialisation shared and nothing else changed. It improves D2 from 0.1466 to
0.1419, a false-effect ratio of 2.60 against 2.69, and **still fails rule 1**.
The initialisation was a genuine defect and fixing it was free, but it is not
the repair. The second mechanism is.

## 3. Rule 1, the regime the gate failed on

Cap on the false-effect ratio: 1.25. Reference is the best of the frozen roster,
0.0545, which a repair variant cannot move by entering the pool.

| Method | D2 `mean_quantile_rmse` | Ratio | D0 `mean_quantile_rmse` | Rule 1 |
|---|---|---|---|---|
| `cwdb_v1` (frozen claimant) | 0.1466 | 2.69 | 0.1091 | **FAIL** |
| `cwdb_v1_pooledinit` (ablation) | 0.1419 | 2.60 | 0.0939 | **FAIL** |
| `cwdb_r1_ridge` | 0.0642 | 1.18 | 0.1574 | **FAIL** |
| `cwdb_r2_threshold` ($c=1$) | 0.0790 | 1.45 | not run | **FAIL** |
| `cwdb_r2_threshold3` ($c=3$) | 0.0194 | 0.36 | 0.1322 | PASS |
| `cwdb_r3_cvridge` | 0.0506 | 0.93 | 0.0952 | PASS |

Two results here are worth stating plainly rather than leaving in the table.

`cwdb_r1_ridge` clears the D2 ratio at 1.18 and then **fails rule 1 anyway, on
D0**, at 0.1574 against a cap of 0.15. D0 has deterministic conditional
distributions, so its conditional mean is exactly recoverable and a fixed
contrast shrinkage that cannot switch itself off biases the recoverable answer.
The hand-tuned fixed ridge is the one variant that was tuned, it was tuned on
D2, and it broke the regime it was not tuned on. That is the argument against a
frozen strength, made by the frozen strength itself.

`cwdb_r2_threshold` at the null-calibrated $c = 1$ missed the cap at 1.45 and
stopped after D2 under the preregistered screen. Its rows stay in the merged
table and in this memo. The calibration that is principled is not the one that
passed; the one that passed, $c = 3$, is a declared conservative sensitivity.

## 4. All six rules

| Method | 1 | 2 | 3 | 4 | 5 | 6 | Verdict |
|---|---|---|---|---|---|---|---|
| `cwdb_v1` | fail | PASS | PASS | PASS | PASS | PASS | `NOT-GO` |
| `cwdb_v1_pooledinit` | fail | PASS | PASS | PASS | PASS | PASS | `NOT-GO` |
| `cwdb_r1_ridge` | fail | PASS | PASS | PASS | PASS | PASS | `NOT-GO` |
| `cwdb_r2_threshold3` | PASS | PASS | PASS | PASS | PASS | PASS | **`GO`** |
| `cwdb_r3_cvridge` | PASS | PASS | PASS | PASS | PASS | PASS | **`GO`** |

The independent recomputation in `research/checks/g3_gate_flags.py` returns `GO`
for both passing variants with 0 disagreements against the analysis code.

Rule 2 holds for both: 3 wins against Causal-DRF on the primary law metric, in
D1, D5, and D7, the same three regimes the frozen claimant won, against a
requirement of 2. Rule 5 holds for both, with D6 mode coverage 0.9921 and 0.9931
against the frozen claimant's 0.9897, and the particle participation ratio at
10.00 of $M = 10$ in every case. **The repulsion
mechanism is untouched**, which is the point: the contrast rules move only the
arm gap and provably cannot move the pooled component.

Rule 6, median runtime against Causal-DRF's 1.17 s, cap 60:

| Method | Median runtime | Ratio |
|---|---|---|
| `cwdb_v1` | 13.30 s | 11.3 |
| `cwdb_r1_ridge` | 14.69 s | 12.5 |
| `cwdb_r2_threshold3` | 15.11 s | 12.9 |
| `cwdb_r3_cvridge` | 55.35 s | 47.2 |

The preregistration recorded, before this run, that R3 was projected near 64 and
might fail. It landed at 47.2. The projection multiplied every cell by the
factor measured on the expensive $K = 25$ cells; the cheap $K = 5$ cells carry a
smaller multiplier, and the median sits among them. The projection was
pessimistic, not the budget wrong.

## 5. The result that separates the two passing variants

Both pass the gate. They are not equally good, and the mechanism ablations say
why.

| Ablation | Comparator | `cwdb_v1` | `cwdb_r2_threshold3` | `cwdb_r3_cvridge` |
|---|---|---|---|---|
| Repulsion, D6 mode coverage | `sqw2_booster` | −0.7550 (0.0050) | −0.7581 (0.0064) | −0.7584 (0.0061) |
| Repulsion, D6 `kernel_law_error` | `sqw2_booster` | −0.1602 (0.0012) | −0.1646 (0.0010) | −0.1646 (0.0012) |
| Sharing, D3 `mean_quantile_rmse` | `cwdb_v0` | −0.0320 (0.0036) | **+0.0944 (0.0042)** | −0.0306 (0.0030) |
| Sharing, D3 `kernel_law_error` | `cwdb_v0` | −0.0001 (0.0003) | **+0.0055 (0.0004)** | −0.0001 (0.0003) |
| Sharing, D4 `mean_quantile_rmse` | `cwdb_v0` | −0.1012 (0.0032) | −0.0805 (0.0062) | −0.1003 (0.0039) |
| Shrinkage, D2 `mean_quantile_rmse` | `cwdb_v1_noshrink` | −0.0338 (0.0015) | −0.1431 (0.0018) | −0.1188 (0.0037) |
| Shrinkage, D8 `mean_quantile_rmse` | `cwdb_v1_noshrink` | −0.0320 (0.0027) | −0.1336 (0.0051) | −0.1420 (0.0042) |

**`cwdb_r2_threshold3` loses the sharing ablation on D3.** It reaches 0.3854
against C-WDB-v0's 0.2910, a paired difference of $+0.0944$ with standard error
0.0042, and loses the law metric there too. D3 is the regime built to favour
separate heads, and a rule that shrinks every arm gap toward the pooled leaf is
punished by exactly the structure D3 contains. The shared-partition mechanism is
one of the project's own claims, supported in the parent memo by this ablation.
A repair that passes the gate and takes that claim with it has traded one asset
for another, and the trade must be reported as one. `cwdb_r1_ridge` loses the
same ablation for the same reason.

**`cwdb_r3_cvridge` keeps it**, at $-0.0306$ (SE 0.0030) against C-WDB-v0, which
is the frozen claimant's $-0.0320$ (SE 0.0036) to within a standard error. It
loses exactly one ablation, D1 `kernel_law_error` against the squared-$W_2$
booster, which is the same one the frozen claimant already lost and is therefore
not damage the repair caused.

The reason is visible in what cross-fitting selects. Held-out energy risk picks
the contrast strength per training sample, and it picks correctly:

| Regime | Median strength selected | What the regime contains |
|---|---|---|
| D0, D3 | 0 | deterministic laws; strong separate-head effect |
| D1, D4, D5 | 0 | moderate real effects |
| D2 | 50 (mean 265) | exactly null effect |
| D6, D7, D9 | 50 | multimodal; unseen-functional effect; poor overlap |
| D8 | 500 | strong confounding, weak effect |

On D3 it selects zero on every seed, so the shared-partition mechanism runs
unregularised where the data say a real contrast exists. On D2 and D8 it selects
hard. That is the adaptivity the fixed rules cannot have, and it is what lets one
variant repair rule 1 without paying for it anywhere else.

## 6. What improved beyond the rule that failed

Paired against C-WDB-v1 on the main grid, `cwdb_r3_cvridge` on
`mean_quantile_rmse`:

| Regime | R3 | v1 | Paired difference | SE | Verdict |
|---|---|---|---|---|---|
| D0 | 0.0872 | 0.1013 | −0.0141 | 0.0025 | repair |
| D1 | 0.1809 | 0.1691 | +0.0118 | 0.0027 | v1 |
| D2 | 0.0417 | 0.1266 | −0.0850 | 0.0037 | repair |
| D3 | 0.2603 | 0.2590 | +0.0014 | 0.0022 | tie |
| D4 | 0.1774 | 0.1765 | +0.0009 | 0.0032 | tie |
| D5 | 0.0910 | 0.1559 | −0.0649 | 0.0099 | repair |
| D6 | 0.2130 | 0.2490 | −0.0360 | 0.0094 | repair |
| D7 | 0.0572 | 0.1264 | −0.0692 | 0.0020 | repair |
| D8 | 0.0641 | 0.1741 | −0.1100 | 0.0062 | repair |
| D9 | 0.2442 | 0.2211 | +0.0231 | 0.0095 | v1 |

Six regimes improve, two are ties, and two regress: D1, a smooth
location-scale law with a real effect, and D9, where overlap deteriorates to the
clipping bounds. Both regressions are small in absolute terms and both are cases
where regularising a real contrast costs something. Neither is a gate rule and
neither should be described as free.

The largest single improvement is D8, strong confounding with a weak effect,
from 0.1741 to 0.0641. That regime combines both defects the repair addresses:
the initialisation gap there was 0.824, and the effect is weak enough that
accumulated contrast noise dominates it.

## 7. What did not improve

**Rule 4 is still weak for `cwdb_r3_cvridge`, in the same way the parent memo
flagged.** It passes on 2 wins, of which **0 are accuracy wins and 2 are
capability wins**. On `TCATE-K-grid_mean` at D5 and D7, PTA-S is at least as
accurate. The repair did not touch this and was never going to: it is a
statement about what a direct target learner delivers, not about the contrast.

`cwdb_r2_threshold3` does better here, with 4 wins of which **3 are accuracy
wins** on targets PTA-S also estimates, at D6 `TCATE-K-grid_sd`, D7
`TCATE-K-grid_mean`, and D7 `TCATE-K-grid_sd`. That is a real strengthening of
the weakest part of the parent memo's case, and it belongs to the variant that
lost the sharing ablation. The two variants are strong in different places and
neither dominates.

**The uncertainty gap is untouched.** The `Uncertainty usable` claim row was not
evaluated in the parent tournament and is not evaluated here. C-WDB still has no
interval construction, and the estimand contract still forbids substituting a
posterior draw of a mean surface. This remains the largest single gap in the
work and it is item 2 of the parent memo's pivot list, not item 1.

**Three grids were not re-run**: `particles`, `resolution`, and `scaling`. They
measure finite-$M$ approximation, grid resolution, and cost scaling, none of
which a contrast rule plausibly changes. No claim about a repaired variant at
$M \ne 10$, at $K = 49$, or at $n = 2000$ is supported until they are.

## 8. Recommendation

Adopt `cwdb_r3_cvridge` as the claimant. It is the only variant that repairs
rule 1 while leaving every mechanism the project claims as its own intact, and
the mechanism that makes that possible, choosing the regularisation strength on
held-out energy risk rather than freezing it, is the one the parent memo asked
for by name.

Report `cwdb_r2_threshold3` alongside it, not as a rejected alternative but as
the variant that buys three accuracy wins over PTA-S at the price of the
shared-partition ablation. A reader who cares more about the direct-learner
comparison than about the sharing claim would prefer it, and the results say so.

The gate rule that failed is repaired. The two things the parent memo listed
after it are not: the interval construction does not exist, and rule 4 still
rests on capability rather than accuracy for the recommended variant. Neither is
a reason to withhold `GO` on this gate, and both are reasons not to describe
this as the work being finished.

## 9. Artefacts

- `results/merged_repair/main_results.parquet` (repair rows)
- `results/merged_repair/merge_audit.json` (reconciliation, `PASS`)
- `results/merged_repair/repair_payload.json` (every number above)
- `results/merged_repair/gate_flags_independent_cwdb_r3_cvridge.json`
- `results/merged_repair/gate_flags_independent_cwdb_r2_threshold3.json`
- `results/manifests/repair_manifest.json` (staged union, 2600 cells)
- `results/manifests/repair_manifest_stage1.json` (the stage-1 screen as frozen)
- `research/simulation_preregistration_repair.md`
