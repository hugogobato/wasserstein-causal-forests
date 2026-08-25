# Phase G3 gate memo: preregistered simulation tournament

**Verdict:** `NOT-GO`
**Rules passed:** 4 of 6
**Merged results checksum:** `d2feee2188c882e10fa7287ac3b70403c6b84f4211cdee0510b75187fd4d47f4`
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

| Regime | Metric | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D6 | `kernel_law_error` | `causal_drf` | 0.0191 | 0.00945 | +0.009648 | 0.000452 | 0% | **comparator** |
| D5 | `reference_tcate_rmse` | `causal_drf` | 0.09606 | 0.02302 | +0.07303 | 0.00455 | 0% | **comparator** |
| D6 | `reference_tcate_rmse` | `causal_drf` | 0.07463 | 0.04116 | +0.03347 | 0.00609 | 5% | **comparator** |
| D7 | `reference_tcate_rmse` | `causal_drf` | 0.02893 | 0.02303 | +0.005893 | 0.00257 | 30% | **comparator** |
| D5 | `reference_effect_rmse` | `causal_drf` | 0.09373 | 0.01792 | +0.07581 | 0.00482 | 0% | **comparator** |
| D6 | `reference_effect_rmse` | `causal_drf` | 0.0485 | 0.02207 | +0.02644 | 0.00741 | 20% | **comparator** |

Structural limits of this tournament, restated from the preregistration:

1. PTA-F runs only at $K=5$, because its cost accelerates in the target dimension; no conclusion about it at the working resolution is available. 2. The `Uncertainty usable` claim row is **not evaluated**: C-WDB has no interval construction, and contract Section 4 forbids substituting a posterior-draw quantity. 3. No claim about Causal-DRF's band coverage relative to the published two-forest benchmark is made, per the Phase 4 limitation. 4. Grid-resolution conclusions hold for $K\in\{5,25,49\}$ only.

## 3. Gate rules

| Rule | Statement | Result |
|---|---|---|
| `rule_1_correctness` | passes D0 through D2 correctness and null checks | **FAIL** |
| `rule_2_law_advantage` | beats Causal-DRF on the primary law metric in at least two scientifically relevant mechanisms | PASS |
| `rule_3_transfer` | transfers the advantage to at least one predeclared functional or reference target | PASS |
| `rule_4_beats_direct_learner` | beats PTA-S on the transferred target of rule 3 | **FAIL** |
| `rule_5_no_collapse` | no systematic particle collapse or excessive projection | PASS |
| `rule_6_cost` | compute cost commensurate with the gain | PASS |

### Rule 1, correctness and nulls

D0 `mean_quantile_rmse` = 0.1091; D2 = 0.1466 against a best baseline of 0.0545, a false-effect ratio of 2.69.

### Rule 2, primary law metric against Causal-DRF

Wins: 3 of 2 required, in ['D1', 'D5', 'D7'].

| Regime | Metric | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D1 | `kernel_law_error` | `causal_drf` | 0.01051 | 0.01883 | -0.00832 | 0.000574 | 100% | claimant |
| D5 | `kernel_law_error` | `causal_drf` | 0.01995 | 0.05546 | -0.03551 | 0.00304 | 100% | claimant |
| D6 | `kernel_law_error` | `causal_drf` | 0.0191 | 0.00945 | +0.009648 | 0.000452 | 0% | **comparator** |
| D7 | `kernel_law_error` | `causal_drf` | 0.01007 | 0.0251 | -0.01503 | 0.00105 | 100% | claimant |

### Rules 3 and 4, transfer to a causal functional

Against Causal-DRF: 1 winning targets (['tcate_functional_rmse@D7']). Against PTA-S on those same targets: 0 (none).

| Regime | Metric | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D5 | `tcate_functional_rmse` | `causal_drf` | 0.04007 | 0.03758 | +0.002485 | 0.00252 | 30% | tie |
| D6 | `tcate_functional_rmse` | `causal_drf` | 0.0945 | 0.1054 | -0.01088 | 0.00554 | 65% | tie |
| D7 | `tcate_functional_rmse` | `causal_drf` | 0.02993 | 0.0434 | -0.01347 | 0.00303 | 90% | claimant |
| D5 | `reference_tcate_rmse` | `causal_drf` | 0.09606 | 0.02302 | +0.07303 | 0.00455 | 0% | **comparator** |
| D6 | `reference_tcate_rmse` | `causal_drf` | 0.07463 | 0.04116 | +0.03347 | 0.00609 | 5% | **comparator** |
| D7 | `reference_tcate_rmse` | `causal_drf` | 0.02893 | 0.02303 | +0.005893 | 0.00257 | 30% | **comparator** |
| D5 | `reference_effect_rmse` | `causal_drf` | 0.09373 | 0.01792 | +0.07581 | 0.00482 | 0% | **comparator** |
| D6 | `reference_effect_rmse` | `causal_drf` | 0.0485 | 0.02207 | +0.02644 | 0.00741 | 20% | **comparator** |
| D7 | `reference_effect_rmse` | `causal_drf` | 0.02016 | 0.01652 | +0.00364 | 0.00269 | 35% | tie |

| Regime | Metric | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D7 | `tcate_functional_rmse` | `pta_s` | 0.02993 | 0.01999 | +0.009941 | 0.00202 | 10% | **comparator** |

### Rule 5, particle collapse

D6 mode coverage 0.99; effective particle support 10 of $M$, a fraction of 1. The squared-$W_2$ comparator, which removes the repulsion term, reaches mode coverage 0.2303125.

### Rule 6, cost

Median runtime 13.2 s against Causal-DRF's 1.235 s, a ratio of 10.7 against a ceiling of 60.

| Method | Median runtime (s) | Max runtime (s) | Median peak RSS (MB) |
|---|---|---|---|
| `causal_drf` | 1.23 | 35.6 | 75.2 |
| `cwdb_v0` | 9.34 | 43.3 | 0 |
| `cwdb_v1` | 13.2 | 188 | 0 |
| `cwdb_v1_noshrink` | 27.6 | 34.4 | 0 |
| `pta_f` | 108 | 141 | 0 |
| `pta_s` | 26.6 | 83.4 | 0 |
| `sqw2_booster` | 1.08 | 2.86 | 0 |
| `wdrft` | 1.43 | 35.8 | 214 |

## 4. Mechanism ablations

### Repulsion, against the squared-$W_2$ booster

| Regime | Metric | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D1 | `kernel_law_error` | `sqw2_booster` | 0.0164 | 0.01387 | +0.002523 | 0.000221 | 0% | **comparator** |
| D1 | `mode_coverage` | `sqw2_booster` | 1 | 1 | +0 | 0 | 0% | tie |
| D1 | `arm_energy_risk` | `sqw2_booster` | 0.07175 | 0.1413 | -0.06954 | 0.000636 | 100% | claimant |
| D6 | `kernel_law_error` | `sqw2_booster` | 0.03681 | 0.197 | -0.1602 | 0.00122 | 100% | claimant |
| D6 | `mode_coverage` | `sqw2_booster` | 0.9853 | 0.2303 | -0.755 | 0.00499 | 100% | claimant |
| D6 | `arm_energy_risk` | `sqw2_booster` | 0.1128 | 0.6818 | -0.569 | 0.00227 | 100% | claimant |

### Arm-shared localisation, against C-WDB-v0

| Regime | Metric | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D3 | `mean_quantile_rmse` | `cwdb_v0` | 0.259 | 0.291 | -0.03199 | 0.00356 | 100% | claimant |
| D3 | `kernel_law_error` | `cwdb_v0` | 0.01325 | 0.01333 | -7.574e-05 | 0.000319 | 60% | tie |
| D4 | `mean_quantile_rmse` | `cwdb_v0` | 0.1765 | 0.2777 | -0.1012 | 0.00322 | 100% | claimant |
| D4 | `kernel_law_error` | `cwdb_v0` | 0.008894 | 0.01146 | -0.002571 | 0.000158 | 100% | claimant |

### Causal regularisation, against `arm_shrinkage = 0`

| Regime | Metric | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
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

| Regime | Metric | Comparator | Claimant | Comparator value | Paired difference | Paired SE | Seeds won | Verdict |
|---|---|---|---|---|---|---|---|---|
| D1 | `kernel_law_error` | `causal_drf` | 0.01051 | 0.01883 | -0.00832 | 0.000574 | 100% | claimant |
| D5 | `kernel_law_error` | `causal_drf` | 0.01995 | 0.05546 | -0.03551 | 0.00304 | 100% | claimant |
| D7 | `kernel_law_error` | `causal_drf` | 0.01007 | 0.0251 | -0.01503 | 0.00105 | 100% | claimant |
| D7 | `tcate_functional_rmse` | `causal_drf` | 0.02993 | 0.0434 | -0.01347 | 0.00303 | 90% | claimant |

## 6. Verdict

**`NOT-GO`**

Rules not met:

- `rule_1_correctness`: passes D0 through D2 correctness and null checks
- `rule_4_beats_direct_learner`: beats PTA-S on the transferred target of rule 3

Under the Phase G3 decision list, C-WDB does not return `GO`. The choice among `PIVOT`, `INCREMENTAL-ONLY`, and `KILL` is argued in Section 7.

The independent recomputation in `research/checks/g3_gate_flags.py` returns `NOT-GO` with 0 disagreements against the analysis code.

## 7. Interpretation and next step

_This section is written by hand against the tables above and is the only part of this memo not generated from the payload._

## 8. Artefacts

- `results/merged/main_results.parquet` (merged rows)
- `results/merged/merge_audit.json` (reconciliation)
- `results/merged/analysis_payload.json` (every number above)
- `results/manifests/main_manifest.json` (frozen manifest)
- `results/manifests/cost_pilot.json` (measured cost basis)
- `tables/simulation/`, `figures/simulation/`

