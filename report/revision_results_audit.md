# Results revision audit

This ledger records what can be stated from the current artifacts and what must be removed or rerun before it appears in the paper. The numerical summaries below use the final calibrated adapter `cwdb_dr`, which is labelled WCF in the proposed replacement section. They do not relabel `cwdb_r3_cvridge` as WCF.

## Source and provenance

| Evidence | Source | Scope | Use in the paper |
|---|---|---|---|
| Final WCF, abstract regimes | `results/phase6/*.parquet`, `results/phase6/trackA_shards/*.parquet`, and `results/phase6/trackB_shards/*.parquet` | `grid=main`, `method=cwdb_dr`, `D0,D2,D5,D6,D7,D8`, (n=500,1000), ten seeds | Use for the final WCF's abstract-suite results. |
| Final WCF, income regimes | Same phase-6 shards | `grid=income`, `method=cwdb_dr`, `IC0`--`IC3`, (n=500,1000), ten seeds | Use for the main comparison. |
| Original Causal-DRF | `results/merged_original_causal_drf/main_results.parquet` | `grid=main`, (n=1000), seeds 0--9 after filtering | Use as the original-code Causal-DRF comparator in the abstract table. |
| Original DRF | `results/merged_original_drf/main_results.parquet` | `grid=main`, (n=1000), seeds 0--9 after filtering | Use as the original-code DRF comparator in the abstract table. |
| Income controls, ablations, and zero inflation | `results/merged_phase65/phase65_results.parquet` | Controls, scaling, ablation, and `e_zi` tracks, ten seeds | Use for control and two-part results only where the final adapter is present. |
| Frozen configuration | `results/manifests/phase6_manifest.json` and `results/manifests/phase65_manifest.json` | Cell keys, targets, $K=25$, $M=10$, seed and test-seed rules | Cite as computational provenance, not as prose about an internal research process. |

The phase-6 manifest currently on disk records `selection_folds=3`, while the checked-in `phase6.py` constant is now 2. The historical parquet files must therefore be described by their manifest, not inferred from the current source. The replacement paper should report the final manifest actually used for any rerun and should not state one fold count globally until this discrepancy is resolved. The AIPW propensity nuisance uses five folds and clips to ([0.02,0.98]) in the checked-in implementation.

## Final estimator identity

`cwdb_dr` is implemented by `DRAdapter` in `src/wasserstein_causal_forests/g3/phase6_methods.py`. It fits the shared-partition particle booster with pooled initialization, cross-validated arm shrinkage, and (M) particles on a (K)-grid. It then computes cross-fitted AIPW scores for the declared grid functionals and the reference distance. The AIPW layer replaces functional and reference contrast outputs, including their marginal and moderator-bin versions. It does not alter the predicted particles, the law metric, or the mean-quantile contrast.

For the paper, use the name WCF for `cwdb_dr` and describe the implementation constants in the simulation-methods section. Use WCF-ZIPT only for the two-part zero-inflated extension. `cwdb_zipt` in phase 6.5 assembles a logistic degeneracy classifier and a positive-part booster; it does not apply the WCF AIPW calibration layer. It must therefore be described as an uncalibrated two-part extension unless it is separately rerun with calibration.

`cwdb_r3_cvridge`, `cwdb_v1`, `WCF-v1`, `WCF-R3`, and `PTA-S` are not final-method results for the paper. They can remain in the audit and code history, but they should not appear in the method comparison or results narrative. The old tables also report `mean_quantile_rmse` as a scalar grid-mean target. The evaluator defines it as the RMSE of the entire $K$-vector mean-quantile contrast, target `MEANQ-A-K`.

## Metric semantics that must be corrected

The evaluator emits one scalar per replication. `mean_quantile_rmse`, `tcate_functional_rmse`, `kernel_law_error`, and `reference_tcate_rmse` are computed over test rows or moderator bins within that replication. `tate_functional_rmse` and `reference_effect_rmse` are absolute errors of the estimated marginal contrast. They are not test-row RMSEs. The summaries below report the mean of these replication-level errors over ten seeds. The seed standard errors are available in the parquet data; no old table should label a mean of absolute marginal errors as a test-row RMSE.

The four TATE targets are `TATE-K-grid_mean`, `TATE-K-grid_sd`, `TATE-K-grid_skewness`, and `TATE-K-grid_upper_tail_mean`. The corresponding TCATE targets are the same four functionals conditional on the four bins of the pre-treatment moderator. The new reference-distribution targets are already implemented as `REF-ATE-K` and `REF-TCATE-K`: they estimate the error in the treatment contrast of the expected grid Wasserstein distance to (q_\star), namely (E[r_1^K(X)-r_0^K(X)]) and its moderator-binned analogue. These are the reference versions of marginal and conditional distributional treatment effects and should be presented as a contribution, with a clear definition.

## Final WCF and incumbent results at (n=1000)

The following abstract-suite values are means over seeds 0--9. WCF values come from phase 6; incumbent values come from the original-code merged files after the same seed filter. Lower is better. The table deliberately retains WCF's losses on D5 and D6 rather than describing a universal win.

| DGP | WCF meanQ | Causal-DRF meanQ | DRF meanQ | WCF law | Causal-DRF law | DRF law | WCF Ref-TCATE | Causal-DRF Ref-TCATE | DRF Ref-TCATE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0, deterministic | 0.0868 | 0.1391 | 0.1672 | 0.0234 | 0.1033 | 0.1191 | 0.0097 | 0.0886 | 0.0799 |
| D2, null effect | 0.0410 | 0.0818 | 0.0627 | 0.0098 | 0.0154 | 0.0138 | 0.0399 | 0.0227 | 0.0224 |
| D5, different outer laws | 0.0946 | 0.0765 | 0.0917 | 0.0220 | 0.0374 | 0.0431 | 0.0418 | 0.0239 | 0.0235 |
| D6, multimodal outer law | 0.2189 | 0.2310 | 0.1986 | 0.0178 | 0.0064 | 0.0059 | 0.0862 | 0.0585 | 0.0501 |
| D7, shape transfer | 0.0553 | 0.0721 | 0.0662 | 0.0094 | 0.0129 | 0.0142 | 0.0386 | 0.0235 | 0.0238 |
| D8, strong confounding | 0.0635 | 0.2100 | 0.1623 | 0.0127 | 0.0417 | 0.0416 | 0.0793 | 0.0584 | 0.0724 |

The abstract suite thus supports a narrower claim. WCF is competitive or superior on mean-quantile and reference targets in D0, D2, D7, and D8, and its law error is lower in D0, D2, D5, D7, and D8. It loses law error and reference-TCATE on D6, where the conditional law has a multimodal outer component. No sentence should attribute a universal abstract-suite reversal to endogenous assignment.

The income-track table below uses the final phase-6 adapter and the two law-producing incumbents. Each cell is the mean over ten seed-level errors at (n=1000); TATE and TCATE average the four registered grid functionals within a replication. `Ref-ATE` and `Ref-TCATE` are the reference-distribution targets defined above.

| Regime | Method | meanQ | law | TATE | TCATE | Ref-ATE | Ref-TCATE |
|---|---|---:|---:|---:|---:|---:|---:|
| IC0 | WCF | 0.0356 | 0.0153 | 0.0100 | 0.0240 | 0.0142 | 0.0298 |
| IC0 | Causal-DRF | 0.1947 | 0.0467 | 0.1134 | 0.1138 | 0.1570 | 0.1572 |
| IC0 | DRF | 0.1963 | 0.0641 | 0.1104 | 0.1105 | 0.1543 | 0.1544 |
| IC1 | WCF | 0.0904 | 0.0174 | 0.0128 | 0.0271 | 0.0143 | 0.0300 |
| IC1 | Causal-DRF | 0.1760 | 0.0455 | 0.0962 | 0.0966 | 0.1329 | 0.1333 |
| IC1 | DRF | 0.1876 | 0.0591 | 0.1012 | 0.1013 | 0.1394 | 0.1395 |
| IC2 | WCF | 0.0700 | 0.0167 | 0.0099 | 0.0255 | 0.0143 | 0.0330 |
| IC2 | Causal-DRF | 0.1975 | 0.0488 | 0.1176 | 0.1179 | 0.1565 | 0.1567 |
| IC2 | DRF | 0.2298 | 0.0724 | 0.1357 | 0.1358 | 0.1795 | 0.1796 |
| IC3 | WCF | 0.1103 | 0.0180 | 0.0188 | 0.0412 | 0.0161 | 0.0463 |
| IC3 | Causal-DRF | 0.3177 | 0.0605 | 0.1784 | 0.1786 | 0.2474 | 0.2476 |
| IC3 | DRF | 0.3481 | 0.0773 | 0.1972 | 0.1973 | 0.2701 | 0.2702 |

The placebo IC0 is useful because all causal effects are zero. WCF has the smallest mean-quantile, law, functional, and reference errors in that table. On IC1--IC3, WCF has the smallest error in every listed column among the three methods. This supports a claim about these income-like designs, not a general theorem about all forms of endogenous adoption.

## Zero-inflated results available now

The phase-6.5 results at (n=1000) include the final WCF, the uncalibrated two-part WCF-ZIPT extension, Causal-DRF, and DRF. The table reports the mass-placement RMSE, the moderator-binned contrast in degenerate mass, and law error.

| Regime | Method | zero-mass | mass contrast | law |
|---|---|---:|---:|---:|
| ZI0 | WCF | 0.4947 | 0.0094 | 0.1113 |
| ZI0 | WCF-ZIPT | 0.0537 | 0.0427 | 0.0133 |
| ZI0 | Causal-DRF | 0.0945 | 0.0595 | 0.0247 |
| ZI0 | DRF | 0.1058 | 0.0561 | 0.0296 |
| ZI1 | WCF | 0.4173 | 0.1532 | 0.1264 |
| ZI1 | WCF-ZIPT | 0.0526 | 0.0365 | 0.0154 |
| ZI1 | Causal-DRF | 0.0901 | 0.0426 | 0.0249 |
| ZI1 | DRF | 0.1022 | 0.0479 | 0.0291 |
| ZI2 | WCF | 0.4954 | 0.0080 | 0.1112 |
| ZI2 | WCF-ZIPT | 0.0537 | 0.0427 | 0.0136 |
| ZI2 | Causal-DRF | 0.0946 | 0.0597 | 0.0258 |
| ZI2 | DRF | 0.1060 | 0.0564 | 0.0304 |
| ZI3 | WCF | 0.4951 | 0.0746 | 0.1741 |
| ZI3 | WCF-ZIPT | 0.0598 | 0.0445 | 0.0176 |
| ZI3 | Causal-DRF | 0.1731 | 0.2051 | 0.0768 |
| ZI3 | DRF | 0.3475 | 0.3823 | 0.2472 |

These values support the structural motivation for an optional two-part assembly. The plain particle cloud cannot place the point mass, while WCF-ZIPT substantially reduces mass and law error. Its functional and reference columns should not be interpreted as AIPW-calibrated WCF results unless a calibrated ZIPT variant is run.

## DGP and attribution decisions

The existing `DAskew`, `DArand`, `DAunit`, `DAref`, and `DAdim` cells are descriptive stress tests, not a clean proof that one feature causes a ranking reversal. In particular, `DArand` changes assignment, while the final WCF's AIPW layer and the comparator's nuisance fits can react differently to the changed training composition. The final paper should state the observed performance under each regime and reserve causal mechanism language for the prespecified new DGP block in `report/sensitivity_and_dgp_experiments.md`.

The new stress probe confirms that a symmetric outcome with nonlinear prognostic and propensity surfaces is easy to construct and has meaningful imbalance. Estimator scores for those regimes are not available yet. Until they are run, the paper should say that the current suite contains nonlinear outcome surfaces and overlap stress, while nonlinear propensity sensitivity remains future simulation work.

## Required paper edits

The old abstract and results prose contains internal-development language and unsupported scope claims. Remove references to preregistration as a selling point, frozen manifests as a narrative device, intermediary WCF versions, PTA-S, and statements that an ablation proves the gain is not due to skewness or scale. Replace them with the final WCF definition, the income-track simulation study, the explicit TATE, TCATE, Ref-ATE, and Ref-TCATE estimands, the zero-inflation extension, and the limitations documented above.
