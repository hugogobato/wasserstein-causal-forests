# WCF sensitivity analysis

Input rows come from `results/wcf_sensitivity/merged/wcf_sensitivity_combined.parquet` (SHA-256 `cc95154d4ced3e31f3131cfc39010ce9c004a8859cbf5af6af50e94f26419551`) and the frozen manifest `WCF-SENSITIVITY-v1` with checksum `1a779680cca9e037ec3f70be86a98943a59ff37489763d24841c0a9c59d48805`, 880 declared cells. The merged frame holds 71680 rows covering 1520 cells; the failure table shows that no failed cells are recorded. Metric tables exclude failed and inapplicable rows.

Aggregation first averages arm-specific law metrics inside each cell and keeps every functional TATE and TCATE target separate. Replication tables report the mean with its Monte Carlo standard error `sd / sqrt(n_seeds)`, and every sensitivity comparison is a seed-paired difference with a Monte Carlo SE computed from the differences themselves.

## Decision rule

The rule was fixed before outcomes were inspected. It retains the primary (K, M) = (25, 10) when condition A holds and condition B does not. Condition A requires every one-factor alternative (5, 10), (49, 10), (25, 5), (25, 25) to stay within 10 percent of the primary on REF-TCATE-K and LAW-A-K in IC1 and IC3. Condition B holds when some larger-resolution pair (49, 10), (25, 25), or (49, 25) improves both targets in both regimes by more than two paired Monte Carlo standard errors.

**Verdict:** `retain_primary = False`, `rule_failed = True`, `evaluable = True`. Condition A passed: False (12 of 16 records within tolerance). Condition B any improvement: False. Recommended pair: None. Reason: at least one one-factor alternative leaves the relative-tolerance band and no larger pair proves an improvement.

### Condition A details

| dgp | n_grid | n_particles | metric | target_id | primary_mean | alternative_mean | absolute_delta | relative_delta | within_tolerance | evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IC1 | 5 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.03085 | 0.02608 | 0.004774 | 0.1547 | False | True |
| IC1 | 49 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.03085 | 0.02651 | 0.004344 | 0.1408 | False | True |
| IC1 | 25 | 5 | reference_tcate_rmse | REF-TCATE-K | 0.03085 | 0.03003 | 0.0008279 | 0.02683 | True | True |
| IC1 | 25 | 25 | reference_tcate_rmse | REF-TCATE-K | 0.03085 | 0.02805 | 0.002803 | 0.09083 | True | True |
| IC1 | 5 | 10 | kernel_law_error | LAW-A-K | 0.01694 | 0.02018 | 0.003234 | 0.1909 | False | True |
| IC1 | 49 | 10 | kernel_law_error | LAW-A-K | 0.01694 | 0.01679 | 0.0001484 | 0.008759 | True | True |
| IC1 | 25 | 5 | kernel_law_error | LAW-A-K | 0.01694 | 0.01742 | 0.000479 | 0.02827 | True | True |
| IC1 | 25 | 25 | kernel_law_error | LAW-A-K | 0.01694 | 0.01717 | 0.0002244 | 0.01324 | True | True |
| IC3 | 5 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.04736 | 0.05312 | 0.00576 | 0.1216 | False | True |
| IC3 | 49 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.04736 | 0.0512 | 0.003838 | 0.08103 | True | True |
| IC3 | 25 | 5 | reference_tcate_rmse | REF-TCATE-K | 0.04736 | 0.04626 | 0.001107 | 0.02336 | True | True |
| IC3 | 25 | 25 | reference_tcate_rmse | REF-TCATE-K | 0.04736 | 0.04873 | 0.001368 | 0.02888 | True | True |
| IC3 | 5 | 10 | kernel_law_error | LAW-A-K | 0.01879 | 0.02036 | 0.001575 | 0.08384 | True | True |
| IC3 | 49 | 10 | kernel_law_error | LAW-A-K | 0.01879 | 0.01878 | 5.072e-06 | 0.00027 | True | True |
| IC3 | 25 | 5 | kernel_law_error | LAW-A-K | 0.01879 | 0.01802 | 0.0007669 | 0.04082 | True | True |
| IC3 | 25 | 25 | kernel_law_error | LAW-A-K | 0.01879 | 0.01926 | 0.0004707 | 0.02505 | True | True |

### Condition B details

| dgp | n_grid | n_particles | metric | target_id | n_pairs | paired_delta | paired_mc_se | improves | evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IC1 | 49 | 10 | reference_tcate_rmse | REF-TCATE-K | 10 | 0.004344 | 0.001527 | True | True |
| IC1 | 49 | 10 | kernel_law_error | LAW-A-K | 10 | 0.0001484 | 0.0005737 | False | True |
| IC3 | 49 | 10 | reference_tcate_rmse | REF-TCATE-K | 10 | -0.003838 | 0.004618 | False | True |
| IC3 | 49 | 10 | kernel_law_error | LAW-A-K | 10 | 5.072e-06 | 0.0004892 | False | True |
| IC1 | 25 | 25 | reference_tcate_rmse | REF-TCATE-K | 10 | 0.002803 | 0.001364 | True | True |
| IC1 | 25 | 25 | kernel_law_error | LAW-A-K | 10 | -0.0002244 | 0.0004161 | False | True |
| IC3 | 25 | 25 | reference_tcate_rmse | REF-TCATE-K | 10 | -0.001368 | 0.003847 | False | True |
| IC3 | 25 | 25 | kernel_law_error | LAW-A-K | 10 | -0.0004707 | 0.0006898 | False | True |
| IC1 | 49 | 25 | reference_tcate_rmse | REF-TCATE-K | 10 | 0.001088 | 0.001162 | False | True |
| IC1 | 49 | 25 | kernel_law_error | LAW-A-K | 10 | -0.0001255 | 0.0004294 | False | True |
| IC3 | 49 | 25 | reference_tcate_rmse | REF-TCATE-K | 10 | -0.001599 | 0.004048 | False | True |
| IC3 | 49 | 25 | kernel_law_error | LAW-A-K | 10 | 0.0001143 | 0.0007499 | False | True |

## One-factor sensitivity, grid resolution (M = 10)

| dgp | n_grid | n_particles | metric | target_id | mean | mc_se | primary_mean | absolute_delta | relative_delta | paired_delta | paired_mc_se | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IC0 | 5 | 10 | kernel_law_error | LAW-A-K | 0.01774 | 0.0006643 | 0.01529 | 0.002454 | 0.1605 | -0.002454 | 0.0004062 | 10 |
| IC0 | 25 | 10 | kernel_law_error | LAW-A-K | 0.01529 | 0.0004918 | 0.01529 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | kernel_law_error | LAW-A-K | 0.015 | 0.0004759 | 0.01529 | 0.0002896 | 0.01895 | 0.0002896 | 0.0004981 | 10 |
| IC0 | 5 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.03842 | 0.01014 | 0.04272 | 0.0043 | 0.1007 | 0.0043 | 0.01221 | 10 |
| IC0 | 25 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.04272 | 0.004141 | 0.04272 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.02495 | 0.00415 | 0.04272 | 0.01777 | 0.416 | 0.01777 | 0.005328 | 10 |
| IC0 | 5 | 10 | reference_effect_rmse | REF-ATE-K | 0.01307 | 0.002565 | 0.01512 | 0.002057 | 0.136 | 0.002057 | 0.002095 | 10 |
| IC0 | 25 | 10 | reference_effect_rmse | REF-ATE-K | 0.01512 | 0.002 | 0.01512 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | reference_effect_rmse | REF-ATE-K | 0.01121 | 0.001932 | 0.01512 | 0.00391 | 0.2585 | 0.00391 | 0.002226 | 10 |
| IC0 | 5 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.03185 | 0.002749 | 0.02857 | 0.003273 | 0.1146 | -0.003273 | 0.002027 | 10 |
| IC0 | 25 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.02857 | 0.002556 | 0.02857 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.02973 | 0.003372 | 0.02857 | 0.001154 | 0.04039 | -0.001154 | 0.002595 | 10 |
| IC0 | 5 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01403 | 0.00298 | 0.01558 | 0.001549 | 0.09939 | 0.001549 | 0.001221 | 10 |
| IC0 | 25 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01558 | 0.003012 | 0.01558 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01353 | 0.002793 | 0.01558 | 0.00205 | 0.1316 | 0.00205 | 0.00188 | 10 |
| IC0 | 5 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.009219 | 0.001731 | 0.01024 | 0.001022 | 0.0998 | 0.001022 | 0.001861 | 10 |
| IC0 | 25 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.01024 | 0.002345 | 0.01024 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.01065 | 0.001275 | 0.01024 | 0.0004069 | 0.03973 | -0.0004069 | 0.002206 | 10 |
| IC0 | 5 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.002194 | 0.0004848 | 0.002547 | 0.0003529 | 0.1385 | 0.0003529 | 0.0008231 | 10 |
| IC0 | 25 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.002547 | 0.0004915 | 0.002547 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.003816 | 0.001003 | 0.002547 | 0.001269 | 0.4981 | -0.001269 | 0.001147 | 10 |
| IC0 | 5 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01534 | 0.002375 | 0.01702 | 0.001681 | 0.09876 | 0.001681 | 0.001527 | 10 |
| IC0 | 25 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01702 | 0.002371 | 0.01702 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01603 | 0.002648 | 0.01702 | 0.0009855 | 0.05791 | 0.0009855 | 0.002214 | 10 |
| IC0 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03535 | 0.004411 | 0.03128 | 0.004071 | 0.1302 | -0.004071 | 0.001994 | 10 |
| IC0 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03128 | 0.004229 | 0.03128 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03414 | 0.004147 | 0.03128 | 0.002856 | 0.09129 | -0.002856 | 0.002328 | 10 |
| IC0 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.01644 | 0.001668 | 0.02123 | 0.004791 | 0.2257 | 0.004791 | 0.001068 | 10 |
| IC0 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02123 | 0.001768 | 0.02123 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02276 | 0.002257 | 0.02123 | 0.001527 | 0.0719 | -0.001527 | 0.001567 | 10 |
| IC0 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.005707 | 0.0006805 | 0.005915 | 0.0002075 | 0.03508 | 0.0002075 | 0.001142 | 10 |
| IC0 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.005915 | 0.0005785 | 0.005915 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.007448 | 0.000894 | 0.005915 | 0.001533 | 0.2591 | -0.001533 | 0.00129 | 10 |
| IC0 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03729 | 0.003784 | 0.03418 | 0.00311 | 0.091 | -0.00311 | 0.001729 | 10 |
| IC0 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03418 | 0.004081 | 0.03418 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03753 | 0.004511 | 0.03418 | 0.00335 | 0.09801 | -0.00335 | 0.002184 | 10 |
| IC1 | 5 | 10 | kernel_law_error | LAW-A-K | 0.02018 | 0.0007965 | 0.01694 | 0.003234 | 0.1909 | -0.003234 | 0.0004496 | 10 |
| IC1 | 25 | 10 | kernel_law_error | LAW-A-K | 0.01694 | 0.0004957 | 0.01694 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | kernel_law_error | LAW-A-K | 0.01679 | 0.0006519 | 0.01694 | 0.0001484 | 0.008759 | 0.0001484 | 0.0005737 | 10 |
| IC1 | 5 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.1092 | 0.008469 | 0.108 | 0.001191 | 0.01103 | -0.001191 | 0.009748 | 10 |
| IC1 | 25 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.108 | 0.00819 | 0.108 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.0952 | 0.005959 | 0.108 | 0.01281 | 0.1186 | 0.01281 | 0.01078 | 10 |
| IC1 | 5 | 10 | reference_effect_rmse | REF-ATE-K | 0.01175 | 0.001982 | 0.01227 | 0.0005178 | 0.04221 | 0.0005178 | 0.001775 | 10 |
| IC1 | 25 | 10 | reference_effect_rmse | REF-ATE-K | 0.01227 | 0.00219 | 0.01227 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | reference_effect_rmse | REF-ATE-K | 0.01408 | 0.002132 | 0.01227 | 0.001813 | 0.1478 | -0.001813 | 0.0009576 | 10 |
| IC1 | 5 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.02608 | 0.002635 | 0.03085 | 0.004774 | 0.1547 | 0.004774 | 0.001571 | 10 |
| IC1 | 25 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.03085 | 0.002588 | 0.03085 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.02651 | 0.002426 | 0.03085 | 0.004344 | 0.1408 | 0.004344 | 0.001527 | 10 |
| IC1 | 5 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01599 | 0.002048 | 0.01481 | 0.001184 | 0.07996 | -0.001184 | 0.002253 | 10 |
| IC1 | 25 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01481 | 0.003016 | 0.01481 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01562 | 0.00297 | 0.01481 | 0.0008129 | 0.0549 | -0.0008129 | 0.001224 | 10 |
| IC1 | 5 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.008847 | 0.001904 | 0.01178 | 0.002934 | 0.2491 | 0.002934 | 0.00216 | 10 |
| IC1 | 25 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.01178 | 0.002384 | 0.01178 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.008389 | 0.001914 | 0.01178 | 0.003392 | 0.2879 | 0.003392 | 0.002108 | 10 |
| IC1 | 5 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.003596 | 0.0006699 | 0.004293 | 0.0006966 | 0.1623 | 0.0006966 | 0.000889 | 10 |
| IC1 | 25 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.004293 | 0.000926 | 0.004293 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.005613 | 0.001414 | 0.004293 | 0.00132 | 0.3074 | -0.00132 | 0.001779 | 10 |
| IC1 | 5 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.0171 | 0.002645 | 0.01917 | 0.002065 | 0.1078 | 0.002065 | 0.003187 | 10 |
| IC1 | 25 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01917 | 0.002893 | 0.01917 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01486 | 0.003444 | 0.01917 | 0.004303 | 0.2245 | 0.004303 | 0.002077 | 10 |
| IC1 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03325 | 0.003658 | 0.03484 | 0.001593 | 0.04571 | 0.001593 | 0.002308 | 10 |
| IC1 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03484 | 0.003914 | 0.03484 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.02951 | 0.003994 | 0.03484 | 0.005328 | 0.1529 | 0.005328 | 0.002403 | 10 |
| IC1 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.01951 | 0.002153 | 0.02118 | 0.001667 | 0.07872 | 0.001667 | 0.002252 | 10 |
| IC1 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02118 | 0.002518 | 0.02118 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02295 | 0.002529 | 0.02118 | 0.001768 | 0.08347 | -0.001768 | 0.001633 | 10 |
| IC1 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.008569 | 0.000965 | 0.009865 | 0.001296 | 0.1314 | 0.001296 | 0.0009576 | 10 |
| IC1 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.009865 | 0.001028 | 0.009865 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.01231 | 0.001562 | 0.009865 | 0.002445 | 0.2479 | -0.002445 | 0.001368 | 10 |
| IC1 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03695 | 0.003036 | 0.03816 | 0.001211 | 0.03173 | 0.001211 | 0.002953 | 10 |
| IC1 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03816 | 0.00398 | 0.03816 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03556 | 0.004226 | 0.03816 | 0.002598 | 0.06808 | 0.002598 | 0.003037 | 10 |
| IC2 | 5 | 10 | kernel_law_error | LAW-A-K | 0.01908 | 0.0005479 | 0.01721 | 0.001872 | 0.1088 | -0.001872 | 0.0005171 | 10 |
| IC2 | 25 | 10 | kernel_law_error | LAW-A-K | 0.01721 | 0.0006393 | 0.01721 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | kernel_law_error | LAW-A-K | 0.01656 | 0.0004447 | 0.01721 | 0.0006509 | 0.03782 | 0.0006509 | 0.0006433 | 10 |
| IC2 | 5 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.08065 | 0.007885 | 0.08228 | 0.001627 | 0.01977 | 0.001627 | 0.01271 | 10 |
| IC2 | 25 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.08228 | 0.008736 | 0.08228 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.0788 | 0.005957 | 0.08228 | 0.003479 | 0.04228 | 0.003479 | 0.01138 | 10 |
| IC2 | 5 | 10 | reference_effect_rmse | REF-ATE-K | 0.01084 | 0.003061 | 0.01586 | 0.005022 | 0.3167 | 0.005022 | 0.002605 | 10 |
| IC2 | 25 | 10 | reference_effect_rmse | REF-ATE-K | 0.01586 | 0.003535 | 0.01586 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | reference_effect_rmse | REF-ATE-K | 0.01182 | 0.003552 | 0.01586 | 0.004043 | 0.2549 | 0.004043 | 0.002745 | 10 |
| IC2 | 5 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.03016 | 0.003585 | 0.031 | 0.0008379 | 0.02703 | 0.0008379 | 0.002635 | 10 |
| IC2 | 25 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.031 | 0.003475 | 0.031 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.03333 | 0.003289 | 0.031 | 0.002331 | 0.0752 | -0.002331 | 0.002812 | 10 |
| IC2 | 5 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01219 | 0.003162 | 0.01698 | 0.004786 | 0.2819 | 0.004786 | 0.002367 | 10 |
| IC2 | 25 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01698 | 0.004234 | 0.01698 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01309 | 0.004289 | 0.01698 | 0.003887 | 0.229 | 0.003887 | 0.002474 | 10 |
| IC2 | 5 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.01125 | 0.002286 | 0.01424 | 0.002992 | 0.21 | 0.002992 | 0.001731 | 10 |
| IC2 | 25 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.01424 | 0.002249 | 0.01424 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.01206 | 0.002236 | 0.01424 | 0.002179 | 0.153 | 0.002179 | 0.001717 | 10 |
| IC2 | 5 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.003266 | 0.0006511 | 0.00288 | 0.000386 | 0.134 | -0.000386 | 0.00067 | 10 |
| IC2 | 25 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.00288 | 0.0006226 | 0.00288 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.003679 | 0.001057 | 0.00288 | 0.0007985 | 0.2772 | -0.0007985 | 0.001267 | 10 |
| IC2 | 5 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01188 | 0.003032 | 0.0177 | 0.005813 | 0.3285 | 0.005813 | 0.002609 | 10 |
| IC2 | 25 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.0177 | 0.00404 | 0.0177 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.0154 | 0.004198 | 0.0177 | 0.002293 | 0.1296 | 0.002293 | 0.002333 | 10 |
| IC2 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03484 | 0.003469 | 0.03344 | 0.001402 | 0.04193 | -0.001402 | 0.003277 | 10 |
| IC2 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03344 | 0.003619 | 0.03344 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03599 | 0.003526 | 0.03344 | 0.002552 | 0.0763 | -0.002552 | 0.002908 | 10 |
| IC2 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.01942 | 0.002438 | 0.02454 | 0.00512 | 0.2087 | 0.00512 | 0.001233 | 10 |
| IC2 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02454 | 0.002509 | 0.02454 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02333 | 0.003295 | 0.02454 | 0.001206 | 0.04914 | 0.001206 | 0.001505 | 10 |
| IC2 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.007768 | 0.0007424 | 0.009596 | 0.001827 | 0.1904 | 0.001827 | 0.001694 | 10 |
| IC2 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.009596 | 0.001555 | 0.009596 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.008381 | 0.0006287 | 0.009596 | 0.001214 | 0.1266 | 0.001214 | 0.001694 | 10 |
| IC2 | 5 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03553 | 0.003403 | 0.03415 | 0.001373 | 0.0402 | -0.001373 | 0.003346 | 10 |
| IC2 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03415 | 0.003008 | 0.03415 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 49 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03528 | 0.00354 | 0.03415 | 0.001125 | 0.03293 | -0.001125 | 0.002922 | 10 |
| IC3 | 5 | 10 | kernel_law_error | LAW-A-K | 0.02036 | 0.0007213 | 0.01879 | 0.001575 | 0.08384 | -0.001575 | 0.0006208 | 10 |
| IC3 | 25 | 10 | kernel_law_error | LAW-A-K | 0.01879 | 0.000681 | 0.01879 | 0 | 0 | 0 | 0 | 10 |
| IC3 | 49 | 10 | kernel_law_error | LAW-A-K | 0.01878 | 0.0007327 | 0.01879 | 5.072e-06 | 0.00027 | 5.072e-06 | 0.0004892 | 10 |
| IC3 | 5 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.1187 | 0.003793 | 0.1183 | 0.0004292 | 0.003627 | -0.0004292 | 0.007127 | 10 |
| IC3 | 25 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.1183 | 0.005502 | 0.1183 | 0 | 0 | 0 | 0 | 10 |
| IC3 | 49 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.121 | 0.007409 | 0.1183 | 0.002645 | 0.02236 | -0.002645 | 0.005984 | 10 |
| IC3 | 5 | 10 | reference_effect_rmse | REF-ATE-K | 0.01608 | 0.004332 | 0.01816 | 0.00208 | 0.1145 | 0.00208 | 0.00194 | 10 |
| IC3 | 25 | 10 | reference_effect_rmse | REF-ATE-K | 0.01816 | 0.003945 | 0.01816 | 0 | 0 | 0 | 0 | 10 |
| IC3 | 49 | 10 | reference_effect_rmse | REF-ATE-K | 0.01874 | 0.00438 | 0.01816 | 0.000574 | 0.03161 | -0.000574 | 0.003823 | 10 |
| IC3 | 5 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.05312 | 0.005873 | 0.04736 | 0.00576 | 0.1216 | -0.00576 | 0.003862 | 10 |
| IC3 | 25 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.04736 | 0.006127 | 0.04736 | 0 | 0 | 0 | 0 | 10 |
| IC3 | 49 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.0512 | 0.005514 | 0.04736 | 0.003838 | 0.08103 | -0.003838 | 0.004618 | 10 |

_Showing 120 of 144 rows._

## One-factor sensitivity, particle resolution (K = 25)

| dgp | n_grid | n_particles | metric | target_id | mean | mc_se | primary_mean | absolute_delta | relative_delta | paired_delta | paired_mc_se | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IC0 | 25 | 5 | kernel_law_error | LAW-A-K | 0.01529 | 0.0004695 | 0.01529 | 3.024e-06 | 0.0001978 | -3.024e-06 | 0.0004077 | 10 |
| IC0 | 25 | 10 | kernel_law_error | LAW-A-K | 0.01529 | 0.0004918 | 0.01529 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | kernel_law_error | LAW-A-K | 0.01526 | 0.0004728 | 0.01529 | 2.335e-05 | 0.001527 | 2.335e-05 | 0.0005857 | 10 |
| IC0 | 25 | 5 | mean_quantile_rmse | MEANQ-A-K | 0.03561 | 0.00538 | 0.04272 | 0.007113 | 0.1665 | 0.007113 | 0.007937 | 10 |
| IC0 | 25 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.04272 | 0.004141 | 0.04272 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | mean_quantile_rmse | MEANQ-A-K | 0.04802 | 0.01013 | 0.04272 | 0.005302 | 0.1241 | -0.005302 | 0.0107 | 10 |
| IC0 | 25 | 5 | reference_effect_rmse | REF-ATE-K | 0.01423 | 0.002467 | 0.01512 | 0.000896 | 0.05925 | 0.000896 | 0.00208 | 10 |
| IC0 | 25 | 10 | reference_effect_rmse | REF-ATE-K | 0.01512 | 0.002 | 0.01512 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | reference_effect_rmse | REF-ATE-K | 0.01263 | 0.001821 | 0.01512 | 0.002495 | 0.165 | 0.002495 | 0.001841 | 10 |
| IC0 | 25 | 5 | reference_tcate_rmse | REF-TCATE-K | 0.0298 | 0.002771 | 0.02857 | 0.001228 | 0.04296 | -0.001228 | 0.002461 | 10 |
| IC0 | 25 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.02857 | 0.002556 | 0.02857 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | reference_tcate_rmse | REF-TCATE-K | 0.02804 | 0.002573 | 0.02857 | 0.000532 | 0.01862 | 0.000532 | 0.002583 | 10 |
| IC0 | 25 | 5 | tate_functional_rmse | TATE-K-grid_mean | 0.01282 | 0.002444 | 0.01558 | 0.002759 | 0.1771 | 0.002759 | 0.001608 | 10 |
| IC0 | 25 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01558 | 0.003012 | 0.01558 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | tate_functional_rmse | TATE-K-grid_mean | 0.01428 | 0.003257 | 0.01558 | 0.001303 | 0.0836 | 0.001303 | 0.001958 | 10 |
| IC0 | 25 | 5 | tate_functional_rmse | TATE-K-grid_sd | 0.01156 | 0.00186 | 0.01024 | 0.001323 | 0.1292 | -0.001323 | 0.002309 | 10 |
| IC0 | 25 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.01024 | 0.002345 | 0.01024 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | tate_functional_rmse | TATE-K-grid_sd | 0.007642 | 0.001868 | 0.01024 | 0.002598 | 0.2537 | 0.002598 | 0.001791 | 10 |
| IC0 | 25 | 5 | tate_functional_rmse | TATE-K-grid_skewness | 0.002825 | 0.0007187 | 0.002547 | 0.0002778 | 0.1091 | -0.0002778 | 0.0008033 | 10 |
| IC0 | 25 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.002547 | 0.0004915 | 0.002547 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | tate_functional_rmse | TATE-K-grid_skewness | 0.00202 | 0.0006827 | 0.002547 | 0.0005269 | 0.2068 | 0.0005269 | 0.0007957 | 10 |
| IC0 | 25 | 5 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.0129 | 0.002266 | 0.01702 | 0.004122 | 0.2422 | 0.004122 | 0.001515 | 10 |
| IC0 | 25 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01702 | 0.002371 | 0.01702 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01733 | 0.002371 | 0.01702 | 0.000315 | 0.01851 | -0.000315 | 0.001286 | 10 |
| IC0 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03081 | 0.004624 | 0.03128 | 0.0004729 | 0.01512 | 0.0004729 | 0.002414 | 10 |
| IC0 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03128 | 0.004229 | 0.03128 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_mean | 0.0322 | 0.004355 | 0.03128 | 0.0009222 | 0.02948 | -0.0009222 | 0.00233 | 10 |
| IC0 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02233 | 0.002242 | 0.02123 | 0.001099 | 0.05175 | -0.001099 | 0.001685 | 10 |
| IC0 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02123 | 0.001768 | 0.02123 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_sd | 0.01779 | 0.001979 | 0.02123 | 0.003446 | 0.1623 | 0.003446 | 0.001277 | 10 |
| IC0 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.008088 | 0.0009454 | 0.005915 | 0.002173 | 0.3673 | -0.002173 | 0.001126 | 10 |
| IC0 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.005915 | 0.0005785 | 0.005915 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.006749 | 0.0009777 | 0.005915 | 0.0008345 | 0.1411 | -0.0008345 | 0.0009809 | 10 |
| IC0 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03485 | 0.004581 | 0.03418 | 0.0006675 | 0.01953 | -0.0006675 | 0.002734 | 10 |
| IC0 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03418 | 0.004081 | 0.03418 | 0 | 0 | 0 | 0 | 10 |
| IC0 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03564 | 0.004071 | 0.03418 | 0.001456 | 0.04259 | -0.001456 | 0.001808 | 10 |
| IC1 | 25 | 5 | kernel_law_error | LAW-A-K | 0.01742 | 0.0006169 | 0.01694 | 0.000479 | 0.02827 | -0.000479 | 0.0005169 | 10 |
| IC1 | 25 | 10 | kernel_law_error | LAW-A-K | 0.01694 | 0.0004957 | 0.01694 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | kernel_law_error | LAW-A-K | 0.01717 | 0.000512 | 0.01694 | 0.0002244 | 0.01324 | -0.0002244 | 0.0004161 | 10 |
| IC1 | 25 | 5 | mean_quantile_rmse | MEANQ-A-K | 0.09044 | 0.006073 | 0.108 | 0.01758 | 0.1628 | 0.01758 | 0.008984 | 10 |
| IC1 | 25 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.108 | 0.00819 | 0.108 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | mean_quantile_rmse | MEANQ-A-K | 0.0991 | 0.008488 | 0.108 | 0.008912 | 0.08251 | 0.008912 | 0.009814 | 10 |
| IC1 | 25 | 5 | reference_effect_rmse | REF-ATE-K | 0.01434 | 0.00276 | 0.01227 | 0.002078 | 0.1694 | -0.002078 | 0.001867 | 10 |
| IC1 | 25 | 10 | reference_effect_rmse | REF-ATE-K | 0.01227 | 0.00219 | 0.01227 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | reference_effect_rmse | REF-ATE-K | 0.01352 | 0.002726 | 0.01227 | 0.001254 | 0.1022 | -0.001254 | 0.00159 | 10 |
| IC1 | 25 | 5 | reference_tcate_rmse | REF-TCATE-K | 0.03003 | 0.00325 | 0.03085 | 0.0008279 | 0.02683 | 0.0008279 | 0.001443 | 10 |
| IC1 | 25 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.03085 | 0.002588 | 0.03085 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | reference_tcate_rmse | REF-TCATE-K | 0.02805 | 0.003133 | 0.03085 | 0.002803 | 0.09083 | 0.002803 | 0.001364 | 10 |
| IC1 | 25 | 5 | tate_functional_rmse | TATE-K-grid_mean | 0.01696 | 0.00285 | 0.01481 | 0.002151 | 0.1453 | -0.002151 | 0.002126 | 10 |
| IC1 | 25 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01481 | 0.003016 | 0.01481 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | tate_functional_rmse | TATE-K-grid_mean | 0.01485 | 0.003695 | 0.01481 | 4.856e-05 | 0.00328 | -4.856e-05 | 0.001762 | 10 |
| IC1 | 25 | 5 | tate_functional_rmse | TATE-K-grid_sd | 0.00877 | 0.002125 | 0.01178 | 0.003011 | 0.2556 | 0.003011 | 0.002989 | 10 |
| IC1 | 25 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.01178 | 0.002384 | 0.01178 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | tate_functional_rmse | TATE-K-grid_sd | 0.009395 | 0.002209 | 0.01178 | 0.002386 | 0.2026 | 0.002386 | 0.001725 | 10 |
| IC1 | 25 | 5 | tate_functional_rmse | TATE-K-grid_skewness | 0.007841 | 0.002488 | 0.004293 | 0.003548 | 0.8265 | -0.003548 | 0.002282 | 10 |
| IC1 | 25 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.004293 | 0.000926 | 0.004293 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | tate_functional_rmse | TATE-K-grid_skewness | 0.004923 | 0.001133 | 0.004293 | 0.0006303 | 0.1468 | -0.0006303 | 0.001168 | 10 |
| IC1 | 25 | 5 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01771 | 0.002556 | 0.01917 | 0.001461 | 0.07623 | 0.001461 | 0.00219 | 10 |
| IC1 | 25 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01917 | 0.002893 | 0.01917 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01779 | 0.003328 | 0.01917 | 0.001375 | 0.07174 | 0.001375 | 0.002358 | 10 |
| IC1 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03551 | 0.003945 | 0.03484 | 0.0006718 | 0.01928 | -0.0006718 | 0.002165 | 10 |
| IC1 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03484 | 0.003914 | 0.03484 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03178 | 0.004417 | 0.03484 | 0.003064 | 0.08795 | 0.003064 | 0.002158 | 10 |
| IC1 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_sd | 0.01909 | 0.002142 | 0.02118 | 0.002092 | 0.09876 | 0.002092 | 0.002268 | 10 |
| IC1 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02118 | 0.002518 | 0.02118 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02057 | 0.002151 | 0.02118 | 0.0006067 | 0.02865 | 0.0006067 | 0.001213 | 10 |
| IC1 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.01425 | 0.00232 | 0.009865 | 0.004382 | 0.4442 | -0.004382 | 0.001909 | 10 |
| IC1 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.009865 | 0.001028 | 0.009865 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.0107 | 0.001258 | 0.009865 | 0.0008336 | 0.0845 | -0.0008336 | 0.001383 | 10 |
| IC1 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03956 | 0.003389 | 0.03816 | 0.001405 | 0.03681 | -0.001405 | 0.002928 | 10 |
| IC1 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03816 | 0.00398 | 0.03816 | 0 | 0 | 0 | 0 | 10 |
| IC1 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03641 | 0.004408 | 0.03816 | 0.001748 | 0.04582 | 0.001748 | 0.002148 | 10 |
| IC2 | 25 | 5 | kernel_law_error | LAW-A-K | 0.01669 | 0.0006065 | 0.01721 | 0.0005224 | 0.03035 | 0.0005224 | 0.0007079 | 10 |
| IC2 | 25 | 10 | kernel_law_error | LAW-A-K | 0.01721 | 0.0006393 | 0.01721 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | kernel_law_error | LAW-A-K | 0.01737 | 0.0008354 | 0.01721 | 0.0001608 | 0.009341 | -0.0001608 | 0.0007327 | 10 |
| IC2 | 25 | 5 | mean_quantile_rmse | MEANQ-A-K | 0.06998 | 0.003353 | 0.08228 | 0.0123 | 0.1495 | 0.0123 | 0.00706 | 10 |
| IC2 | 25 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.08228 | 0.008736 | 0.08228 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | mean_quantile_rmse | MEANQ-A-K | 0.07397 | 0.007776 | 0.08228 | 0.008313 | 0.101 | 0.008313 | 0.0112 | 10 |
| IC2 | 25 | 5 | reference_effect_rmse | REF-ATE-K | 0.01426 | 0.003274 | 0.01586 | 0.001596 | 0.1007 | 0.001596 | 0.002393 | 10 |
| IC2 | 25 | 10 | reference_effect_rmse | REF-ATE-K | 0.01586 | 0.003535 | 0.01586 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | reference_effect_rmse | REF-ATE-K | 0.01706 | 0.003774 | 0.01586 | 0.001203 | 0.07584 | -0.001203 | 0.002472 | 10 |
| IC2 | 25 | 5 | reference_tcate_rmse | REF-TCATE-K | 0.03298 | 0.003548 | 0.031 | 0.001985 | 0.06405 | -0.001985 | 0.001814 | 10 |
| IC2 | 25 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.031 | 0.003475 | 0.031 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | reference_tcate_rmse | REF-TCATE-K | 0.0339 | 0.003819 | 0.031 | 0.002908 | 0.09381 | -0.002908 | 0.002054 | 10 |
| IC2 | 25 | 5 | tate_functional_rmse | TATE-K-grid_mean | 0.01262 | 0.00361 | 0.01698 | 0.004359 | 0.2567 | 0.004359 | 0.003198 | 10 |
| IC2 | 25 | 10 | tate_functional_rmse | TATE-K-grid_mean | 0.01698 | 0.004234 | 0.01698 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | tate_functional_rmse | TATE-K-grid_mean | 0.0161 | 0.004103 | 0.01698 | 0.000881 | 0.05189 | 0.000881 | 0.002771 | 10 |
| IC2 | 25 | 5 | tate_functional_rmse | TATE-K-grid_sd | 0.01249 | 0.001932 | 0.01424 | 0.001757 | 0.1234 | 0.001757 | 0.001484 | 10 |
| IC2 | 25 | 10 | tate_functional_rmse | TATE-K-grid_sd | 0.01424 | 0.002249 | 0.01424 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | tate_functional_rmse | TATE-K-grid_sd | 0.01273 | 0.001769 | 0.01424 | 0.00151 | 0.106 | 0.00151 | 0.001585 | 10 |
| IC2 | 25 | 5 | tate_functional_rmse | TATE-K-grid_skewness | 0.003013 | 0.0009412 | 0.00288 | 0.0001332 | 0.04623 | -0.0001332 | 0.0009818 | 10 |
| IC2 | 25 | 10 | tate_functional_rmse | TATE-K-grid_skewness | 0.00288 | 0.0006226 | 0.00288 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | tate_functional_rmse | TATE-K-grid_skewness | 0.002615 | 0.000544 | 0.00288 | 0.0002655 | 0.09217 | 0.0002655 | 0.0008688 | 10 |
| IC2 | 25 | 5 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01129 | 0.002938 | 0.0177 | 0.006406 | 0.362 | 0.006406 | 0.002847 | 10 |
| IC2 | 25 | 10 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.0177 | 0.00404 | 0.0177 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | tate_functional_rmse | TATE-K-grid_upper_tail_mean | 0.01394 | 0.003682 | 0.0177 | 0.003753 | 0.2121 | 0.003753 | 0.002473 | 10 |
| IC2 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03527 | 0.003026 | 0.03344 | 0.001826 | 0.05461 | -0.001826 | 0.00331 | 10 |
| IC2 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_mean | 0.03344 | 0.003619 | 0.03344 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_mean | 0.0346 | 0.003231 | 0.03344 | 0.001159 | 0.03465 | -0.001159 | 0.002334 | 10 |
| IC2 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02364 | 0.002187 | 0.02454 | 0.0008985 | 0.03662 | 0.0008985 | 0.001771 | 10 |
| IC2 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02454 | 0.002509 | 0.02454 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_sd | 0.02506 | 0.002227 | 0.02454 | 0.0005231 | 0.02132 | -0.0005231 | 0.001818 | 10 |
| IC2 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.009287 | 0.001028 | 0.009596 | 0.0003091 | 0.03221 | 0.0003091 | 0.002454 | 10 |
| IC2 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.009596 | 0.001555 | 0.009596 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_skewness | 0.008073 | 0.0009177 | 0.009596 | 0.001523 | 0.1587 | 0.001523 | 0.001176 | 10 |
| IC2 | 25 | 5 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03373 | 0.002946 | 0.03415 | 0.0004263 | 0.01248 | 0.0004263 | 0.002907 | 10 |
| IC2 | 25 | 10 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03415 | 0.003008 | 0.03415 | 0 | 0 | 0 | 0 | 10 |
| IC2 | 25 | 25 | tcate_functional_rmse | TCATE-K-grid_upper_tail_mean | 0.03346 | 0.003232 | 0.03415 | 0.0006982 | 0.02044 | 0.0006982 | 0.002462 | 10 |
| IC3 | 25 | 5 | kernel_law_error | LAW-A-K | 0.01802 | 0.0005689 | 0.01879 | 0.0007669 | 0.04082 | 0.0007669 | 0.000458 | 10 |
| IC3 | 25 | 10 | kernel_law_error | LAW-A-K | 0.01879 | 0.000681 | 0.01879 | 0 | 0 | 0 | 0 | 10 |
| IC3 | 25 | 25 | kernel_law_error | LAW-A-K | 0.01926 | 0.0009366 | 0.01879 | 0.0004707 | 0.02505 | -0.0004707 | 0.0006898 | 10 |
| IC3 | 25 | 5 | mean_quantile_rmse | MEANQ-A-K | 0.1103 | 0.004273 | 0.1183 | 0.007971 | 0.06737 | 0.007971 | 0.004368 | 10 |
| IC3 | 25 | 10 | mean_quantile_rmse | MEANQ-A-K | 0.1183 | 0.005502 | 0.1183 | 0 | 0 | 0 | 0 | 10 |
| IC3 | 25 | 25 | mean_quantile_rmse | MEANQ-A-K | 0.1259 | 0.008289 | 0.1183 | 0.007569 | 0.06397 | -0.007569 | 0.00666 | 10 |
| IC3 | 25 | 5 | reference_effect_rmse | REF-ATE-K | 0.01614 | 0.004166 | 0.01816 | 0.002026 | 0.1116 | 0.002026 | 0.002078 | 10 |
| IC3 | 25 | 10 | reference_effect_rmse | REF-ATE-K | 0.01816 | 0.003945 | 0.01816 | 0 | 0 | 0 | 0 | 10 |
| IC3 | 25 | 25 | reference_effect_rmse | REF-ATE-K | 0.0167 | 0.002927 | 0.01816 | 0.001461 | 0.08042 | 0.001461 | 0.002768 | 10 |
| IC3 | 25 | 5 | reference_tcate_rmse | REF-TCATE-K | 0.04626 | 0.006381 | 0.04736 | 0.001107 | 0.02336 | 0.001107 | 0.00292 | 10 |
| IC3 | 25 | 10 | reference_tcate_rmse | REF-TCATE-K | 0.04736 | 0.006127 | 0.04736 | 0 | 0 | 0 | 0 | 10 |
| IC3 | 25 | 25 | reference_tcate_rmse | REF-TCATE-K | 0.04873 | 0.006625 | 0.04736 | 0.001368 | 0.02888 | -0.001368 | 0.003847 | 10 |

_Showing 120 of 144 rows._

## Runtime and memory

| method | n_grid | n_particles | n_runs | mean_runtime_seconds | sum_runtime_seconds | mean_wall_seconds | mean_process_peak_ram_mb | max_process_peak_ram_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| causal_drf | 5 | 10 | 40 | 19.9 | 796.1 | 46.34 | 618.1 | 726.1 |
| causal_drf | 25 | 10 | 280 | 32.2 | 9015 | 40.82 | 478.4 | 726.1 |
| causal_drf | 49 | 10 | 40 | 55.77 | 2231 | 81.43 | 642.5 | 726.1 |
| cwdb_dr | 5 | 5 | 20 | 73.83 | 1477 | 91.3 | 733.4 | 968.3 |
| cwdb_dr | 5 | 10 | 40 | 112 | 4478 | 126.9 | 838.5 | 949.9 |
| cwdb_dr | 5 | 25 | 20 | 460.2 | 9205 | 483.5 | 786.7 | 968.3 |
| cwdb_dr | 25 | 5 | 40 | 150.7 | 6030 | 167.1 | 835.1 | 949.9 |
| cwdb_dr | 25 | 10 | 360 | 299.4 | 1.078e+05 | 306.1 | 645.9 | 955.9 |
| cwdb_dr | 25 | 25 | 40 | 1436 | 5.743e+04 | 1456 | 872.5 | 947.9 |
| cwdb_dr | 49 | 5 | 20 | 307.8 | 6156 | 329.3 | 803.9 | 968.3 |
| cwdb_dr | 49 | 10 | 120 | 544 | 6.528e+04 | 559.1 | 889.4 | 955.9 |
| cwdb_dr | 49 | 25 | 20 | 3527 | 7.053e+04 | 3553 | 956.8 | 1026 |
| cwdb_dr_flex | 25 | 10 | 40 | 262.9 | 1.052e+04 | 265.6 | 400.2 | 458.8 |
| cwdb_dr_flex_rf | 25 | 10 | 40 | 225.9 | 9035 | 228.4 | 454.1 | 477.7 |
| cwdb_dr_oracle | 25 | 10 | 40 | 258.1 | 1.032e+04 | 260.5 | 407.3 | 463.4 |
| drf | 5 | 10 | 40 | 28.17 | 1127 | 50.98 | 618.8 | 726.1 |
| drf | 25 | 10 | 280 | 28.09 | 7865 | 36.35 | 479.3 | 726.1 |
| drf | 49 | 10 | 40 | 38.92 | 1557 | 63.48 | 645.7 | 726.1 |

The complete tables, including paired method comparisons, the factorial surface, native versus common-grid diagnostics, propensity variants, the alignment block with its balance numbers, and the decision payload, are written beside this file as CSV and `decision.json`.
