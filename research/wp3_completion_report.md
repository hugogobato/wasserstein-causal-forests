# WP3 completion report

**Date:** 2026-07-27  
**Input:** `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md`, WP3  
**Scope:** finite-grid pre-simulation prototype only  
**Overall status:** `PARTIAL`, not a theory or inference completion certificate

## Deliverables

| Task | Output | Status |
|---|---|---|
| WP3-T1 | `research/algorithm_spec.md` | PASS for the corrected finite prototype specification |
| WP3-T2 | `research/wp3_t2_split_gain.md`, `research/wp3_odcf.py` | PASS for the finite considered-candidate gain identity only |
| WP3-T3 | `research/wp3_scaling_selection.md`, `research/wp3_scaling.py` | PASS for the implementation decision `robust_sd`; C3.2 remains pending |
| WP3-T4 | `oracle_dr_scores` and oracle forest path in `research/wp3_odcf.py` | PASS for the finite oracle interface and DGP sanity check |
| WP3-T5 | `cross_fitted_dr_scores` and fold-disjointness assertions | PASS for stratified region-fold separation and known-propensity bypass |
| WP3-T6 | `HonestTree`, `honesty_report`, and index assertions | PASS for score/index honesty conditional on fixed scores and split-side scales |
| WP3-T7 | arm-specific AIPW scores, separate arm curve forests, weighted `pava`, and unconstrained effect curves | PASS for the finite diagnostic interface |
| WP3-T8 | curve-only, composite, specialized, and MMD-on-score variants | PARTIAL, official Causal-DRF baseline remains mandatory in WP9 |
| WP3-T9 | `research/method_equivalence_memo.md` | PASS, novelty warning recorded |
| WP3-T10 | provisional within-region bootstrap and direct-score noise heuristic | PENDING, experimental and excluded from primary validity claims |

## Checks

The following commands form the minimum finite implementation gate. They pass together after the WP3 repairs:

```text
python3 research/wp2_invariants.py
python3 research/wp3_invariants.py
python3 research/wp3_scaling.py
python3 -m py_compile research/wp3_odcf.py research/wp3_invariants.py research/wp3_scaling.py
```

Passing these commands establishes only the assertions encoded in the scripts. In particular, the suite must separately check the finite candidate-gain identity, an end-to-end pure-functional score construction, frozen trapezoidal weights, exact grid-duplication invariance, treatment-arm support in every nuisance-training fold, the known-propensity classifier bypass, disjoint split/populate indices, nonempty balanced populate children, weighted PAVA, unprojected effect curves, API consistency, the direct-score scale used by the experimental SSE noise heuristic, and explicit rejection of that unsupported option for `mmd_score`.

Index disjointness does not establish full raw-outcome honesty. Cross-fitted nuisance models for split observations can depend on other regions that later appear in a tree's populate sample, and global preprocessing can also couple the two sides. Any stronger claim requires an independent nuisance/calibration sample, a stronger multiway split, or a separate asymptotic argument.

## Mathematical and scientific boundary

The finite composite calculation proves that a considered candidate partition has positive empirical gain when at least one active scaled child mean differs. It does not prove population detection, greedy selection of that partition, forest consistency, or empirical superiority.

The scientific score and target vector are unscaled. `robust_sd` affects split geometry only and is an implementation decision, not an asymptotic result. The shared finite-grid forest is conditionally equivalent to an ordinary weighted multi-output regression forest on fixed scores and scales. The curve loss is a finite trapezoidal approximation to a trimmed log-scale \(L^2\) quantile loss, not full \(W_2^2\).

Therefore the project must not claim ODCF-v1's shared partition as standalone novelty. The local `mmd_score` comparator is DRF-inspired but is not official DRF or Causal-DRF. Any theory opened after G2A must explain a demonstrated advantage involving the causal score, nonlinear pre-averaging, inner-sample uncertainty, or confirmatory inference.

WP4–WP8 remain locked until the empirical promise gate in the plan is passed. WP3-T10 remains outside the primary tournament unless explicitly labeled experimental. No rate, consistency, full-honesty, CLT, bootstrap-validity, or inner-sample-validity claim is made here.

`PARTIAL`
