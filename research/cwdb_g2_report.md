# C-WDB Phase 2 G2 Implementation Report

**Report ID:** `CWDB-G2-v1`

**Date:** 2026-07-30

**Observation regime:** `ORACLE-V1`

**Decision:** `GO`

**Consequence:** C-WDB enters G3 with v1 as the proposed architecture. C-WDB-v0 remains a mandatory ablation and fallback.

## 1. Gate claim and scope

G2 asks whether the narrowly scoped C-WDB method can be implemented without
changing the frozen estimand, dropping the ensemble repulsion gradient,
imputing counterfactual gradients, violating the monotone quantile cone, or
creating a causal interpretation for particle labels. The implementation
targets the fixed-\(M\), collision-smoothed score projection
\(P_{a,M,\varepsilon}^{K,\star}(x)\) from the Phase 0 contract.

This report certifies implementation correctness and one small architectural
ablation. It does not establish comparative value against WRF, Causal-DRF, or
other G3 baselines, consistency, inference, continuum-\(K\) recovery, or
robustness to finite inner samples.

## 2. Implemented work packages

| Work package | Status | Main evidence |
|---|---|---|
| WP2-A1, score and geometry | `PASS` | Exact weighted rescaling; weighted PAVA; certified smoothed score; full attraction and repulsion gradient; collision-safe zero subgradient; projection diagnostics |
| WP2-A2, independent-arm v0 | `PASS` | Two independent arm boosters; deterministic empirical initialization; projected preconditioned gradient steps; backtracking descent; canonical public particles; law-invariant summaries |
| WP2-A3, shared-partition v1 | `PASS` | Pooled observed-arm gradient SSE splits; child arm-count constraints; partially pooled arm leaf vectors; forced-sharing and no-sharing limits; fair-budget held-out ablation |

All implementation code is under `src/wasserstein_causal_forests/`. The
repository's ignored `code/` directory is not used. No R component is needed.

## 3. Source-reuse deviation

Phase 1 deliberately deferred WP1-B, so the planned WGBoost snapshot did not
have a completed repository provenance record. Phase 2 therefore uses a
clean-room implementation derived from the frozen score equations and public
scikit-learn tree interfaces. No WGBoost file is copied, vendored, or imported.
This removes WP1-B as a dependency of the C-WDB source itself. It does not
complete the provenance work still needed for external incumbent
implementations in later phases.

## 4. Mechanical verification

The clean test command is:

```bash
python -m pytest -q
```

The recorded result is `27 passed`. The tests cover the following checks:

1. analytic gradients agree with central finite differences within
\(10^{-6}\) for \(\varepsilon=0\) and \(10^{-3}\), 2. score output is invariant
and gradients are equivariant under particle permutations within
\(10^{-12}\), 3. weighted rescaling gives the exact finite-grid distance, 4.
collisions have finite scores and gradients, 5. weighted isotonic projection
returns monotone vectors and diagnostics, 6. a projected preconditioned step
descends in the clean case, 7. the proper score distinguishes a two-mode law
from its barycenter, 8. v0 is exactly two direct arm-wise fits, 9. fitting is
row-order invariant and deterministic under a fixed seed, 10. pickle
serialization preserves predictions, 11. public particles are canonical and
summaries are law-invariant, 12. shared split gains match hand calculations,
13. every shared child satisfies arm counts, 14. forced sharing and no sharing
reduce to their declared limits, and 15. a zero-shrinkage null-arm gradient
produces a null update.

The upstream executable Phase 0 certificate was rerun:

```bash
python research/checks/check_proper_score.py
```

It returned `PASS` for all nine groups. Its largest smooth-gradient error was
\(7.55\times10^{-11}\), particle permutation error was
\(2.22\times10^{-16}\), and projection KKT residual was
\(1.73\times10^{-17}\).

## 5. Preregistered shared-partition ablation

The valid manifest is `CWDB-G2-SHARED-ABLATION-v2`. An initial pilot used 20
trees per arm for v0 but only 20 total trees for v1. That pilot violated the
identical-tree-budget requirement and was invalidated before the gate verdict.
The v2 manifest corrected only this protocol defect and retained the announced
decision thresholds and all other settings.

The valid experiment uses \(n_{\mathrm{train}}=160\),
\(n_{\mathrm{test}}=1000\), \(K=5\), \(M=5\), seeds 0 through 9, 40 total
trees per fitted model, depth 2, learning rate 0.12, and
\(\varepsilon=10^{-3}\). C-WDB-v0 receives 20 trees in each arm. C-WDB-v1
receives 40 shared-partition trees. Every method uses the same score, total
tree budget, seed set, train and test construction, and fixed
hyperparameters.

The gate thresholds were fixed before the run: v1 must reduce mean held-out
energy risk by at least 2% in the shared-structure DGP and may increase risk by
at most 5% in the separate-structure DGP.

| DGP | v0 mean risk | v1 mean risk | Relative v1 change | Seed evidence | Gate comparison |
|---|---:|---:|---:|---|---|
| Shared structure | 0.335372 | 0.317116 | 5.443% lower | v1 wins 10 of 10; paired absolute improvement 95% interval [0.013425, 0.023087] | Passes the 2% improvement threshold |
| Separate structure | 0.338825 | 0.341051 | 0.657% higher | v1 wins 3 of 10; paired absolute v0-minus-v1 95% interval [-0.007875, 0.003423] | Passes the 5% loss tolerance |

All 40 result rows have status `ok`. Each method used and accepted its full
40-tree budget. Recorded cumulative cell runtime was 115.7 seconds, maximum
single-cell runtime was 5.51 seconds, and process peak RAM was 181.4 MB.

The result artifact is
`results/smoke/cwdb_shared_ablation.parquet`; the machine-readable decision is
`results/smoke/cwdb_shared_ablation.summary.json`. The Parquet file was read
back and compared with the in-memory frame before acceptance.

## 6. G2 decision

The claimed contribution tested at this gate is an implementable
treatment-aware proper-score particle learner whose shared covariate
partitions improve estimation when arm laws share structure without material
loss when their active structures differ.

The evidence supports `GO` for G2. WP2-A1, WP2-A2, and WP2-A3 pass their
mechanical checks. Under the preregistered and fair total-tree budget, v1
clears both architectural thresholds. C-WDB therefore enters G3 with v1 as
the proposed architecture.

The decision is deliberately narrow. G3 must retain v0, forced sharing, and
no sharing as ablations, and must compare C-WDB with WRF and faithful
Causal-DRF under common information and tuning budgets. It must also vary
\(M\), \(K\), overlap, arm imbalance, misspecification, and computational
scale. Failure there can still reduce C-WDB to `INCREMENTAL-ONLY` or `KILL`.
