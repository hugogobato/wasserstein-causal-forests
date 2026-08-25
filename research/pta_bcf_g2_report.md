# PTA-BCF Phase 3 G2 Implementation Report

**Report ID:** `PTA-G2-v1`

**Date:** 2026-07-30

**Observation regime:** `ORACLE-V1`

**Decision:** `RETAIN-STRONGEST-ENDPOINT`

**Consequence:** WP2-B4 (joint PTA-BCF sampler) stays `DORMANT` and must not be
implemented. PTA-S (target-specific separate `stochtree` heads) is the endpoint
carried into G3. PTA-F (forced-shared MVBCF) is retained as a mandatory
baseline. PTA-DIAGNOSTIC is retained as a diagnostic instrument only, never as
an inferential method.

## 1. Gate claim and scope

G2 asks whether PTA-BCF can be implemented correctly at both architectural
endpoints and whether a partially pooled composition between them buys anything
real. The gate does not ask whether PTA-BCF beats any external incumbent, and it
does not upgrade the standing novelty verdict. PTA novelty remains
`INCREMENTAL-ONLY` absent a demonstrated adaptive-sharing advantage in G3, and
nothing in this report supplies that advantage.

The target vector is the frozen PTA contract
\(U(Y) = \{q(Y), T_1(Y),\dots,T_J(Y), d_W(q(Y), q(\nu_\star))\}\) with
\(D = K + J + 1\). All reported estimands use grid target IDs from the Phase 0
contract `G0-WP0-A-v1`. No continuum-\(K\) claim is made anywhere in this phase.

This report certifies implementation correctness and one preregistered
architectural crossover. It does not establish consistency, posterior validity,
coverage, continuum recovery, or comparative value against WRF, Causal-DRF, or
C-WDB.

## 2. Implemented work packages

| Work package | Status | Main evidence |
|---|---|---|
| WP2-B1, target contract and separate heads | `PASS` | Frozen `PTA-U-v1` manifest; training-only scale manifest with SHA-256 fingerprint; one scalar `BCFModel` per coordinate; treatment-stratified deterministic folds; contrast-safe inverse scaling; JSON serialization |
| WP2-B2, forced-shared MVBCF baseline | `PASS` | Pinned MIT `mvbcf` revision behind a fixed target-matrix interface; published-cell reproduction inside the preregistered band; \(D \in \{2,4,8\}\) wall-time and peak-RAM curves |
| WP2-B3, diagnostic partial-residual prototype | `PASS` (implemented and verified) | Four-stage cross-fitted composition; out-of-fold weight tuning; full D2-D3-D4 crossover executed; no posterior-uncertainty field emitted |
| WP2-B4, joint sampler | `DORMANT` | Not implemented. The crossover did not return `ENABLE-WP2-B4` |

WP2-B3 passing its mechanical verification is a statement about the
implementation, not about the architecture. The architecture question is settled
in Section 6 and the answer is negative.

## 3. Repository layout and source-reuse deviations

### 3.1 Layout override

The phase file maps the Python modules to `research/pta_bcf/`. Following the
Phase 2 precedent, the Python implementation lives under
`src/wasserstein_causal_forests/pta_bcf/` so it is importable as a package, and
only the R component stays at the planned path `research/pta_bcf/mvbcf_bridge.R`.
Three modules beyond the plan's file map were necessary: `mvbcf.py` (the Python
side of the R bridge), `dgps.py` (the D2, D3, and D4 generators), and `smoke.py`
(the experiment CLI).

### 3.2 Licensing decision, a WP2-B2 precondition

The paper repository `github.com/Nathan-McJames/MVBCF_Paper` declares no
license. Under this project's own fail rule, unlicensed source cannot be used.
The official package `github.com/Nathan-McJames/mvbcf` is MIT licensed and ships
the same `fast_bart` sampler, so WP2-B2 wraps that package at pinned commit
`fc3b89b0a78ce8a31ae75c43a6ec75f1945ca0c8`. No file from the ignored `code/`
directory is copied, sourced, or imported anywhere in this phase.

### 3.3 Deviations inside the published-cell reproduction

Three deviations are recorded because they affect how closely the reproduction
can be expected to match. First, the licensed `run_mvbcf` takes a length-\(n\)
treatment vector and replicates it internally, so the published two-treatment
design is reduced to a single shared treatment. Second, the paper's
\(P(Z=1) \propto \mu\) is resolved as a linear rescaling of \(\mu\) onto
\([0.05, 0.95]\). Third, the stated signal-to-noise ratio is read as a variance
ratio, an interpretive choice made because it reproduces the published
magnitudes.

## 4. Mechanical verification

The clean commands and their recorded results are:

```bash
python3 -m pytest -q                  # 52 passed
Rscript tests/test_mvbcf_bridge.R     # 23 checks, 0 failures
python3 research/checks/check_proper_score.py   # PASS (nine groups)
```

The Phase 0 certificate was rerun unchanged and still returns `PASS`, so this
phase did not disturb the frozen geometry.

The Python tests cover: target recomputation from stored manifests, refusal to
scale with test-fold statistics, the scalar \((K=1, J=0)\) reduction, null
treatment shrinkage, deterministic serialization round trips, pure functional
signal recovery, fold disjointness and treatment stratification, contrast
inverse scaling that reapplies scale without recentering, cross-fitting that
never scores a row with a model fitted on that row, agreement of the diagnostic
and separate-head scale manifests, a residual weight tuner that returns exactly
zero on uninformative residual predictions and exactly one on perfectly
predicted residuals, absence of every posterior-uncertainty column name, and a
two-sided decision rule that returns `INDETERMINATE` on incomplete cells.

The R checks cover the fixed target-matrix interface, draw shapes and burn-in
removal, seed determinism, clean-session execution, and the published-cell
construction invariants.

Two verification defects found and fixed during the phase are worth recording.
`ndarray.tofile` writes C order regardless of the array's memory layout, so the
first bridge silently transposed every array crossing into R; the fix is an
explicit `ravel(order="F")` and the regression test is
`test_binary_exchange_preserves_array_orientation`. Separately, the null
shrinkage check failed at \(n=120\) from small-sample noise rather than
non-convergence, confirmed by ratio stability across 120, 600, and 1200
iterations, and was moved to \(n=400\).

## 5. WP2-B2 evidence

### 5.1 Published-cell reproduction

The manifest is `PTA-F-PUBLISHED-CELL-v1`, 30 replicates, 0 failures, decision
`REPRODUCED`. The preregistered acceptance band is 30% relative to the published
value.

| Metric | Published MVBCF | Observed MVBCF | Relative gap | Within band |
|---|---:|---:|---:|---|
| RMSE \(\mu\) | 1.58 | 1.6975 | +7.4% | yes |
| PEHE \(\tau\) | 0.34 | 0.4262 | +25.4% | yes |
| RMSE \(y\) | 3.97 | 4.5457 | +14.5% | yes |

The published ordering, MVBCF better than univariate BCF on all three metrics,
is reproduced: observed univariate BCF gives 1.8214, 0.5198, and 4.6006.

One earlier run must be reported alongside this one. The first attempt used a
linear logistic propensity and missed the band on PEHE by +51%. That is a
genuine deviation from the paper's probit BART propensity with \(k=3\), not a
tolerance problem. The propensity model was corrected, a `propensity_method`
column was added, and the invalidated run is preserved verbatim at
`results/smoke/mvbcf_published_cell_logistic_propensity.csv`. The acceptance
band was fixed before either run and was not touched.

### 5.2 Dimension scaling

The manifest is `PTA-F-DIMENSION-SCALING-v1`, 3 cells per dimension, 0 failures,
decision `VIABLE` against an 1800 second per-cell limit.

| \(D\) | Mean runtime (s) | Max runtime (s) | Mean peak RAM (MB) |
|---:|---:|---:|---:|
| 2 | 20.4 | 20.8 | 202.5 |
| 4 | 27.4 | 32.6 | 236.0 |
| 8 | 51.0 | 51.8 | 357.6 |

Growth ratios are 1.35 from \(D=2\) to \(D=4\) and 1.86 from \(D=4\) to
\(D=8\). Over this range the dense \(D \times D\) covariance inverse is not yet
the binding cost, which is exactly why the plan required starting at \(D=2\)
rather than at \(K=49\). Nothing here licenses extrapolation to \(K=49\); the
observed ratio is already accelerating.

## 6. Preregistered WP2-B3 crossover

### 6.1 Design, fixed before the run

The manifests are `PTA-CROSSOVER-HYPER-v1` for hyperparameters and
`WP2-B3` for the claim. Settings: \(n_{\mathrm{train}}=400\),
\(n_{\mathrm{test}}=500\), \(K=5\), \(J=2\) (`grid_mean`, `grid_sd`), \(D=8\),
seeds 0 through 4, 4 cross-fitting folds, residual weight grid
\(\{0, 0.25, 0.5, 0.75, 1\}\), 400 Monte Carlo draws for the truth. MVBCF used
600 iterations with 300 burn-in, 50 prognostic and 20 treatment trees. The
separate heads used 50 prognostic and 20 treatment trees, 10 GFR, 100 burn-in,
and 200 MCMC draws. All three methods saw identical data, folds, and seeds.

The metric is `scaled_contrast_rmse` on target
`MEANQ-A-K+TCATE-K-j+REF-TCATE-K`, evaluated against the Monte Carlo truth on
held-out rows and reported on the training scale. Every row carries
`inference = point-prediction-only`.

The regimes are D2 (null treatment effect), D3 (target-specific structure,
favorable to separate heads), and D4 (shared structure, favorable to forced
sharing).

The decision thresholds were fixed before the run and were not modified after
results were seen: the diagnostic must beat PTA-S by at least 2% in D4, beat
PTA-F by at least 2% in D3, and lose no more than 10% against the better
endpoint in D2. All three conditions must hold to enable WP2-B4.

### 6.2 Results

All 45 rows have status `ok`.

| Regime | PTA-S | PTA-F | PTA-DIAGNOSTIC | Preregistered comparison | Verdict |
|---|---:|---:|---:|---|---|
| D4, shared structure | 0.20962 | 0.28685 | 0.22232 | Diagnostic is 6.06% **worse** than PTA-S; needs 2% better | **FAIL** |
| D3, target-specific | 0.31456 | 0.72796 | 0.30509 | Diagnostic is 58.09% better than PTA-F; needs 2% | PASS |
| D2, null | 0.10449 | 0.09114 | 0.09122 | Diagnostic loses 0.09% against the better endpoint; allowed 10% | PASS |

Per-seed evidence, 5 seeds per regime: in D4 PTA-S wins 4 of 5 against the
diagnostic; in D3 the diagnostic wins 5 of 5 against PTA-F and 4 of 5 against
PTA-S; in D2 the diagnostic wins 3 of 5 against PTA-F.

Endpoint against endpoint, PTA-S beats PTA-F by 56.79% in D3 and 26.92% in D4,
and loses by 14.65% in D2. The D2 loss is on a near-zero contrast where all
methods sit between 0.091 and 0.105, so it measures how hard each method
shrinks toward zero rather than how well it recovers signal.

### 6.3 Why the shared-favorable clause failed

The tuned residual weight averages 0.844 in D4, where the correct weight is near
zero because the forced-shared component is already the right model. The
out-of-fold tuner cannot separate "the residual head is fitting signal" from
"the residual head is fitting noise" sharply enough at \(n=400\) with a 5-point
weight grid, so it admits a residual component that degrades an already-correct
shared fit. The corresponding residual share of the predicted contrast magnitude
is 0.244 in D4 against 0.901 in D3, so the composition does modulate, just not
far enough toward zero. Under the null the weight is also high (0.881) but
harmless, because the residual predictions themselves are near zero there.

This is a real limitation of out-of-fold weight tuning at this sample size, not
a coding error. It is the mechanism the gate was designed to detect.

### 6.4 A reporting ambiguity that must not be overread

The machine summary's `retained_endpoint` field reads `PTA-DIAGNOSTIC`. That
field is computed as the method with the lowest unweighted mean across the three
regimes: 0.20621 for the diagnostic, 0.20956 for PTA-S, 0.36865 for PTA-F. Two
cautions apply. First, PTA-DIAGNOSTIC is by construction not an endpoint, so
naming it as the retained endpoint contradicts the plan's own vocabulary and the
`RETAIN-STRONGEST-ENDPOINT` label, which means simplify. Second, its margin over
PTA-S is 1.60%, below the 2% margin this phase preregistered as the smallest
difference it would call real, and it is produced by averaging over a regime set
that was chosen to be adversarial in both directions rather than to represent any
population.

The tie-break code was left unmodified so that the preregistered logic and
constants remain exactly as they were before the results were seen. The correct
reading of the field is recorded here instead: among the two architectural
endpoints, PTA-S is the strongest, and the diagnostic's cross-regime average
advantage over it is not statistically or practically established.

### 6.5 Cost

Mean per-cell runtime is 4.0 seconds for PTA-S, 22.7 seconds for PTA-F, and
105.7 seconds for PTA-DIAGNOSTIC, for 1986.1 seconds total across all 45 cells.
The diagnostic costs roughly 26 times PTA-S for no established accuracy gain.
Cells were run serially so the recorded runtimes are not contaminated by CPU
contention.

The artifacts are `results/smoke/pta_diagnostic_crossover.parquet` and
`results/smoke/pta_diagnostic_crossover.summary.json`. Every Parquet write in
this phase was read back and compared with the in-memory frame before
acceptance.

## 7. G2 decision

The claimed contribution tested at this gate is that partially pooled
target-augmented BCF can adapt between forced sharing and target-specific
fitting and thereby beat both endpoints. The evidence does not support that
claim.

The decision is `RETAIN-STRONGEST-ENDPOINT`. WP2-B1 and WP2-B2 pass their
mechanical checks, so both endpoints are correctly implemented and neither needs
debugging or removal. WP2-B3 is correctly implemented and its preregistered
crossover fails on the shared-favorable clause: the composition is 6.06% worse
than separate heads exactly where partial pooling was supposed to help.

WP2-B4 therefore stays `DORMANT`. Per the phase file it may not be implemented
without a passing diagnostic, and there is no passing diagnostic. Building a
joint sampler now would be building the expensive version of an architecture
whose cheap version has already failed its own test.

PTA-S enters G3 as the PTA architecture. PTA-F is retained as a mandatory
baseline because it wins D2 and because it is the published comparator.
PTA-DIAGNOSTIC is retained as a diagnostic instrument for reading how much
target-specific structure a dataset carries, never as an inferential method: it
produces no posterior and its output frame deliberately contains no coverage,
interval, or credible-set field.

## 8. What this does not settle

The crossover used 5 seeds at \(n=400\) with \(D=8\) on three synthetic regimes.
It is a gate, not a benchmark. Three specific limits should be carried forward.

The D4 failure is tied to weight-tuner variance at \(n=400\); a larger sample, a
finer weight grid, or a shrinkage prior on the weight could change the sign of
that comparison. If G3 has a reason to revisit partial pooling, that is the
experiment to run, and it needs its own preregistration rather than a reanalysis
of these rows.

The dimension-scaling evidence stops at \(D=8\) with an accelerating growth
ratio, so the cost of PTA-F at the full \(K=49\) grid is unmeasured.

The published-cell reproduction carries three documented deviations and clears a
30% band, which is enough to certify that the bridge drives the sampler
correctly and not enough to claim an exact replication.

Finally, this gate says nothing about PTA-BCF against WRF, Causal-DRF, or C-WDB.
The standing `INCREMENTAL-ONLY` novelty verdict is unchanged, and G3 can still
reduce PTA-BCF to `KILL`.
