# DRF Baselines Phase 4 G2 Implementation Report

**Date:** 2026-07-30
**Phase:** `research_phases/Phase 4 - DRF Baselines Implementation.md`
**Gate:** G2, are the DRF baselines correctly implemented?
**Decision:** `ENTER-G3-AS-MANDATORY-INCUMBENTS`, with one scoped restriction on
band-coverage claims.

| Artefact | Path |
|---|---|
| Shared geometry and prediction schema | `research/baselines/baseline_common.R` |
| W-DRF-T (WP2-C1) | `research/baselines/drf_tlearner.R` |
| Causal-DRF (WP2-C2) | `research/baselines/causal_drf_r/` |
| Provenance and licensing | `research/baselines/PROVENANCE.md` |
| Deviations from the paper | `research/baselines/causal_drf_r/DEVIATIONS.md` |
| Replicate-level results | `results/smoke/causal_drf_reproduction.parquet` |
| Machine-readable verdict | `results/smoke/causal_drf_reproduction.summary.json` |

---

## 1. Gate claim and scope

This phase builds the two forest baselines C-WDB must be measured against. It
makes no claim of its own. The scientific point of the gate is narrow and
adversarial: C-WDB cannot claim value against a weakened or mislabelled version
of its closest published competitor, so the competitor has to be built well
enough that reimplementation error is not the dominant source of tournament
uncertainty.

Two distinct objects were built and must never be conflated.

**W-DRF-T** fits one *ordinary predictive* DRF per treatment arm on the
rescaled quantile coordinates. There is no shared partition and no
treatment-aware split criterion. It is the cheapest full-law forest comparator.

**Causal-DRF** fits ONE forest whose split criterion targets the conditional
kernel treatment effect directly, with signed prediction weights. It is the
direct-hit published incumbent of Näf, Park and Susmann (2026).

The `W-` prefix on either records the input geometry only. Since
`||z - z'||^2 = W_{2,K}^2(q, q')` for `z = diag(sqrt(w)) q`, the Gaussian kernel
these forests already use is exactly a discretized Wasserstein-RBF kernel. That
is a representation map, not a contribution, and the manuscript must not present
it as one.

## 2. Implemented work packages

| Package | Status | Evidence |
|---|---|---|
| WP2-C1, two-arm Wasserstein DRF baseline | Complete | `tests/test_drf_tlearner.R`, 33 checks, 0 failures |
| WP2-C2, faithful shared Causal-DRF | Complete | `tests/test_causal_drf.R`, 47 checks, 0 failures |
| WP2-C3, Du et al. WRF adapter | `DORMANT`, not implemented | Phase 1 never closed the applicability determination |

WP2-C3 is untouched. `research/baselines/wrf.R` does not exist and nothing in
this phase may be read as a WRF baseline.

## 3. Provenance, licensing, and layout

### 3.1 The pinned DRF revision

`drf` 1.3.1 from CRAN, GPL-3.0, source tarball SHA256
`6b3e9bf28d636fa9be012602883e236723813ec7be19b5e5d64a33251455279d`.

The local `code/drf-master` snapshot was **not** used to build anything. It is
unbuildable in this checkout: `r-package/drf/src/` relies on symbolic links into
`bindings/` and `core/src/`, and in this working copy those links were flattened
into one-line text files holding their target paths. It is retained as read-only
E3 evidence for EV4 only.

The GPL-3.0 isolation boundary is stated in `research/baselines/PROVENANCE.md`:
`drf` is reached through its exported R API only, no source is vendored, and the
Causal-DRF reimplementation neither loads its namespace nor shares code with it.

### 3.2 No author code exists for Causal-DRF

The bounded acquisition queue item "Causal-DRF author artifacts, checked once at
G2" is **closed with a negative result**. The AISTATS page records
`Code Dataset Promise: No`, and the paper's own checklist leaves both the source
code item and the reproduction instructions item unresolved. The documented
reimplementation stands.

The arXiv PDF carries only the eleven-page main body; the appendices holding the
exact splitting criterion and the algorithm pseudocode were read from the LaTeX
source tarball. Both artefacts are checksummed in `PROVENANCE.md`.

### 3.3 Layout additions beyond the phase file map

Three, each recorded in the phase file. `baseline_common.R` holds the geometry
and prediction-schema helpers both baselines share, so neither depends on the
other. `causal_drf_r/causal_tree.cpp` is a compiled split search, adopted after
a pure-R recursion measured fifteen times slower at `n = 1000`; G3 runs this
baseline in every cell, so the cost mattered. `src/wasserstein_causal_forests/
baselines/reproduction.py` holds the Python result schema, matching the Phase 3
precedent of an R engine behind a Python parquet writer.

## 4. Mechanical verification

### 4.1 The geometry identity

Maximum squared-distance error `1.8e-15` and maximum kernel error `2.2e-16`
across all pairs of a sixty-row quantile matrix, with equal and with unequal
quadrature weights. This is the E4 check both work packages require.

### 4.2 W-DRF-T

The canonical `?drf` example reproduces: forest weight rows sum to one to
`8.9e-16`, the conditional mean read off the weights equals the package's own
`functional = "mean"` output exactly, and the fitted `E[Y_1 | X]` correlates
`0.97` with the `X_1` the example builds it from.

Arm separation is exact. Treated weights place zero mass on control units and
conversely, the two arm forests are fitted on disjoint index sets whose union is
the sample, and the same seed reproduces every weight and every derived summary
to the bit.

One non-default setting is required and is not a tuning choice. `drf` sets
`response.scaling = TRUE` by default, which standardizes each response column
and would replace the exact `W_2` geometry of `z` with an arbitrary
coordinatewise rescaling. It is forced off, and the test suite asserts it.

### 4.3 Causal-DRF

Every mechanism named in the phase file is implemented and pinned by a test.

**Split criterion.** A four-unit hand case with one treated and one control on
each side has a closed form; the reference implementation matches it to
`1e-12`. The criterion is zero when the two sides carry the same contrast, and
zero when a child holds only one arm. The Fourier-approximated criterion the
compiled search actually uses converges to the exact-kernel criterion as the
frequency count grows, and a brute-force sweep over every admissible
`(variable, value)` pair confirms that the compiled search returns the argmax of
the criterion it approximates.

**Both-arm minimums.** Every populated leaf of a fitted forest holds at least
`min_arm_leaf` units of each arm. This is a correctness requirement, not a
regularization choice: the arm normalizations in the criterion and in the
prediction weights are undefined when a child holds no unit of one arm.
Alpha-regularity is enforced and an `alpha` above the paper's `0.2` bound is
rejected.

**Honesty.** The union of a tree's leaf members equals its leaf-populating
sample exactly and intersects its split-determining sample in the empty set.

**Signed weights.** The weights of eq. (3) split into two arm weight vectors,
each normalized to one and each supported on its own arm, so the shared,
CKTE-targeted partition still yields usable arm summaries. They carry an
explicit `shared_partition = TRUE` flag so no downstream table can silently
present them as W-DRF-T's independently fitted arm laws.

**Subforest grouping.** Each of the `B` groups yields normalized weights of its
own, the groups hold an equal share of the trees, and the pooled weights equal
their average to `1e-10`.

**Null behaviour.** Under the null regime the CKTE norm is an order of magnitude
below its value under an effect, the witness function is flat near zero, and the
resampling test does not reject. Under an effect the test rejects and the
witness has the correct sign on either side of the common centre.

### 4.4 A latent crash in the pinned package

`drf` defaults `ci.group.size` to `as.integer(num.trees / 30)`, which is zero
for fewer than thirty trees and makes its C++ core divide by zero. The process
dies with SIGFPE and no R-level error. Both call sites now pass an explicit
positive value. This is recorded because any G3 shard that tunes the tree count
downward would otherwise hit it.

## 5. Published-cell reproduction

### 5.1 Design, fixed before the run

Target: Näf, Park, Susmann (2026), Appendix B, Table 3, the conditional witness
function at the fixed test point `x = (0.7, 0.3, 0.5, 0.68, 0.43)`.

Regimes 1 (no confounding, no effect), 3 (effect), and 4 (confounding and
effect), at `n` in `{250, 1000}`, with the published `N = 2500` trees and
`B = 50` groups, and 200 replications per cell. Both methods share one
bandwidth, one Gram matrix, and one evaluation grid in every replication,
because the witness is kernel dependent and anything else would compare two
estimands rather than two estimators.

The rule, fixed before results were seen: each reproduced cell mean must lie
within 20% of its published value **or** within three Monte Carlo standard
errors of it; the test must hold type-I error at or below the nominal 5% under
the null regimes and retain power under the effect regimes. The rule is
two-sided on purpose. A reimplementation that is much *better* than the
published method is as much a fidelity failure as one that is worse, because the
tournament would then be comparing against something the literature does not
contain.

### 5.2 A bandwidth discrepancy that had to be settled first

Appendix C states that the bandwidth is "the median pairwise distance between
all training responses". The `drf` package the authors build on computes
`sqrt(median(sqrt(dist(Y) / 2)))` instead. On this design the two differ by
roughly a factor of two, and the witness is kernel dependent, so the choice
moves every published number.

Which one produced the published table was settled empirically. At regime 3,
`n = 250`, the published Causal-DRF MAE is `0.065`. This implementation returns
`0.064` under the package chain and `0.038` under the median distance. The
published results were produced with the package chain.

Both are implemented and selected explicitly, because the package chain is not
scale equivariant: scaling the response by `c` scales it by `c^(1/4)`. That is
tolerable inside `drf`, which standardizes the response by default. It is not
tolerable on the rescaled quantile representation, whose entire purpose is to
carry an exact and comparable `W_2` scale. The reproduction uses the package
chain; **G3 must use `bandwidth_rule = "median_distance"`.**

### 5.3 Results

1200 replications, 2400 rows, zero failures.

**Witness-function mean absolute error.** Published values in parentheses.

| Regime | n | Causal-DRF | (published) | gap | W-DRF-T | (published DRF) | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 no effect | 250 | 0.0356 | (0.041) | -13.1% | 0.0351 | (0.035) | +0.4% |
| 1 no effect | 1000 | 0.0267 | (0.029) | -7.8% | 0.0286 | (0.027) | +5.9% |
| 3 effect | 250 | 0.0685 | (0.065) | +5.3% | 0.0679 | (0.066) | +2.8% |
| 3 effect | 1000 | 0.0523 | (0.053) | -1.3% | 0.0563 | (0.054) | +4.2% |
| 4 both | 250 | 0.0784 | (0.070) | +12.0% | 0.0774 | (0.072) | +7.5% |
| 4 both | 1000 | 0.0606 | (0.053) | +14.3% | 0.0619 | (0.055) | +12.6% |

Every cell is inside the 20% tolerance for both methods; the worst gaps are
14.3% (Causal-DRF) and 12.6% (W-DRF-T). Monte Carlo standard errors on the MAE
run from `0.0009` to `0.0016`, so several of these gaps, particularly in regime
4, are statistically distinguishable from the published values even though they
are small. That is expected and is not hidden here: `kappa`, `mtry`, and the
exact composition of the two subsampling stages are underdetermined by the paper
and no reimplementation can pin them down. The gaps are reported at stated
precision rather than declared to be noise.

**The directional finding reproduces.** The paper's substantive point about
point estimation is that the two methods are close under the null and that
Causal-DRF pulls ahead once there is an effect. Here Causal-DRF is 7.1% better
than W-DRF-T at regime 3, `n = 1000`, and 2.1% better at regime 4, `n = 1000`,
while being marginally worse in the smallest null cell.

**Coverage of the 95% band.**

| Regime | n | Causal-DRF | (published) | W-DRF-T | (published DRF) |
|---|---:|---:|---:|---:|---:|
| 1 | 250 | 1.000 | (1.000) | 1.000 | (1.000) |
| 1 | 1000 | 1.000 | (1.000) | 1.000 | (1.000) |
| 3 | 250 | 0.970 | (0.970) | 0.955 | (0.782) |
| 3 | 1000 | 0.990 | (0.974) | 0.995 | (0.844) |
| 4 | 250 | 0.975 | (0.974) | 0.930 | (0.776) |
| 4 | 1000 | 0.970 | (0.968) | 0.980 | (0.918) |

Causal-DRF's coverage reproduces closely, within 1.6 percentage points in every
cell and exactly in one. **W-DRF-T's does not**, and Section 5.5 says why.

**Validity of the test.** Under the null regimes the rejection rate is 0.010 at
`n = 250` and 0.005 at `n = 1000`, both below the nominal 0.05. Under every
effect regime it is 1.000. The test is valid and powered.

### 5.4 Cost

Measured single-threaded, `N = 2500` trees, on the reproduction cells.

| n | Causal-DRF fit + inference | W-DRF-T fit + inference |
|---:|---:|---:|
| 250 | 1.4 - 1.8 s | 1.2 - 4.3 s |
| 1000 | 5.4 - 5.8 s | 3.0 - 3.1 s |

Carrying each covariate's sort order down the recursion, instead of re-sorting
every candidate covariate at every node, reduced the `n = 1000` Causal-DRF fit
from 35 s to 2.4 s. The remainder of the measured time is the inference step,
which builds the `n x n` Gram matrix and the `B` resampled statistics.

The full reproduction ran in about 35 minutes on six workers. Six, not ten:
free physical memory was measured at roughly 4 GB with other work already
resident on the machine.

### 5.5 The one material fidelity gap

The published DRF benchmark is Näf et al. (2023): two DRFs fitted separately per
arm, each carrying *that* paper's own half-sampling uncertainty construction.
Its headline behaviour in Table 3 is severe undercoverage under an effect,
77.6% to 84.4% against a nominal 95%.

What is implemented here is W-DRF-T under **Causal-DRF's own** group structure:
for each of the `B` half-samples, two arm forests are fitted, and the resulting
signed weights enter exactly the same eqs. (4)-(6) inference as Causal-DRF. That
design was chosen so the two baselines differ in exactly one respect, whether
the partition is shared and treatment-aware, which is the comparison WP2-C1 and
WP2-C2 exist to support.

The consequence is visible in the coverage table. The *direction* is right, as
W-DRF-T's bands are consistently narrower than Causal-DRF's (half-width 0.187 vs
0.238 at regime 3, `n = 250`) and its coverage does degrade in the hardest cell
(0.930 vs 0.975 at regime 4, `n = 250`). But the magnitude is nowhere near the
published 77.6%, because the resampling scheme used here is more conservative
than Näf et al. (2023)'s.

**Restriction carried into G3.** No claim may be made in this project that
W-DRF-T undercovers, nor that Causal-DRF's coverage advantage over the published
DRF benchmark was reproduced. Coverage comparisons against Näf et al. (2023) are
out of scope until that construction is implemented. Point-estimation and
law-level comparisons, which are G3's primary metrics, are unaffected.

## 6. G2 decision

| Condition | Outcome |
|---|---|
| Both W-DRF-T and Causal-DRF reproduce their expected mechanisms | **Observed: enter G3 as mandatory incumbents** |
| W-DRF-T passes but Causal-DRF fidelity unresolved | Not observed |
| Neither passes | Not observed |

`ENTER-G3-AS-MANDATORY-INCUMBENTS`. The machine-readable verdict in
`results/smoke/causal_drf_reproduction.summary.json` reads
`CAUSAL-DRF-FIDELITY-ESTABLISHED`, on all four of its clauses: MAE within
tolerance in every cell, type-I error controlled at 0.010, power 1.000, and
coverage at or above nominal.

Both baselines enter G3. The restriction of Section 5.5 travels with them.

## 7. What this does not settle

- **The benchmark's uncertainty construction.** Näf et al. (2023)'s two-forest
  scheme is not implemented. Until it is, the published coverage contrast is an
  unreproduced result, not a refuted one.
- **WP2-C3.** No WRF adapter exists. If Du et al.'s method turns out to apply to
  quantile-vector outcomes, a third mandatory baseline is still owed.
- **Behaviour at tournament scale.** Every number here is at `d = 1`, the
  published design. The G3 representation is a `K`-dimensional quantile vector.
  The split search cost is independent of `K` by construction, since the
  response enters only through precomputed Fourier features, but the *statistical*
  behaviour at `K = 25` upward is untested and belongs to the WP3-A cost pilot.
- **Tuning parity.** Both baselines run at the published or `drf`-default
  settings. WP3-A must give them the same tuning budget as C-WDB, or the
  tournament compares a tuned method against untuned incumbents.
