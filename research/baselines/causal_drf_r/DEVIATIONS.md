# Causal-DRF Reimplementation: Deviations from the Paper

**Reference:** Näf, Park, Susmann (2026), *Causal-DRF: Conditional Kernel
Treatment Effect Estimation using Distributional Random Forest*, AISTATS 2026
(PMLR v300), arXiv:2411.08778v2.
**Implementation:** `CAUSAL-DRF-REIMPLEMENTATION-v1`
**Author code:** none published; see `research/baselines/PROVENANCE.md`.

WP2-C2 requires that every deviation from the paper be recorded, because the
manuscript will claim results *against* this baseline and reimplementation
error must not be mistaken for a method difference. Deviations are graded:

- **Underdetermined** — the paper does not fix the choice; a choice was made.
- **Corrected** — the paper's text and its own software disagree; both are
  implemented and the applicable one is selected explicitly.
- **Reduced** — implemented at smaller scale than published, for cost.
- **Not implemented** — out of scope, with the consequence stated.

---

## 1. Implemented exactly as published

For the avoidance of doubt, these are transcriptions, not reinterpretations:

- The weighted-MMD split criterion of eq. (2) / eq. (C.1), including the
  `|I_L| |I_R| / (|I_L| + |I_R|)^2` normalization and the arm-normalized signed
  weights `nu_{i,L} = W_i / |I_L, W=1| - (1 - W_i) / |I_L, W=0|`.
- The random Fourier approximation of eq. (C.2), with `omega_b` drawn from the
  Gaussian spectral measure and the criterion averaged over the `S` frequencies.
- The signed prediction weights of eq. (3), averaged over trees.
- Half-sampling and subforest grouping: `B` groups, `L = round(N / B)` trees per
  group, each group fitted on `S = {i : U_i = 1}` with `U_i ~ Bernoulli(1/2)`.
- Honesty (F1): each tree's subsample is split, one part determining the splits
  and the other populating the leaves.
- Both-arm minimums and alpha-regularity (F4), with `alpha <= 0.2` enforced.
- The resampled statistic, test, and uniform confidence band of eqs. (4), (5),
  (6), using `C = sup_y k(y, y) = 1` for the Gaussian kernel.

`tests/test_causal_drf.R` pins each of these, including a closed-form hand case
for the criterion and a brute-force check that the compiled search returns the
argmax of the criterion it approximates.

---

## 2. Corrected: the median-heuristic bandwidth

**Status:** Corrected. Both conventions implemented; selected explicitly.

Appendix C states: "The bandwidth `sigma` is chosen as the median pairwise
distance between all training responses (the 'median heuristic';
Gretton et al. 2012)." The `drf` package the authors build on computes
something else:

```r
drf:::medianHeuristic <- function(Y) median(sqrt(dist(Y) / 2))
# and then
bandwidth <- sqrt(medianHeuristic(Y))
```

that is, `sqrt(median(sqrt(dist(Y) / 2)))`. On the published simulation design
the two differ by roughly a factor of two (median distance `~1.87`, package
chain `~0.98`), and the witness function is kernel dependent, so the choice
moves every reported number.

**Which is which was settled empirically.** At regime 3, `n = 250`, the
published Causal-DRF witness MAE is `0.065`. This implementation returns
`0.064` under the package chain and `0.038` under the median distance. The
published results were therefore produced with the package chain, whatever the
appendix says.

**Consequence, and why both are kept.** The package chain is *not scale
equivariant*: scaling the response by `c` scales the bandwidth by `c^(1/4)`.
That is tolerable inside `drf`, which standardizes the response by default, but
it is not tolerable on this project's rescaled quantile representation `z`,
whose entire purpose is to carry an exact and comparable `W_2` scale. A
non-equivariant bandwidth would make the kernel depend on the arbitrary units
of the outcome distributions and would break comparability across DGPs.

So:

| Use | Rule | Reason |
|---|---|---|
| Reproducing published cells | `drf_package` | Matches what the authors ran |
| G3 tournament on quantile vectors | `median_distance` | Equivariant; the default of `causal_drf_fit` |

Both are asserted in `tests/test_causal_drf.R`, including the equivariance
property that separates them.

---

## 3. Corrected: the Fourier spectral density

**Status:** Corrected, following the paper.

Appendix C specifies `omega_1, ..., omega_S ~ N_d(0, sigma^-2 I_d)`, i.e.
per-coordinate standard deviation `1 / sigma`. The `drf` source draws from

```cpp
std::normal_distribution<double> distribution(0.0, 1.0/(bandwidth*bandwidth));
```

whose second argument is the **standard deviation**, not the variance, giving
`sd = 1 / sigma^2`. This implementation follows the paper. The discrepancy is
recorded because it affects the split criterion of the `drf` package itself,
and therefore the W-DRF-T baseline, which is used as shipped.

## 4. Underdetermined: when the Fourier frequencies are redrawn

**Status:** Underdetermined; drawn once per forest.

The paper does not say whether `omega` is redrawn per node, per tree, or once.
The `drf` source constructs the frequencies inside `find_best_split`, but from a
default-constructed `std::default_random_engine`, so in practice the same
frequencies are produced at every node. This implementation draws them once per
forest from the seeded R stream, which reproduces that effective behaviour and
is deterministic under a fixed seed.

## 5. Underdetermined: which sample carries (F4)

**Status:** Underdetermined; both, with the stated sample binding.

(F4) states alpha-regularity and the per-arm leaf counts "for the second sample
in (F1)", the leaf-populating sample. (F1) permits `(W_i, X_i)` from that second
sample to be used in defining the splits, which is what makes the constraint
enforceable at split time. This implementation checks a candidate split against
both samples: `min_arm_build` (default 2) per arm in the split-determining
sample, needed for the criterion to be defined at all, and `min_arm_leaf`
(default 5, the paper's `kappa`) per arm plus `alpha` in the populating sample.

The paper's upper bound in (F4), leaves holding *between* `kappa` and
`2 kappa - 1` units per arm, is a lower bound only here: the recursion stops
when no admissible split exists rather than forcing further splits to shrink
oversized leaves. This is the standard `grf` behaviour and cannot inflate the
apparent quality of the baseline, since it yields coarser, not finer, leaves.

## 6. Underdetermined: how subforest grouping composes with subsampling

**Status:** Underdetermined; composed as `grf` composes it.

Algorithm 1 subsamples twice: once per group (the Bernoulli half-sample) and
once per tree (F5). Taken literally with `sample.fraction = 0.5` at each stage,
a tree would see `n / 4` units and, after honesty, split on `n / 8`. Trees would
be near-stumps at the published sample sizes. `grf` and `drf` instead size the
per-tree subsample against the full sample when confidence groups are active.
This implementation does the same: the per-tree subsample targets
`sample_fraction * n` and is drawn from the group's half-sample, so a tree
splits on `n / 4` units, matching what `drf` would do at the same settings.

(F5) itself asks for `s_n = n^beta` with `beta` above a bound that evaluates to
about `0.997` for `p = 5, alpha = 0.05`, i.e. essentially the whole sample. No
forest implementation in this family, including `grf` and `drf`, honours that;
the asymptotic condition is not a practical setting and is not adopted here.

## 7. Underdetermined: the random-split rule (F2)

**Status:** Underdetermined; uniform `mtry` draw.

(F2) requires the probability of splitting on any covariate to be bounded below
by `pi / p`. This is implemented as a uniform draw of `mtry` candidate
covariates at every node, so each covariate is a candidate with probability
`mtry / p`. The default `mtry` follows `drf`,
`min(ceiling(sqrt(p) + 20), p)`, which equals `p` for the small `p` used here.

---

## 8. Reduced: reproduction scale

**Status:** Reduced, deliberately.

| | Published | Here |
|---|---|---|
| Replications | 500 | 200 |
| Sample sizes | 250, 500, 1000, 5000 | 250, 1000 |
| Regimes | 1-4 plus two heavy-tailed variants | 1, 3, 4 |
| Trees / groups | `N = 2500`, `B = 50` | identical |

The retained cells span the null regime, the effect regime, and the
confounding-plus-effect regime at the smallest and a middle sample size. Monte
Carlo standard errors are reported alongside every reproduced number in
`research/drf_baselines_g2_report.md`, so the comparison against the published
table is made at stated precision rather than by eye.

## 9. Not implemented: the benchmark's uncertainty construction

**Status:** Not implemented. This is the one material fidelity gap.

The paper's DRF benchmark is Näf et al. (2023): two DRFs fitted separately per
arm, each carrying that paper's own half-sampling uncertainty construction. The
headline contrast of Table 3 is that under a treatment effect this benchmark
undercovers badly (77.6-84.4% at the nominal 95%) while Causal-DRF holds near
nominal.

What is implemented here instead is W-DRF-T under **Causal-DRF's own** group
structure: for each of the `B` half-samples, two arm forests are fitted and the
resulting signed weights enter exactly the same eqs. (4)-(6) inference. This
makes the two baselines differ in exactly one respect, whether the partition is
shared and treatment-aware, which is the comparison WP2-C1 and WP2-C2 exist to
support.

**Consequence.** The published *point estimation* result reproduces for both
methods (see the G2 report). The published *undercoverage* of the benchmark does
not, because the benchmark's bands here are built by a different, more
conservative resampling scheme than Näf et al. (2023)'s. Therefore:

- No claim may be made in this project that W-DRF-T undercovers, nor that
  Causal-DRF's coverage advantage over the published DRF benchmark was
  reproduced.
- Coverage comparisons against Näf et al. (2023) are out of scope until that
  construction is implemented. Point-estimation and law-level comparisons are
  unaffected.

Implementing Näf et al. (2023)'s two-forest uncertainty is tracked as a
follow-up and is not required for G3, whose primary metrics are law-level and
functional risks rather than the benchmark's band coverage.

## 10. Not implemented: the applied 401(k) study

Section 5 of the paper analyses SIPP 1991 401(k) data. That is an applied
result, not a mechanism check, and it belongs to Phase G4 if at all. Not
required for G2.
