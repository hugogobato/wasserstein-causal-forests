# Phase G3 simulation preregistration

**Manifest contract:** `G3-MAIN-v1`
**Estimand contract:** `G0-WP0-A-v1` (`research/estimand_contract.md`, frozen)
**Evaluation manifest:** `G3-EVAL-v1`
**Status:** frozen before the first decisive seed
**Date frozen:** 2026-07-31

This document fixes what the tournament will run, what it will measure, how a
metric may and may not be interpreted, and what counts as a win, before any
result that could rank the methods exists. The only prior measurement is the
cost pilot in Section 8, which ran one regime (D6) at seed 9999, a seed the
manifest does not enumerate, and recorded wall time and peak memory only.

## 1. Roster

| Label | Registry name | Role | Produces a conditional law |
|---|---|---|---|
| C-WDB-v1 | `cwdb_v1` | claimant | yes |
| C-WDB-v0 | `cwdb_v0` | ablation, independent-arm limit | yes |
| C-WDB-v1 without arm shrinkage | `cwdb_v1_noshrink` | ablation, `arm_shrinkage = 0` | yes |
| Squared-$W_2$ booster | `sqw2_booster` | ablation, repulsion removed | yes, degenerate by construction |
| PTA-S | `pta_s` | baseline, separately tuned scalar heads | no |
| PTA-F | `pta_f` | baseline, forced-shared MVBCF | no |
| W-DRF-T | `wdrft` | baseline, two ordinary predictive DRFs | yes |
| Causal-DRF | `causal_drf` | baseline, published incumbent | yes |

C-WDB-v1 is the only model that can return `GO`. The baselines are held to
correctness and cost reporting; none of them is a claim of this project.

Two restrictions carried in from earlier phases hold here without exception.
No adaptive-sharing claim may be made for PTA-S or PTA-F, which enter as the two
surviving endpoints of a line that returned `RETAIN-STRONGEST-ENDPOINT`. No
claim about band coverage relative to the published two-forest benchmark may be
made for Causal-DRF, whose coverage contrast was not reproduced; its point
estimation was, and only law-level and functional risk comparisons are used.

## 2. Data generating processes

Ten regimes, D0 through D9, are implemented in
`src/wasserstein_causal_forests/g3/dgps.py`. Every regime shares one generative
form,

$$q(Y^a)_k = m_a(x) + \xi + e^{s_a(x)+\eta}\,\psi(z_k;\gamma_a(x)),$$

with $z_k$ the standard normal quantile at grid level $u_k$, the outer latent
pair $(\xi,\eta)$ independent of $x$ and $A$ given the arm, and
$\psi(z;\gamma) = z + \gamma\,\mathrm{He}_2(z)/2 + \gamma\,\mathrm{He}_3(z)/6$.
Both Hermite terms are orthogonal to the identity under the standard normal, so
$\psi$ leaves the mean alone while moving skewness and the upper tail, and its
minimum slope is $1-\gamma$, so every draw stays strictly inside the monotone
cone for $0\le\gamma<1$ at any $K$.

The regimes differ only in $m_a$, $s_a$, $\gamma_a$, the outer law, and the
propensity. D0 is deterministic, D1 is the smooth stochastic reference, D2 has
an exactly null effect, D3 gives the two arms disjoint covariate supports, D4
puts every arm's structure on one moderator, D5 matches the arms' barycenters
exactly while differing in law, D6 gives the outer location a two-component
mixture, D7 places the entire treatment effect in the inner law's shape, D8
combines strong confounding with a weak effect, and D9 lets overlap deteriorate
to propensities at the clipping bounds.

The moderator is $V=g(X)$, a four-bin discretisation of $X_0$ at the cut points
$-0.5$, $0$, and $0.5$. A discrete moderator keeps every `TCATE-K-j` statement a
finite collection of conditional means, which is what the contract's
prohibition on claims at every continuous $v$ requires.

## 3. Oracle truth

Conditional targets are computed by Gauss-Hermite quadrature over the outer
latent pair, using twelve nodes per non-degenerate dimension, rather than by
Monte Carlo. The truth that enters an RMSE denominator therefore carries
deterministic quadrature error rather than sampling noise.
`research/checks/g3_dgp_truth_accuracy.py` measures that error against a
50,000-draw Monte Carlo at 140 regime-arm-target combinations and returns
`PASS`, with a worst discrepancy of 0.0179 against Monte Carlo standard errors
in the range 0.001 to 0.007. A twelve-node rule agrees with a twenty-node rule
to $7.5\times10^{-3}$ across every regime and target.

One target is not computed by quadrature. The `tail_calibration` truth is the
expectation of an indicator, and no Gaussian quadrature rule integrates a step
function; the generic path was wrong by 0.069 against a Monte Carlo standard
error of 0.002. The outer location is therefore integrated in closed form
through the normal survival function and only the smooth log-scale latent goes
through the quadrature rule.

## 4. Estimands, metrics, and what each method may claim

Every result row carries a target identifier from
`research/estimand_contract.md` Section 2 and a metric identifier from its
Section 7. Operational rows use `target_id = NONE_OPERATIONAL`.

The mandatory common target is the grid causal mean
$\tau_q^K(x)=E\{q(Y^1)-q(Y^0)\mid X=x\}$, reported as `mean_quantile_rmse`
against `MEANQ-A-K`. Arm-level barycenter error is reported separately as
`barycenter_rmse`, and the two identifiers are never pooled even where they are
numerically related.

Law-level metrics are `arm_energy_risk` (reported as excess energy risk over the
true law, since the raw risk contains an irreducible regime-dependent term),
`kernel_law_error` (squared MMD under a Gaussian kernel on the rescaled
coordinates, with a median-distance bandwidth so the discrepancy is scale
equivariant in the outcome's units), `tail_calibration`, and `mode_coverage`.

**PTA-S and PTA-F supply none of these.** They estimate conditional means of a
fixed target vector, and contract Section 4 forbids relabelling a posterior draw
of a mean surface as a draw from the outcome law. Their law-level rows carry
`status = "not_applicable"` with that reason recorded. This is a finding about
what a direct target learner can deliver, not a gap to be filled with a
substitute quantity.

The two forest baselines are scored on their native representation, a weighted
empirical law over the training sample, and C-WDB on its $M$ particles. Both are
valid predictive distributions and the energy score is proper, so a richer
representation is allowed to win; each row records its atom count so the reader
can see which is which. C-WDB's fixed-$M$ restriction is a real property of the
method, and its $M$-sensitivity is reported separately as the contract requires.

## 5. Training manifest and the transfer test

Every method is trained on a target manifest containing the $K$ quantile
coordinates, the functionals `grid_mean` and `grid_sd`, and the reference
distance to the standard normal, so $D = K + 3$.

`grid_skewness` and `grid_upper_tail_mean` are deliberately **excluded** from
every training manifest and evaluated anyway. A method that outputs a full law
can integrate them at evaluation time; a method with fixed target coordinates
cannot, and reports `not_applicable`. D7 is built so that the location and scale
surfaces agree across arms and the entire treatment effect sits in these two
unseen functionals.

## 6. Grids

| Grid | Purpose | DGPs | $n$ | $K$ | $M$ | Methods | Seeds |
|---|---|---|---|---|---|---|---|
| `main` | primary tournament | D0-D9 | 500, 1000 | 25 | 10 | v1, v0, W-DRF-T, Causal-DRF, PTA-S | 20 |
| `smallk` | full roster including PTA-F | D0-D9 | 500 | 5 | 10 | all seven | 20 |
| `particles` | finite-particle claim | D1, D6 | 1000 | 25 | 2, 5, 10, 25 | v1, squared-$W_2$ | 20 |
| `resolution` | grid-resolution sensitivity | D1, D5, D6, D7 | 1000 | 49 | 10 | the four law methods | 10 |
| `shrinkage` | causal regularization claim | D2, D8 | 1000 | 25 | 10 | v1, v1 without shrinkage | 20 |
| `scaling` | runtime and memory at the largest $n$ | D1, D4, D6 | 2000 | 25 | 10 | four law methods, PTA-S | 10 |

**PTA-F appears only in `smallk`.** Its cost accelerates in the target dimension
$D=K+J+1$: Phase 3 measured 20.4 s at $D=2$, 27.4 s at $D=4$, and 51.0 s at
$D=8$, with the ratio itself rising from 1.35 to 1.86 because of the dense
residual covariance. At $K=25$ the dimension is 28, which no safe compute budget
reaches. The Phase G3 computational notes anticipate exactly this and cap PTA-F
at $D\in\{2,4,8\}$; restricting it to $K=5$ gives $D=8$ and keeps it in a grid
where every other method also runs. Its absence from the $K=25$ grids is a
declared cost limitation and is reported as such, never as a loss.

Every method in a replication receives the same training sample and is evaluated
on the same test design of 1000 points, drawn at seed $900000+s$, which is
disjoint from every training seed.

## 7. Frozen budgets

Boosting, shared by every C-WDB variant and by the squared-$W_2$ comparator:
100 trees, learning rate 0.12, maximum depth 4, minimum leaf size 10, minimum
arm leaf 5, collision parameter $\varepsilon=10^{-3}$, arm shrinkage 5.0 except
in the shrinkage ablation where it is 0.

W-DRF-T: 1000 trees, minimum node size 15, honesty on, the pinned CRAN `drf`
1.3.1. Causal-DRF: 2500 trees, minimum arm leaf 5, and
`bandwidth_rule = "median_distance"`, which the computational notes require
because the package convention is not scale equivariant and would tie the kernel
to the arbitrary units of the outcome distributions. PTA-S: 50 prognostic and
20 treatment trees, 10 grow-from-root, 100 burn-in, 200 MCMC draws, five
cross-fitting folds. PTA-F: 1000 iterations, 500 burn-in, 50 prognostic and
20 treatment trees.

No budget depends on the regime, the seed, or the sample size. The validator in
`research/checks/g3_manifest_validator.py` rejects a registry entry that carries
a regime-dependent parameter, which is how comparable tuning effort is enforced
rather than asserted.

## 8. Cost basis

Measured by `research/checks/g3_cost_pilot.py` on regime D6 at seed 9999, one
thread per process. D6 is the most expensive truth in the suite, because its
outer law is a mixture and its quadrature rule therefore has twice the nodes, so
the estimate is conservative. Per-cell wall time includes fitting, prediction,
and the full evaluation against oracle truth.

| Method | $n$ | $K$ | $M$ | Wall seconds per cell | Peak RSS delta (MB) | Status |
|---|---|---|---|---|---|---|
| `cwdb_v1` | 500 | 25 | 10 | 12.5 | 70 | ok |
| `cwdb_v1` | 1000 | 25 | 10 | 20.9 | 0 | ok |
| `cwdb_v0` | 500 | 25 | 10 | 8.6 | 0 | ok |
| `cwdb_v0` | 1000 | 25 | 10 | 13.4 | 0 | ok |
| `wdrft` | 500 | 25 | 10 | 11.1 | 214 | ok |
| `wdrft` | 1000 | 25 | 10 | 18.5 | 235 | ok |
| `causal_drf` | 500 | 25 | 10 | 10.5 | 75 | ok |
| `causal_drf` | 1000 | 25 | 10 | 17.3 | 99 | ok |
| `pta_s` | 500 | 25 | 10 | 33.5 | 75 | ok |
| `pta_s` | 1000 | 25 | 10 | 47.4 | 66 | ok |
| `cwdb_v1` | 500 | 5 | 10 | 5.5 | 0 | ok |
| `cwdb_v0` | 500 | 5 | 10 | 4.4 | 0 | ok |
| `sqw2_booster` | 500 | 5 | 10 | 3.1 | 0 | ok |
| `wdrft` | 500 | 5 | 10 | 8.7 | 214 | ok |
| `causal_drf` | 500 | 5 | 10 | 7.9 | 75 | ok |
| `pta_s` | 500 | 5 | 10 | 9.8 | 0 | ok |
| `pta_f` | 500 | 5 | 10 | 95.5 | 0 | ok |
| `cwdb_v1` | 1000 | 25 | 2 | 4.8 | 0 | ok |
| `cwdb_v1` | 1000 | 25 | 5 | 7.8 | 0 | ok |
| `cwdb_v1` | 1000 | 25 | 25 | 50.0 | 0 | ok |
| `sqw2_booster` | 1000 | 25 | 2 | 2.8 | 0 | ok |
| `sqw2_booster` | 1000 | 25 | 5 | 2.8 | 0 | ok |
| `sqw2_booster` | 1000 | 25 | 10 | 2.8 | 0 | ok |
| `sqw2_booster` | 1000 | 25 | 25 | 2.9 | 0 | ok |
| `cwdb_v1` | 1000 | 49 | 10 | 22.2 | 0 | ok |
| `cwdb_v0` | 1000 | 49 | 10 | 15.3 | 0 | ok |
| `wdrft` | 1000 | 49 | 10 | 12.3 | 239 | ok |
| `causal_drf` | 1000 | 49 | 10 | 11.5 | 92 | ok |
| `cwdb_v1_noshrink` | 1000 | 25 | 10 | 12.2 | 0 | ok |
| `cwdb_v1` | 2000 | 25 | 10 | 23.9 | 0 | ok |
| `cwdb_v0` | 2000 | 25 | 10 | 16.0 | 0 | ok |
| `wdrft` | 2000 | 25 | 10 | 20.9 | 276 | ok |
| `causal_drf` | 2000 | 25 | 10 | 33.6 | 123 | ok |
| `pta_s` | 2000 | 25 | 10 | 76.8 | 0 | ok |

Projected total, costing every manifest cell at its own measured
shape where one exists and at that method's worst measured shape
otherwise (0 cells fall in the second case):

| Method | Cells | Projected CPU hours |
|---|---|---|
| `pta_s` | 630 | 5.7 |
| `pta_f` | 200 | 5.3 |
| `cwdb_v1` | 870 | 3.8 |
| `wdrft` | 670 | 2.4 |
| `causal_drf` | 670 | 2.4 |
| `cwdb_v0` | 670 | 1.8 |
| `sqw2_booster` | 360 | 0.3 |
| `cwdb_v1_noshrink` | 40 | 0.1 |
| **total** | 4110 | **21.8** |

At 6 workers that is about 3.6 wall hours. The projection is conservative in two ways: it costs every cell on regime D6, whose mixture outer law gives the quadrature truth twice the nodes of any other regime, and it charges every cell a full oracle evaluation, whereas the dispatcher keeps a replication's methods in one worker so they share one cached oracle.

Two structural costs were found and removed before this basis was measured. The
oracle energy risk is identical for every method in a replication and originally
dominated the cell, so it is now computed once per replication through a cache
and through the closed form $E_{y\sim P}S_\varepsilon(P,y)=\tfrac12\sum_{j,l}
\omega_j\omega_l d_\varepsilon(t_j,t_l)$; the dispatcher keeps a replication's
methods in one worker so the cache hits. Pairwise distances in the metric layer
go through the inner-product identity so they reach BLAS. Together these took a
representative cell from 80.6 s to 20.9 s.

Execution uses six worker processes with every numerical library pinned to one
thread. Free physical memory was roughly 4 GB during Phase 4 and was measured at
4 GB again here, so six workers rather than ten is the safe default; the peak
resident figure per worker is in the table above.

## 9. Failure handling

A cell that raises is recorded as a failure row carrying its exception type,
message, and a truncated traceback, and it is never retried under a different
seed. The manifest fixes which seeds exist. Silently replacing a seed a method
failed on would convert a robustness result into a selection artefact, which is
the trap this work package names.

The merge in `wasserstein_causal_forests.g3.merge` refuses rather than cleans. A
duplicate cell key, a result key absent from the manifest, a manifest cell with
no rows, or a row disagreeing with the manifest on its own coordinates makes the
audit `FAIL`. Failed cells stay in the merged table as failures.

## 10. Decision rules

All six must hold for `GO`. Every threshold below is fixed as of this document
and appears in `GATE_RULES` in the frozen manifest.

A win means a seed-paired difference more negative than twice its paired
standard error, where methods are paired within a replication because they share
a training sample and a test design. `mode_coverage` is sign-inverted so that
every reported effect has one convention.

1. **Correctness and nulls.** On D0, which is deterministic, C-WDB-v1's
   `mean_quantile_rmse` is at most 0.15. On D2, which has an exactly null
   effect, it is at most 0.15 and at most 1.25 times the best baseline's, so a
   difficulty shared by every method is not charged to C-WDB alone.
2. **Law advantage.** C-WDB-v1 beats Causal-DRF on `kernel_law_error`, the
   declared primary law metric, in at least two of D1, D5, D6, D7.
3. **Transfer.** The advantage reaches at least one of
   `tcate_functional_rmse`, `reference_tcate_rmse`, or
   `reference_effect_rmse` in at least one of D5, D6, D7, against Causal-DRF.
4. **Beats the direct target learner.** On a target rule 3 actually won,
   C-WDB-v1 also beats PTA-S. Evaluating rule 4 only on rule 3's winners is
   deliberate: passing by beating PTA-S somewhere C-WDB lost to Causal-DRF would
   not be the claimed transfer.
5. **No collapse.** On D6, `mode_coverage` is at least 0.90 and the particle
   weights' participation ratio is at least 0.60 of $M$.
6. **Cost commensurate.** C-WDB-v1's median runtime is at most 60 times
   Causal-DRF's, and the memo states the size of the advantage that cost buys.

Failing any rule yields `PIVOT`, `INCREMENTAL-ONLY`, or `KILL`, argued in the
memo. The gate flags are recomputed by
`research/checks/g3_gate_flags.py`, which shares only the thresholds with the
analysis code and reimplements the statistics, so a bug shows up as a
disagreement rather than as two copies of one wrong answer.

## 11. What this design cannot settle

The `Uncertainty usable` row of the claims matrix asks for coverage and width
against a bootstrap or incumbent. C-WDB has no interval construction: none was
built in Phase 2 and none is built here. That row is therefore **not evaluated**
and is reported as unevaluated in the memo rather than approximated by a
posterior-draw quantity that would violate contract Section 4. It is the first
item of Phase 6 work if the gate opens.

Causal-DRF's coverage limitation from Phase 4 compounds this: even a comparison
of interval widths would have no admissible incumbent here.

The `resolution` grid varies $K$ between 25 and 49 but not to 99. At $K=99$ the
per-cell cost of the law metrics rises with $K$ while PTA-S's head count rises
to 102, and the cost pilot places that beyond the budget. Grid-resolution
conclusions are therefore stated for $K\in\{5,25,49\}$ only.
