# Phase G3 repair preregistration (addendum)

**Manifest contract:** `G3-REPAIR-v1`
**Parent manifest contract:** `G3-MAIN-v1` (`research/simulation_preregistration.md`, frozen)
**Estimand contract:** `G0-WP0-A-v1` (unchanged)
**Evaluation manifest:** `G3-EVAL-v1` (unchanged)
**Status:** frozen before the first decisive repair seed
**Date frozen:** 2026-08-01

The G3 tournament returned `NOT-GO` on rule 1. C-WDB-v1's `mean_quantile_rmse`
on D2, where the true treatment effect is exactly null, was 0.147 against a best
baseline of 0.0545, a false-effect ratio of 2.69 against a preregistered cap of
1.25. `research/gates/G3_simulation_memo.md` Section 7 names the repair and its
order: add a contrast-level regulariser, re-test D2 and D8, then re-run the same
frozen manifest. This addendum fixes what that repair will run and what will
count as a fix, before any decisive repair seed exists.

Nothing in the parent preregistration is amended. Every threshold, metric,
eligible regime, decision multiple, and comparator stays exactly as frozen on
2026-07-31. This document adds methods and cells; it removes and relaxes
nothing.

## 1. What is not recomputed

Every baseline, ablation, and C-WDB-v1 row already exists in
`results/merged/main_results.parquet`, checksum
`a871fd7b6ab72544e1f0f317c16cb18551759b575899188c0bfb1d048f9a4a69`. The repair
manifest enumerates cells for the **new methods only**. A repair variant is
compared against the same numbers the G3 memo reports, not against a fresh draw
of them, and a repair run cannot change a published number by re-rolling it.

Repair cells reuse the frozen coordinates exactly: the same grid labels, sample
sizes, grid resolutions, particle counts, and seeds 0 to 19. A repair row and a
frozen row therefore pair seed by seed, so `paired_comparison` removes the
replication effect between a repair variant and a baseline exactly as it does
between two frozen methods.

Cell keys are content addresses of `(grid, dgp, n, K, M, method, seed)`, so a
repair key can never collide with a frozen key, and the repair track writes to
`results/repair/` and `results/merged_repair/`, never into `results/main/` or
`results/merged/`. `tests/test_g3_repair.py` pins all of this, including that
the frozen manifest checksum `5a672a60c382091c` survives the new registry
entries.

## 2. Diagnosis the repair is built on

Two mechanisms produce the false contrast, and they were separated before any
repair was written. Both diagnostics ran on pilot seeds 100 to 104, which the
manifest does not enumerate.

**The initialisation is confounded.** `compute_init_base` is applied per arm, so
each arm starts from the empirical law of its *own* treated or control sample.
That is a marginal quantity. Under a propensity that depends on covariates the
two marginals differ even when the two conditional laws are identical, so the
contrast is nonzero before a single tree is fitted. Measured over 20 samples at
$n=1000$: the initial arm gap has root mean square 0.215 on D2, where the truth
is exactly zero, and 0.824 on D8. The booster then spends its budget removing an
offset the initialisation created.

**The contrast accumulates over the boosting path.** Replaying the fitted path
on a held-out design, D2's contrast error falls to a minimum near iteration 20
and then rises monotonically to the frozen budget of 100: $0.077 \to 0.093 \to
0.104 \to 0.114 \to 0.122$ at iterations $20, 40, 60, 80, 100$ with
`arm_shrinkage = 5`. The same sweep at `arm_shrinkage` of 0, 50, and 500 gives
final errors of 0.161, 0.076, and 0.045, which locates the defect precisely:
`arm_shrinkage` pulls each arm's leaf vector toward the pooled vector, and the
contrast it leaves behind at a balanced leaf is the raw gap times
$n_a/(n_a+\lambda)$, which is 0.86 at a typical main-grid leaf. The contrast is
barely regularised at all, and 100 boosting steps compound what survives.

## 3. Roster

Every repair variant keeps C-WDB-v1's architecture (`v1`, `sharing = "partial"`),
its frozen boosting budget, its energy score, and therefore its repulsion term.
They differ only in how the arm contrast is regularised. No repair claims a new
mechanism; each is a candidate fix for one named defect.

| Label | Registry name | Role | Contrast rule |
|---|---|---|---|
| Pooled initialisation | `cwdb_v1_pooledinit` | ablation | frozen `arm_shrinkage = 5`, shared initial law |
| R1, fixed ridge | `cwdb_r1_ridge` | repair | linear, $\kappa = n_{\text{eff}}/(n_{\text{eff}}+50)$ |
| R2, adaptive threshold | `cwdb_r2_threshold` | repair | positive part at $c=1$, $\kappa = (1 - c\hat\sigma^2/\lVert\delta\rVert^2)_+$ |
| R2', conservative calibration | `cwdb_r2_threshold3` | repair | the same rule at $c=3$ |
| R3, cross-fitted ridge | `cwdb_r3_cvridge` | repair | linear, strength selected on held-out energy risk |

`cwdb_v1_pooledinit` is not a candidate method. It is the mechanism ablation for
the initialisation defect: frozen v1 in every respect except that both arms
start from one pooled base law. Its role is to say how much of the D2 failure
the initialisation alone explains, so the contrast rules are not credited with
a repair the initialisation made.

All three repairs also take the pooled initialisation, because Section 2 shows
the per-arm base is wrong on its own terms. The ablation is what keeps the two
mechanisms separable in the results.

### The common reparameterisation

Rules R1, R2, and R3 write a leaf's arm updates as

$$v_a = \bar g + (a - \pi)\,\kappa\,\delta, \qquad
\delta = \bar g_1 - \bar g_0, \qquad \pi = n_1/(n_0+n_1),$$

with $\bar g$ the leaf's pooled gradient mean. Then $v_1 - v_0 = \kappa\delta$
and $\pi v_1 + (1-\pi) v_0 = \bar g$ identically, so a contrast rule can shrink
the contrast and **cannot** move the pooled component. This is the property that
makes the three rules comparable: they differ only in $\kappa$.

### R1, fixed ridge

$\kappa = n_{\text{eff}}/(n_{\text{eff}} + \lambda_\tau)$ with
$n_{\text{eff}} = n_0n_1/(n_0+n_1)$, the effective sample size the arm gap is
estimated from. This is the posterior mean under a mean-zero Gaussian prior on
the contrast, and is the direct analogue of the BCF half-Cauchy prior the memo
names as what C-WDB lacks.

$\lambda_\tau = 50$. **This constant was chosen by hand on pilot seeds 100 to
104**, at the value that cleared rule 1 with the least damage elsewhere;
$\lambda_\tau \in \{50, 200, 500\}$ were compared. R1 is therefore the tuned
reference in this roster, and R2 and R3 receive no equivalent tuning. Any
advantage the adaptive rules hold over R1 is an advantage held against a tuned
opponent, and any advantage R1 holds over them must be read with its tuning in
view.

### R2, adaptive threshold

$\kappa = \left(1 - c\,\hat\sigma^2 / \lVert\delta\rVert^2\right)_+$ where
$\hat\sigma^2 = \sum_d (s^2_{0d}/n_0 + s^2_{1d}/n_1)$ is the plug-in variance of
the arm gap in that leaf. This is a positive-part James-Stein factor calibrated
against the leaf's own noise: under an exactly null gap
$E\lVert\delta\rVert^2 = \hat\sigma^2$, so the retained fraction is zero in
expectation, while a gap far above the noise passes through undamped.

Two calibrations are entered. $c = 1$ is the null-calibrated value and is the
default; $c = 3$ thresholds at a higher quantile of the same reference
distribution and is the conservative sensitivity. **Neither is tuned**: both are
declared here, both are run, and both are reported. The stage-1 screen in
Section 5 decides which continues, on a criterion fixed before the first
decisive seed, which is a preregistered filter rather than a choice made after
seeing the answer.

Within-leaf covariate heterogeneity inflates $\hat\sigma^2$, so the rule is
conservative in the direction that matters for a null regime. That is a declared
property, not an accident: it costs power in regimes with strong within-leaf
structure and is expected to show up as such.

### R3, cross-fitted ridge

R1's rule with $\lambda_\tau$ chosen on held-out energy risk instead of frozen.
Candidates $\{0, 50, 500\}$ over 2 folds, stratified by arm. Each held-out unit
is scored only against its **own observed arm**, so no counterfactual enters the
selection; the score is strictly proper for the arm law that unit realises. Ties
break toward the stronger regulariser. Assumption A15 is satisfied: every
candidate is scored only on folds excluded from the fit that produced it, and
the final refit uses a strength chosen without seeing any unit's held-out score
twice.

The size of the scan is a **cost** decision and is declared as one. Selection
costs `n_folds * len(candidates)` extra fits. Three folds over four strengths
measured 6.9 times C-WDB-v1's fit time on the pilot seeds, which projects past
rule 6's ceiling of 60 times Causal-DRF's median runtime; two folds over three
strengths is about four ordinary fits. No accuracy result entered this choice.

**Cost correction, recorded before the stage-2 run.** The pilot measured wall
time under nine-way contention and put the two-fold selector at 3.6 times
C-WDB-v1. The stage-1 cells, measured on the tournament's own `runtime` metric
and its own machine, put it at 6.34 times, and projecting that factor over the
repair grid set gives a median of about 75 s against Causal-DRF's 1.17 s, a
ratio near 64 against the cap of 60. The pilot estimate was optimistic and the
budget was set from it. The budget is **not** revised now: revising it after
seeing that R3 is the most accurate repair would be selection dressed as
costing. R3 runs at the declared budget and its rule 6 outcome is reported as it
lands, including a failure by a small margin if that is what happens.

## 4. Grids

| Grid | Stage | DGPs | $n$ | $K$ | $M$ | Seeds | Cells per variant |
|---|---|---|---|---|---|---|---|
| `main` | 1 | D2 | 500, 1000 | 25 | 10 | 20 | 40 |
| `main` | 2 | D0-D9 | 500, 1000 | 25 | 10 | 20 | 400 |
| `smallk` | 2 | D0-D9 | 500 | 5 | 10 | 20 | 200 |
| `shrinkage` | 2 | D2, D8 | 1000 | 25 | 10 | 20 | 40 |

Staged deliberately. Stage 1 is D2 alone, because rule 1 is the only rule the
repair exists to fix and a variant that does not fix it should cost nothing
further. Stage 2 runs every regime and the two ablation grids, and only for
variants that cleared stage 1.

`smallk` carries the repulsion ablation against the squared-$W_2$ booster and
`shrinkage` the causal-regularisation ablation against `arm_shrinkage = 0`, so
a repaired claimant can be scored on the same four mechanism ablations the memo
reports. `particles`, `resolution`, and `scaling` are **not** re-run: they
measure finite-$M$ approximation, grid resolution, and cost scaling, none of
which a contrast rule can plausibly change, and re-running them would spend
compute on a question the repair does not raise. This is a declared limitation,
so any claim about a repaired variant at $M \ne 10$, at $K = 49$, or at
$n = 2000$ is out of scope until those grids are run.

## 5. Decision rule

A repair variant is evaluated by `compute_gate_flags` with that variant named as
claimant, against the frozen `GATE_RULES`. Rule 1's reference stays the best of
`FROZEN_G3_METHODS`, so a repair variant cannot lower the bar it is judged
against by entering the pool.

**Stage 1 screen:** the D2 component of rule 1. D2 `mean_quantile_rmse` at most
0.15 in absolute terms and at most 1.25 times the best frozen baseline's 0.0545,
which is 0.0681. Rule 1's D0 component is not evaluable at stage 1, because D0
is a stage-2 regime; it is checked at stage 2 with everything else. Stage 1 is a
screen on the defect the repair exists to fix, not a verdict.

**Stage 2 pass:** all six rules pass, with rule 1 now including D0 and rules 2 to
6 evaluated exactly as in the parent preregistration.

Variants that clear the stage-1 screen continue. `cwdb_v1_pooledinit` continues
regardless of the screen and is exempt from it: it is the mechanism ablation for
the initialisation, its purpose is attribution rather than candidacy, and its
value comes precisely from showing what the initialisation does and does not fix
across every regime.

Two failure modes are declared in advance because both are live given how these
rules work.

*Fixing rule 1 by breaking rule 2.* Every contrast rule trades a false effect
against a real one. A variant that clears D2 by shrinking every contrast to zero
would also lose the law-level advantage that rule 2 measures. Rules 1 and 2 must
both pass; neither is allowed to be traded for the other, and the paired
comparison against C-WDB-v1 on every regime is reported whether or not it
favours the repair.

*Fixing rule 1 by breaking an ablation.* The sharing ablation compares the
claimant to C-WDB-v0 on D3 and D4, and D3 is the regime built to favour separate
heads. A contrast rule shrinks toward the pooled leaf, which is exactly what D3
punishes. If a repair variant loses D3 to C-WDB-v0 the ablation that supports
the shared-partition mechanism is gone, and that is reported as a cost of the
repair rather than omitted. The ablation is not a gate rule and cannot fail the
gate on its own; it can and will change what the memo may claim.

## 6. Prior measurement

The only measurements taken before this document was frozen are the diagnostics
in Section 2 and an exploratory pilot over eight variant settings, seven
regimes, and five seeds, all on seeds 100 to 104. Those seeds are outside the
manifest, which enumerates seeds 0 to 19 only. The pilot selected R1's
$\lambda_\tau$ (Section 3, declared), fixed R3's fold and candidate count on
cost grounds (Section 3, declared), and produced no decisive ranking. No pilot
number is reported as a result.
