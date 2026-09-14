# Project STAR applied study (WCF): final report

Repo root: `/home/hugo_souto/Stuff/Research/Wasserstein_Causal_Forests`
All fits used the frozen `run_wcf` spec with `extra_functionals={p10, p90}` and
the pre-registered unit/covariate/moderator choices in `PREREGISTRATION.md`.
All numbers below are copied from the JSON/CSV outputs in
`results/applied_study_exploration/project_star/` (paths listed in Section 6).

## 1. Data

Sources (verified byte-identical to local copies by sha256 on 2026-09-13):

- Student file: `STAR_Students.tab`, Harvard Dataverse DOI 10.7910/DVN/SIWH9F,
  file id 666716, URL https://dataverse.harvard.edu/api/access/datafile/666716
  (CC0), sha256 `769be163...d3ba97`. 11,601 students x 379 columns.
- School file: `STAR_K-3_Schools.tab`, same DOI, file id 666717, URL
  https://dataverse.harvard.edu/api/access/datafile/666717 (CC0), sha256
  `776f2d3c...89cba9`. 80 schools x 53 columns.

Schema used: `g?classtype` (1 small, 2 regular, 3 regular+aide), `g?schid`,
`g?tchid`, `g?tgen` (1 male/2 female), `g?trace` (1 White, 2 Black, 3 Asian),
`g?thighdegree` (1-6), `g?tyears`, `g?tmathss`, `g?treadss`, `g?freelunch`
(1 free, 2 non-free), `gender` (1 male/2 female), `race` (1 White, 2 Black,
other pooled), school file `var1` (urbanicity 1 inner city, 2 suburban,
3 rural, 4 urban) and `G?FRLNCH` = var9/var21/var33/var45 (percent
free/reduced lunch by grade). No negative missing codes; empty = system missing.

Unit: one classroom-grade, unique `(grade in {K,1,2,3}, g?schid, g?tchid)`.
Verified: classtype constant within all 1,340 raw units; one kindergarten unit
had two teacher-race values (modal value used, flagged in units CSV).
Filter: at least 5 tested students. Seven units had zero tested math students
(all 5 classes of school 168211 grade 2, plus 1 regular class each in schools
209510 and 231616 at grade 3); dropped, leaving **n = 1,333 units**
(**522 small, 811 regular/aide**; regular only = 400, aide = 411), covering
**24,610 tested students**. By grade: K 325, G1 339, G2 335, G3 334 units.
Tested students per unit: small mean 14.1 (median 14, range 7-20), control mean
21.3 (median 21, range 13-29).

Missing scores are excluded from Q (no imputation). Math missing shares by arm
(small/regular): K 7.3%/7.1%, G1 3.0%/3.5%, G2 11.3%/11.4%, G3 10.9%/10.6%.
Reading missing shares: K 8.5%/8.5%, G1 5.3%/6.8%, G2 11.0%/11.2%,
G3 11.9%/11.7%. Missingness is balanced across arms (max gap 1.5 pp).

Moderator: school free/reduced-lunch share = mean over grades of the official
school-file percentages (never missing), ecdf-transformed into (-1,1), so the
frozen edges (-0.5, 0, 0.5) are school free-lunch quartiles. Raw quartile
ranges: Q1 0.02-0.31 (mean 0.21), Q2 0.31-0.41, Q3 0.41-0.63, Q4 0.64-0.97
(mean 0.88); bins contain 333/333/334/333 units.

## 2. Design

- A = 1 for `g?classtype == 1`, A = 0 for classtype in {2,3} (pooled);
  robustness 1 vs 2 only. Every school-grade has both arms.
- Q = 25 empirical quantiles at u_k=(k-0.5)/25 of the unit's tested students'
  math scaled scores, standardized within grade (mean 0, sd 1 over all tested
  students, both arms) so effects are in within-grade student SD units and
  grades are pooled. Reading is the robustness outcome.
- X (p = 9, moderator first): school free-lunch quartile (ecdf); teacher female;
  teacher non-white; teacher graduate degree; teacher total experience;
  school urbanicity; class share female; class share Black; class share free
  lunch. Class size is excluded (determined by treatment). Balance (small minus
  control, standardized): teacher experience -0.012, share female +0.063,
  share Black -0.074, share free lunch -0.115, school FL -0.043, urbanicity
  +0.060; class size -3.89 (by design). Within-school balance (small minus
  control, school-FE regression with school-clustered SEs): teacher experience
  -0.044 (0.459), share female +0.005 (0.009), share Black -0.003 (0.006),
  share free lunch -0.023 (0.011), teacher non-white +0.009 (0.020),
  graduate degree -0.047 (0.027); all small in magnitude, with the free-lunch
  share the only nominal 2-sigma deviation.
- q_star: equal-weight average over grades of pooled control-student quantile
  vectors (primary); all-student benchmark (sensitivity).
- Estimands: TATE and 4-bin TCATE for mean, sd, skew, upper-half mean (tail),
  reference distance to q_star, plus p10 (u=0.10) and p90 (u=0.90) coordinates.
- Propensity: cross-fitted logistic (5 folds, clipped [0.02,0.98]); DR scores
  with influence-function SEs; plug-in particle contrasts also reported.

## 3. WCF results (primary, 3 seeds, frozen defaults)

Marginal DR contrasts (mean over seeds; IF SE is the seed-mean; plug-in is the
seed-mean particle contrast):

| functional | DR | seed SD | IF SE | plug-in |
|---|---|---|---|---|
| mean | +0.161 | 0.007 | 0.028 | +0.150 |
| sd | -0.001 | 0.001 | 0.011 | -0.011 |
| skew | +0.019 | 0.008 | 0.031 | +0.024 |
| tail | +0.166 | 0.006 | 0.031 | +0.148 |
| reference | +0.053 | 0.004 | 0.017 | +0.045 |
| p10 | +0.091 | 0.008 | 0.029 | +0.098 |
| p90 | +0.179 | 0.005 | 0.036 | +0.165 |

4-bin DR contrasts by school free-lunch quartile (seed mean; SE from seed 0):

| functional | Q1 low FL | Q2 | Q3 | Q4 high FL |
|---|---|---|---|---|
| mean | +0.164 (0.056) | +0.092 (0.047) | +0.126 (0.057) | +0.262 (0.062) |
| sd | -0.010 (0.022) | +0.019 (0.022) | -0.027 (0.020) | +0.014 (0.024) |
| skew | +0.038 (0.063) | -0.081 (0.057) | +0.130 (0.062) | -0.011 (0.063) |
| tail | +0.161 (0.064) | +0.112 (0.054) | +0.108 (0.063) | +0.282 (0.070) |
| reference | +0.176 (0.038) | +0.060 (0.027) | +0.078 (0.036) | -0.103 (0.036) |
| p10 | +0.119 (0.055) | -0.050 (0.055) | +0.116 (0.062) | +0.179 (0.064) |
| p90 | +0.173 (0.075) | +0.115 (0.065) | +0.117 (0.069) | +0.311 (0.078) |

Reading: location effects are similar. The mean DR is +0.151 (SE 0.026),
tail +0.154, p10 +0.076, p90 +0.150, sd -0.002, reference +0.062. The p10
effect is about half the p90 effect in both math and reading, consistent with
an upper-tail-dominated location shift.

School-saturated X (post-hoc, 79 school dummies, p=88, seed 0): mean +0.173
(SE 0.031), tail +0.182, p90 +0.201, p10 +0.093, sd +0.006, reference +0.066;
shrinkage 50; propensity range 0.144-0.674 with no clipping and no unit
outside [0.1,0.9]. With the propensity now able to recover each school's
assignment fraction exactly, the mean estimate moves by only +0.012 (0.4 of
the primary IF SE), so the coarse school aggregates in the primary X do not
drive the result.

Propensity/overlap: per-seed minimum 0.232-0.251, maximum 0.540-0.551; no unit
at the clip bounds and none outside [0.1, 0.9]; treated mean 0.393-0.394 vs
control 0.389-0.390. Selected contrast shrinkage: 50 in all three primary
seeds (candidates 0/50/500; held-out energy risk).

## 4. Stability, inner-n, placebo

- Seed stability (primary): mean +0.163/+0.154/+0.167, tail +0.168/+0.159/
  +0.170, reference +0.056/+0.048/+0.055, sd 0.000/-0.001/-0.003, p10
  +0.091/+0.083/+0.099, p90 +0.182/+0.173/+0.182; seed SD <= 0.008
  everywhere, well inside the IF SEs.
- Placebo (3 permuted-treatment seeds): mean +0.003 (seed SD 0.031) vs primary
  +0.161; tail -0.001 vs +0.166; reference -0.003 vs +0.053; p90 -0.003 vs
  +0.179; p10 +0.012 vs +0.091. False-effect ratios (placebo mean / primary
  IF SE) are -0.47 to +0.86; the skew functional is the noisiest (+0.86) and
  the mean/tail/p90 ratios are 0.11/-0.02/-0.09. Placebo selected shrinkage
  500 in all three seeds (maximal regularization). Placebo bin contrasts are
  within +-0.13 and have no coherent moderator pattern.
- Inner-n >= 15 (math, 1 seed, frozen defaults): 1,030 units (221 small,
  809 control); mean +0.136 (SE 0.038), tail +0.160, reference +0.024,
  p10 +0.050, p90 +0.176, sd +0.024 (SE 0.014); shrinkage 500. The treated
  subsample is the larger small classes (15-17 tested), so the attenuation is
  a selected-sample result, not a clean noise correction.
- Inner-n >= 20 is infeasible: only 1 of 522 small-class units has 20+ tested
  students (small classes hold 13-17), below the adapter's five-unit arm
  minimum. Thresholds: >=15 keeps 221/809; >=17 keeps 55/791; >=18 keeps 6/771.
- 1 vs 2 only (aide classes dropped, 1 seed): mean +0.156 (SE 0.034),
  tail +0.161, p90 +0.188, reference +0.049; shrinkage 500.
- Benchmark sensitivity: all-student q_star leaves every non-reference
  functional identical to primary seed 0 (same fitted law) and changes the
  reference marginal from +0.056 to +0.047 (seed 0).

## 5. Trusted-scalar comparison

Naive differences in within-grade SD units: student-level regression with
grade fixed effects +0.201 (classroom-clustered SE 0.032); unit-level class
means with grade fixed effects +0.203 (SE 0.032). By grade: K +0.166 (0.066),
G1 +0.263 (0.066), G2 +0.189 (0.065), G3 +0.184 (0.061). The WCF DR mean
marginal is +0.161 (SE 0.028), and the particle plug-in is +0.150. All three
are inside or at the low end of the Krueger (1999) / Chetty et al. (2011)
range of about +0.15 to +0.20 SD for small classes; the DR estimate is
numerically close to the naive contrast and clearly distinct from the placebo.

## 6. Files written

Under `results/applied_study_exploration/project_star/`:

- `data/` primary math dataset (`dataset.npz`, `manifest.json`), plus
  `math_min15/`, `math_reg2/`, `math_qstarall/`, `reading/`,
  `math_schoolsat/` variants and `units_*.csv` unit tables.
- `primary/primary_seed{0,1,2}.json`, `placebo/placebo_seed{0,1,2}.json`,
  `sensitivity/{min15,reg2,reading,qstarall,schoolsat}_seed0.json`,
  `misc/smoke_seed99.json`.
- `summary.json`, `RESULTS.md`, `figures/bin_contrasts_math.png`.
- `misc/`: `inspection.json`, `design_checks.json`, `reference_noise.json`,
  `run_all.log`, `run_seed0_timing.log`.
- Scripts and pre-registration under `research/applied/project_star/`
  (`PREREGISTRATION.md`, `star_common.py`, `star_checks.py`,
  `01_inspect_data.py` ... `08_reference_noise_check.py`).

## 7. Caveats and next steps

- Within-school randomization: X contains only coarse school aggregates, so
  the logistic propensity is not the exact within-school assignment
  probability. The post-hoc school-saturated variant (79 school dummies,
  p=88, `sensitivity/schoolsat_seed0.json`, unregistered) gives mean +0.173
  vs +0.161 primary, so this concern does not move the headline estimate.
- Differential quantile noise: small classes have about two thirds of the
  tested students of control classes, so their unit quantiles are noisier.
  A split-half variance estimate (misc/reference_noise.json) implies a
  mechanical component of about +0.021 of the observed +0.056 reference
  contrast; the Q4 reference contrast (-0.103) survives this mechanism.
- Sensitivity variants run with one seed each (pre-registered two for
  min15/reg2/reading); flagged as a compute-driven deviation.
- The min15 sample is selected on tested count, which correlates with true
  class size; report as a robustness direction only.
- Possible spillovers across arms within schools and cross-grade mobility of
  students mean the estimates are classroom-assignment ITT-style contrasts.
