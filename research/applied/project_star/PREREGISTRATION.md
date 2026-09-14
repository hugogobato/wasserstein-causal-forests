# Project STAR applied study: pre-registration

Written before any WCF fit was run. Frozen: 2026-09-13.

This document fixes the unit definition, filters, outcome, treatment, covariates,
moderator, benchmark, and estimator settings for the Project STAR (Tennessee
class-size experiment) applied study of Wasserstein Causal Forests (WCF). All
choices below were made after inspecting the raw file schema (column names,
codes, missing codes, per-grade counts) but before fitting the estimator.

## 1. Sources and provenance

- Student file: `STAR_Students.tab`, Harvard Dataverse DOI
  <https://doi.org/10.7910/DVN/SIWH9F>, datafile id 666716, license CC0.
- School file: `STAR_K-3_Schools.tab`, same DOI, datafile id 666717, CC0.
- Local copies under `/tmp/opencode/wcf_scout/` were verified byte-identical
  (sha256) to fresh downloads of the two datafile ids on 2026-09-13:
  - students: `769be163ed54515858efa60b1a069c49ca0c475f0b0f9f5bdf90413be9d3ba97`
  - schools:  `776f2d3c4c09f047bfdfa5ce9406ed745ded425c4f5b4721f2c48aa81389cba9`
- File schema: 11,601 students x 379 columns; 80 schools x 53 columns. The
  student file is wide: one row per student with grade-specific blocks
  (`gk*` kindergarten, `g1*`-`g3*` grades 1-3) plus later-grade and high-school
  blocks that are not used here. Test scores are scaled scores (`g?tmathss`,
  `g?treadss`); missing is system-missing (empty), there are no negative
  missing codes in these columns.

## 2. Unit of analysis (pre-registered)

A unit is one classroom in one grade: the unique combination

    (grade in {K,1,2,3}) x g?schid (school) x g?tchid (teacher/class)

Verified checks before fitting: within each unit the class type `g?classtype`
is constant (0 violations in 1,340 units) and teacher attributes are constant
except one kindergarten unit with two teacher-race values (modal value used).
The teacher id is school-unique per the STAR codebook (8-digit id = 6-digit
school id + 2-digit teacher id). The unit is the classroom, matching the
class-size treatment.

Primary unit filter: at least 5 students with a non-missing math score.
All units have at least 7 tested students except 7 units with zero tested
students: all five grade-2 classes of school 168211 (2 small, 3 regular/aide)
and one regular grade-3 class each in schools 209510 and 231616. These are
dropped (2 small, 5 control), leaving 1,333 grade-classroom units
(522 small, 811 regular/aide).

## 3. Treatment (pre-registered)

`A = 1` if `g?classtype == 1` (small class, design 13-17 students),
`A = 0` if `g?classtype in {2, 3}` (regular class and regular class with a
full-time teacher aide, pooled). Robustness: `1 vs 2` only, dropping aide
classes. Every school-grade has both arms in the raw design (79, 76, 75, 75
school-grades for K, 1, 2, 3).

## 4. Outcome and benchmark (pre-registered)

Primary outcome (`math`): the 25 quantiles at `u_k = (k - 0.5)/25`, k = 1..25,
of the tested students' math scaled scores in the unit. Student scores are
standardized within grade (mean 0, sd 1 over all STAR students with a
non-missing score in that grade, both arms pooled), so all effects are in
within-grade student SD units and the four grade cohorts can be pooled.
Equal student weights within a unit; a unit's quantile vector is computed by
linear interpolation of the weighted empirical CDF (adapter
`weighted_quantiles`).

Robustness outcome (`reading`): same construction on `g?treadss`.

Missing scores are excluded from `Q` (they are not imputed). Student-level
missingness rates by arm and grade are reported; the design randomization
should make them similar across arms.

Benchmark `q_star` (primary): for each grade, the 25 quantiles of all tested
students in control classes (`classtype in {2,3}`) on the within-grade z-score
scale; the four grade-specific vectors are averaged with equal weight.
Sensitivity: same with all students (both arms) per grade.

## 5. Covariates X and moderator (pre-registered)

9 numeric columns; column 0 is the moderator because the WCF bin edges are
built from it (frozen estimator interface).

0. Moderator: school free/reduced-lunch share, `ecdf_moderator` transformed to
   (-1, 1), so the estimator's edges (-0.5, 0, 0.5) are the school-level
   free-lunch quartiles. Raw share = mean over grades of the official school
   file percentages `GKFRLNCH, G1FRLNCH, G2FRLNCH, G3FRLNCH` (columns var9,
   var21, var33, var45 of the school file, verified against student-record
   aggregates), divided by 100; fallback to the student-computed school-grade
   share when an official value is missing (none missing at school level).
1. Teacher female (`g?tgen == 2`).
2. Teacher non-white (`g?trace != 1`; 1=White, 2=Black, 3=Asian).
3. Teacher holds a graduate degree (`g?thighdegree >= 3`).
4. Teacher total years of experience (`g?tyears`), grade-median imputation for
   the 9 units with a fully missing value.
5. School urbanicity from the school file (1 inner city, 2 suburban,
   3 rural, 4 urban); verified identical to the student-file `g?surban`.
6. Class share female (`gender == 2`, 1=male, 2=female).
7. Class share Black (`race == 2`, 1=White, 2=Black, other categories pooled
   into the reference).
8. Class share free/reduced lunch (`g?freelunch == 1`).

Class size (`g?classsize`) is NOT used as a covariate: it is determined by the
treatment. Class composition shares are assignment-determined and are included
as precision covariates; with randomization they are balanced in expectation.

## 6. Estimator settings (pre-registered, frozen defaults)

`run_wcf` defaults: M=10 particles, 100 trees, learning rate 0.12, depth 4,
min leaf 10, min arm leaf 5, collision epsilon 1e-3, contrast shrinkage
candidates (0, 50, 500), 3 outcome folds, cross-fitted logistic propensity
(5 folds, clipped to [0.02, 0.98]), default architecture v1 / partial sharing.

Declared functionals: mean, sd, skew, upper-half mean ("tail"), reference
distance to `q_star`; plus p10 (`q[:,2]`, the u=0.10 coordinate) and p90
(`q[:,22]`, the u=0.90 coordinate) as extra functionals.

Reported: DR marginal (TATE) with influence-function SE, 4-bin DR contrasts
(TCATE) with per-bin SEs recomputed from AIPW scores, plug-in particle
contrasts, propensity/overlap summary, selected contrast shrinkage.

## 7. Runs (pre-registered)

- Primary: math, control `q_star`, seeds (0, 1, 2).
- Placebo: permuted treatment, seeds (0, 1, 2).
- Inner-n sensitivity: units with at least 15 tested students (seeds 0, 1).
  The at-least-20 threshold is infeasible on the treated side: only 1 of 524
  small-class units has 20+ tested students (small classes hold 13-17), and the
  adapter requires at least 5 units per arm; this is reported, not run.
- Treatment-pooling robustness: `1 vs 2` only (seeds 0, 1).
- Outcome robustness: reading (seeds 0, 1).
- Benchmark robustness: all-student `q_star` (seed 0).

## 8. Trusted-scalar checks (pre-registered)

- Naive unit-level difference in means of the class mean score, by grade and
  pooled, plus a student-level OLS of the z-score on `A` with grade fixed
  effects and classroom-clustered SEs.
- Compare the WCF DR mean marginal with the literature range of about
  +0.15 to +0.20 within-grade SD for small classes (Krueger 1999;
  Chetty et al. 2011).
- Placebo estimates should be near zero relative to the primary estimates.
