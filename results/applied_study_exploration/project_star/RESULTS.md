# Project STAR WCF applied study: results summary

Primary runs: 3 seeds; placebo runs: 3 seeds.

## Primary (math, control q_star, frozen defaults)

| functional | DR marginal | seed SD | IF SE | plug-in | bins b1..b4 (DR) | bin IF SE |
|---|---|---|---|---|---|---|
| mean | 0.161 | 0.007 | 0.028 | 0.150 | 0.164, 0.092, 0.126, 0.262 | 0.056, 0.047, 0.057, 0.062 |
| sd | -0.001 | 0.001 | 0.011 | -0.011 | -0.010, 0.019, -0.027, 0.014 | 0.022, 0.022, 0.020, 0.024 |
| skew | 0.019 | 0.008 | 0.031 | 0.024 | 0.038, -0.081, 0.130, -0.011 | 0.063, 0.057, 0.062, 0.063 |
| tail | 0.166 | 0.006 | 0.031 | 0.148 | 0.161, 0.112, 0.108, 0.282 | 0.064, 0.054, 0.063, 0.070 |
| reference | 0.053 | 0.004 | 0.017 | 0.045 | 0.176, 0.060, 0.078, -0.103 | 0.038, 0.027, 0.036, 0.036 |
| p10 | 0.091 | 0.008 | 0.029 | 0.098 | 0.119, -0.050, 0.116, 0.179 | 0.055, 0.055, 0.062, 0.064 |
| p90 | 0.179 | 0.005 | 0.036 | 0.165 | 0.173, 0.115, 0.117, 0.311 | 0.075, 0.065, 0.069, 0.078 |

Selected contrast shrinkage per primary run: 50.0, 50.0, 50.0

Propensity across primary seeds: min=0.232, max=0.551, treated mean=0.394, control mean=0.390, max share outside [0.1,0.9]=0.000, max share clipped=0.000.

## Placebo (permuted treatment)

| functional | DR marginal | seed SD | IF SE | plug-in | bins b1..b4 (DR) | bin IF SE |
|---|---|---|---|---|---|---|
| mean | 0.003 | 0.031 | 0.028 | 0.002 | -0.005, -0.008, 0.040, -0.016 | 0.052, 0.047, 0.055, 0.066 |
| sd | -0.005 | 0.009 | 0.010 | -0.004 | 0.003, -0.014, -0.015, 0.006 | 0.021, 0.020, 0.021, 0.022 |
| skew | 0.026 | 0.025 | 0.031 | 0.022 | -0.011, 0.030, 0.013, 0.074 | 0.058, 0.055, 0.063, 0.064 |
| tail | -0.001 | 0.031 | 0.031 | -0.001 | -0.001, -0.017, 0.028, -0.012 | 0.060, 0.053, 0.061, 0.073 |
| reference | -0.003 | 0.011 | 0.017 | -0.003 | 0.006, -0.002, -0.007, -0.007 | 0.034, 0.027, 0.034, 0.039 |
| p10 | 0.012 | 0.026 | 0.029 | 0.010 | -0.018, 0.025, 0.055, -0.016 | 0.054, 0.054, 0.059, 0.062 |
| p90 | -0.003 | 0.031 | 0.036 | -0.003 | -0.005, -0.021, 0.016, -0.003 | 0.071, 0.064, 0.069, 0.081 |

False-effect ratio (placebo mean / primary IF SE): mean=0.11, sd=-0.47, skew=0.86, tail=-0.02, reference=-0.15, p10=0.40, p90=-0.09

## Trusted scalar (within-grade SD units)

- student-level difference in means with grade FE: 0.201 (cluster SE 0.032)
- unit-level mean with grade FE: 0.203 (SE 0.032)

By grade (student-level, cluster SE): gk 0.166 (0.066); g1 0.263 (0.066); g2 0.189 (0.065); g3 0.184 (0.061)

## Sensitivity variants

| variant | seeds | mean | tail | reference | p10 | p90 | sd | shrink |
|---|---|---|---|---|---|---|---|---|
| min15 | 1 | 0.136 | 0.160 | 0.024 | 0.049 | 0.176 | 0.0237 | 500.0 |
| reg2 | 1 | 0.156 | 0.161 | 0.049 | 0.090 | 0.188 | 0.0012 | 500.0 |
| reading | 1 | 0.151 | 0.154 | 0.062 | 0.076 | 0.150 | -0.0024 | 50.0 |
| qstarall | 1 | 0.163 | 0.168 | 0.047 | 0.091 | 0.182 | 0.0000 | 50.0 |
| schoolsat | 1 | 0.173 | 0.182 | 0.066 | 0.093 | 0.201 | 0.0062 | 50.0 |

## Inner-n threshold feasibility

- tested_n >= 13: 429 small / 811 control units; adapter minimum satisfied: True
- tested_n >= 15: 221 small / 809 control units; adapter minimum satisfied: True
- tested_n >= 17: 55 small / 791 control units; adapter minimum satisfied: True
- tested_n >= 18: 6 small / 771 control units; adapter minimum satisfied: True
- tested_n >= 20: 1 small / 646 control units; adapter minimum satisfied: False

Covariate balance (standardized differences), small minus control:
classsize=-3.891, assigned_n=-3.891, teacher_tyears=-0.012, share_female=0.063, share_black=-0.074, share_freelunch=-0.115, school_fl_official=-0.043, school_urban=0.060

Figure: `results/applied_study_exploration/project_star/figures/bin_contrasts_math.png`
