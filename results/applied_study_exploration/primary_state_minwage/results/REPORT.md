# Primary applied study: WCF results

Frozen defaults: M=10 particles, 100 trees, lr 0.12, depth 4, min leaf 10, min arm leaf 5, epsilon 1e-3, contrast candidates (0, 50, 500), 3 folds. All fits below use these settings.

### Confirmatory window 2000-2016, treatment A1, three seeds (0, 1, 2)

| functional | DR marginal (mean over seeds) | across-seed sd | IF SE (seed 0) | plug-in (seed 0) | DR bins Q1..Q4 (mean) |
|---|---|---|---|---|---|
| mean | +0.3344 | 0.0184 | 0.1315 | +0.0210 | +0.822 -0.092 +0.396 +0.209 |
| sd | +0.1285 | 0.0247 | 0.1184 | -0.0056 | +0.550 -0.263 +0.367 -0.141 |
| skew | -0.0022 | 0.0023 | 0.0176 | +0.0000 | -0.015 +0.015 +0.011 -0.020 |
| tail | +0.4431 | 0.0327 | 0.2134 | +0.0182 | +1.248 -0.276 +0.630 +0.166 |
| reference | +0.1068 | 0.0229 | 0.1579 | -0.0005 | +0.179 -0.228 +0.435 +0.041 |
| p10 | +0.2211 | 0.0102 | 0.0393 | +0.0282 | +0.279 +0.175 +0.241 +0.189 |
| p90 | +0.4222 | 0.0434 | 0.3129 | +0.0119 | +1.181 -0.355 +0.943 -0.084 |
| ref_us2016 | -0.2107 | 0.0258 | 0.1702 | -0.0074 | -0.803 +0.285 -0.510 +0.188 |
| ref_mean_norm | -0.0015 | 0.0010 | 0.0033 | -0.0007 | +0.004 -0.009 +0.008 -0.009 |

### Placebo (permuted A1), three seeds (0, 1, 2)

| functional | DR marginal (mean over seeds) | across-seed sd | IF SE (seed 0) | plug-in (seed 0) | DR bins Q1..Q4 (mean) |
|---|---|---|---|---|---|
| mean | -0.0235 | 0.1079 | 0.1374 | +0.0092 | +0.017 -0.093 -0.146 +0.129 |
| sd | -0.0377 | 0.1739 | 0.1257 | -0.0449 | -0.152 +0.018 -0.130 +0.114 |
| skew | -0.0071 | 0.0248 | 0.0172 | -0.0114 | -0.032 +0.011 -0.009 +0.002 |
| tail | -0.0366 | 0.1963 | 0.2246 | -0.0038 | -0.051 -0.100 -0.227 +0.232 |
| reference | -0.0598 | 0.1858 | 0.1552 | -0.0599 | -0.102 -0.082 -0.179 +0.125 |
| p10 | +0.0048 | 0.0372 | 0.0434 | +0.0410 | +0.061 -0.019 -0.034 +0.012 |
| p90 | -0.1386 | 0.3442 | 0.3367 | -0.0630 | -0.170 -0.147 -0.382 +0.144 |
| ref_us2016 | -0.0261 | 0.0949 | 0.1310 | -0.0120 | +0.181 -0.118 +0.037 -0.205 |
| ref_mean_norm | -0.0013 | 0.0051 | 0.0033 | -0.0026 | -0.010 +0.004 -0.002 +0.002 |

### Robustness fits (seed 0 unless noted)

| run | n | treated | shrinkage | mean | p10 | sd | skew | reference |
|---|---|---|---|---|---|---|---|---|
| A1 primary seed 0 | 867 | 117 | 500 | +0.316 | +0.224 | +0.126 | -0.001 | +0.117 |
| A2 (federal increases included) | 867 | 176 | 50 | +0.243 | +0.185 | +0.106 | +0.006 | +0.124 |
| A3 (state above federal floor) | 867 | 336 | 0 | +0.130 | +0.168 | -0.081 | -0.004 | -0.084 |
| A1 + calendar year in X (post-hoc) | 867 | 117 | 50 | +0.336 | +0.233 | +0.099 | -0.005 | +0.132 |
| First-difference A1 seed 0 (post-hoc) | 867 | 117 | 500 | +0.226 | +0.197 | +0.026 | -0.079 | +0.106 |
| First-difference A1 seed 1 (post-hoc) | 867 | 117 | 50 | +0.227 | +0.194 | +0.042 | +0.252 | +0.111 |
| Extension 1980-2016 seed 0 | 1887 | 163 | 50 | +0.440 | +0.262 | +0.262 | +0.029 | +0.177 |
| Extension 1980-2016 seed 1 | 1887 | 163 | 50 | +0.398 | +0.251 | +0.222 | +0.023 | +0.125 |
| Extension 2017-2022 seed 0 | 306 | 66 | 0 | +0.720 | +0.344 | +0.560 | -0.021 | +0.888 |
| Leave out CA | 850 | 112 | 500 | +0.268 | +0.210 | +0.072 | +0.008 | +0.010 |
| Leave out TX | 850 | 117 | 50 | +0.313 | +0.197 | +0.146 | +0.006 | +0.100 |
| Leave out NY | 850 | 113 | 50 | +0.236 | +0.205 | +0.073 | +0.008 | +0.006 |

### First-difference variant detail (seeds 0 and 1)

| functional | seed 0 | seed 1 |
|---|---|---|
| mean | +0.226 (se 0.0980) | +0.227 (se 0.1047) |
| sd | +0.026 (se 0.0647) | +0.042 (se 0.0676) |
| skew | -0.079 (se 0.3508) | +0.252 (se 0.3681) |
| tail | +0.260 (se 0.1350) | +0.267 (se 0.1452) |
| reference | +0.106 (se 0.0872) | +0.111 (se 0.0959) |
| p10 | +0.197 (se 0.0586) | +0.194 (se 0.0618) |
| p90 | +0.250 (se 0.1777) | +0.260 (se 0.1961) |
| ref_ramp | +0.006 (se 0.0695) | +0.016 (se 0.0778) |

### Propensity diagnostics (seed 0, primary)

| quantity | value |
|---|---|
| min | 0.0200 |
| max | 0.4083 |
| mean | 0.1354 |
| share_at_clip_low | 0.0104 |
| share_at_clip_high | 0.0000 |
| share_outside_0p1_0p9 | 0.4302 |
| treated_mean | 0.1815 |
| control_mean | 0.1282 |

### Selected contrast shrinkage and held-out risks (primary seeds)

| seed | selected | risks (shrinkage: risk) |
|---|---|---|
| 0 | 500 | 0: 1.04487, 50: 1.03460, 500: 1.03423 |
| 1 | 500 | 0: 0.96860, 50: 0.96140, 500: 0.95980 |
| 2 | 500 | 0: 1.04187, 50: 1.03789, 500: 1.03687 |

## Post hoc overlap trimming (added 2026-09-14, not pre-registered)

The paper reports fits after dropping units whose full-sample fitted propensity lies outside
[0.05,0.95] or [0.10,0.90], with the model refit on the retained sample. This is a robustness
exercise prompted by the DR/plug-in divergence, not part of the pre-registered analysis.
Reported variant: [0.05,0.95], seed-0 retained sample 760 (112/648). The restriction shrinks the
sd DR/plug-in gap from 0.134 to 0.051 and raises the plug-in mean, but the DR mean stays positive
and the pre-registered sign pattern is still not recovered; p10 stays positive.
Fits: results/fit_trim005_095_seed*.json. (The stricter [0.10,0.90] fits were computed but are no
longer reported; see EVIDENCE_LEDGER for their values.)
