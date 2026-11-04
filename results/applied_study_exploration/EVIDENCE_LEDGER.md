# Applied-study evidence ledger (audited, 2026-09-13)

Internal audit artifact for the WCF preprint. Every row was verified by reading the
result files under `results/applied_study_exploration/`. Claims marked CONFLICT or OPEN
must not appear in the manuscript as stated. Status codes: V = verified, C = conflict,
O = open, I = inference.

## Three-study overview (as verified)

| Study | Unit | n (treated/control) | Outcome distribution | Treatment | Benchmark | Overlap |
|---|---|---|---|---|---|---|
| Project STAR (main) | classroom-grade | 1,333 (522/811) | within-grade z math scaled scores, K=25 | small class vs regular/aide | pooled control-class distribution | prop 0.23-0.55, 0% outside [0.05,0.95] |
| State minimum wage | state-year | 867 (117/750) | hourly wage distribution, K=25, 2016 USD | real state MW rise > $0.25, own-rate, non-federal | Nordic (Eurostat) scaled; US 2016 | prop 0.02-0.41, 12.3% below 0.05, 0% above 0.95 |
| Kenya cash transfers | village | 653 (328/325) | per-capita nondurable consumption, K=25 | village randomization | pooled control-village distribution | prop 0.24-0.76, 0% outside [0.05,0.95] |

## Project STAR (main text candidate; all values V)

Source: Harvard Dataverse doi:10.7910/DVN/SIWH9F, CC0; STAR_Students.tab (11,601 x 379).
Unit: (grade in {K,1,2,3}, school id, teacher id); 1,333 classroom-grades (522 small; 811 control
= 400 regular + 411 aide); 24,610 tested students; tested per unit: small mean 14.09 (range 7-20),
control mean 21.28 (13-29). Q: 25 midpoint quantiles of within-grade z-scored math scaled scores,
equal student weights. Moderator: school free-lunch share, quartiles (333/333/334/333).
X (9): moderator, teacher female, non-white, graduate degree, years, school urbanicity, class shares
female/Black/free-lunch. Class size excluded. q_star: grade-average of pooled control-student quantiles.
Estimator: frozen defaults (M=10, 100 trees, lr 0.12, depth 4, min leaf 10/5, eps 1e-3,
candidates (0,50,500), 3 folds), seeds 0-2.

DR marginals (seed mean; IF SE; seed SD):
mean +0.1610 (0.0277; 0.0067); sd -0.00115 (0.01095; 0.00134); skew +0.01925 (0.03082; 0.00793);
upper-half mean +0.16562 (0.03120; 0.00628); p10 +0.09085 (0.02936; 0.00775);
p90 +0.17893 (0.03583; 0.00506); reference +0.05278 (0.01749; 0.00423).
Plug-in: mean +0.1497; tail +0.1484; p10 +0.098; p90 +0.1649; reference +0.0450; sd -0.0107; skew +0.0237.
4-bin DR contrasts (seed mean; seed-0 SE):
mean +0.164(0.056)/+0.092(0.047)/+0.126(0.057)/+0.262(0.062);
tail +0.161/+0.112/+0.108/+0.282; reference +0.176/+0.060/+0.078/-0.103;
p10 +0.119/-0.050/+0.116/+0.179; p90 +0.173/+0.115/+0.117/+0.311.
Propensity: min 0.232, max 0.551, treated 0.394 vs control 0.390, none clipped/outside.
Selected shrinkage 50 all seeds. Placebo (permuted, 3 seeds): mean +0.00305 (seed SD 0.0313);
tail -0.0006; reference -0.0025; p10 +0.0116; p90 -0.0034.
Naive: student grade-FE +0.2015 (classroom-clustered SE 0.0324); unit-mean +0.2026 (0.0325);
grade K +0.166, G1 +0.263, G2 +0.189, G3 +0.184; reading +0.2058.
Sensitivities: min15 n=1,030 mean +0.1356 (0.0380); regular-only n=922 +0.1561 (0.0337);
reading n=1,322 +0.1508 (0.0256); all-student benchmark reference +0.0473 (0.0173);
school-saturated (post hoc) +0.1730 (0.0312).
Caveats (must appear): reference benchmark is the control distribution (self-referential), and a
split-half diagnostic attributes 0.0209 of the observed 0.0557 reference contrast to finite-class
sampling noise (37.5%); placebo runs selected shrinkage 500 while primary selected 50 (not
like-for-like); inner classes have 7-20 tested students; composition covariates are
assignment-determined; sensitivity variants used one seed, not two as pre-registered.

## State minimum wage (appendix; V unless flagged)

Source: NBER CPS MORG 1979-2024 (data.nber.org/morg/annual/morgYY.dta); Vaghul-Zipperer v1.4.0;
CPI-U-RS (2016 base). n=867 (51 x 2000-2016), A1=117 treated; extensions 1980-2016 n=1,887 (163 T),
2017-2022 n=306 (66 T). Q: 25 quantiles of hourly wages, 2016 USD, MORG earnwt; age 16-64,
not self-employed. A1: real effective MW rose > $0.25, own nominal rate rose, not federal-only.
q_star: Eurostat Nordic vector scaled by 6.6553e-4 (US 2016 median 17.6922 / Nordic median 26583.71).

DR marginals (seed mean; IF SE seed 0; seed SD):
mean +0.3344 (0.1315; 0.0184); sd +0.1285 (0.1184; 0.0247); skew -0.0022 (0.0176; 0.0023);
tail +0.4431 (0.2134; 0.0327); reference +0.1068 (0.1579; 0.0229); p10 +0.2211 (0.0393; 0.0102);
p90 +0.4222 (0.3129; 0.0434); US-2016 reference -0.2107 (0.1702; 0.0258);
mean-normalized reference -0.001 (0.0033; 0.0010).
Plug-in: mean +0.021; sd -0.006; tail +0.018; reference -0.0005; p10 +0.028; p90 +0.012.
Overlap: prop 0.020-0.408; treated 0.182 vs control 0.128; 12.3% below 0.05, 0% above 0.95; 1.0% at the 0.02 low clip.
Shrinkage 500 all primary seeds. Placebo: mean -0.023 (SD 0.108), p10 +0.005 (0.037).
Robustness: A2 mean +0.243/p10 +0.185; A3 mean +0.130/p10 +0.168; first-difference mean +0.226
(0.098)/p10 +0.197 (0.059); 1980-2016 mean +0.44/+0.40, p10 +0.263/+0.251; leave-one-state-out
p10 +0.210 (CA), +0.197 (TX), +0.205 (NY); Nordic reference collapses to +0.010 (CA) and +0.006 (NY).
C (do not use): the study report's claimed Cengiz-panel validation and TWFE/ATT-IPW trusted
scalars have no artifact in the repository; they must not appear in the paper. The pre-registered
sign check failed under the DR layer for mean (+0.334) and sd (+0.129) while the plug-in
satisfies it; the DR/plug-in gap where propensities are small is a disclosed limitation.
Overlap standard: the paper now reports propensities outside [0.05,0.95]. Full-sample shares
outside that band: seed0 12.34% (1 - 0.87659), seed1 12.57%, seed2 12.46%; all of it below 0.05,
none above 0.95; 1.0% at the 0.02 lower clip.
O: no state-clustered inference; "year-FE" variant was a linear year covariate.

Post hoc overlap restriction (fits in results/): units are dropped when their full-sample fitted
propensity falls below 0.05 (none above 0.95), then the model is refit on the retained sample (so
reported propensities are not constrained to the interval).
Trim [0.05,0.95]: seed0 n=760 (112/648) [seed1 758 (112/646); seed2 759 (113/646)];
mean +0.2442 (IF SE 0.1321; seed SD 0.0775); sd +0.0412 (0.1224; 0.0464); p10 +0.2172 (0.0451; 0.0299);
reference +0.0922 (0.1588); plug-in mean +0.0878, sd -0.0099, p10 +0.1003, reference -0.0016.
Reads: the restriction shrinks the sd DR/plug-in gap from 0.134 to 0.051 and raises the plug-in
mean toward the DR mean, but the mean gap stays at 0.156 and the DR mean stays positive, so the
pre-registered pattern is still not recovered; p10 (+0.217 DR, +0.100 plug-in) and the calibrated
reference (+0.092, SE 0.159) are the stable quantities.
The [0.10,0.90] restriction was computed (fits fit_trim010_090_seed*.json) but is no longer
reported: the standard band is [0.05,0.95]. Its values, for the record: n=494 (94/400);
mean +0.2849 (SE 0.1550); sd +0.0888 (0.1369); p10 +0.1905 (0.0615); plug-in mean +0.0131.

## Kenya cash transfers (appendix; V unless flagged)

Source: Zenodo doi:10.5281/zenodo.16548593, CC BY 4.0. n=653 villages (328/325); 8,226 usable
households; households per village min 8, median 12.0, mean 12.597, max 25 (C: the study report
said "median 12.6"; it is the mean). Q: 25 quantiles of per-capita nondurable consumption,
household-weighted (hhweight_EL). Moderator: baseline assets, quartiles 163/163/164/163.
q_star: pooled control-village distribution (p10 11,101; median 26,207; p90 69,933 KSH pc);
capped variant at control p75 = 43,827.

DR marginals (seed mean; IF SE; seed SD):
mean +958.5 (1308.6; 70.6); sd -779.5 (2660.0; 110.3); skew +0.0601 (0.0787; 0.0016);
tail +1060.4 (2353.7; 144.2); reference -3057.5 (2573.6; 121.5); p10 +1125.5 (418.7; 13.0);
p90 +72.5 (5121.7; 340.6). Raw contrasts and permutation p: mean +543.5 (p=0.675);
p10 +913.9 (p=0.021); reference -2572.1 (p=0.374); p90 -632.5 (p=0.912).
Overlap: 0.243-0.758, 0% outside. Shrinkage 0 all primary seeds. Placebo mean +199 (SD 854);
reference -292 (SD 3243). Q measurement-noise bootstrap: mean +975 (SD 2090); reference -2655 (SD 2700).
Trusted scalar: eligible-household nondurables +1157 (SE 1433), total consumption +3221 (SE 1549);
all-household nondurables -118 (SE 1997). Saturation splits: low-sat mean +980, ref -6620;
high-sat mean +1890, ref +1641.
Caveats: only p10 is nominally significant among seven functionals under the control benchmark;
reference not significant (raw p=0.374); measurement error dominates fit-seed spread by 22-33x;
saturation reverses signs.

External normative benchmark (added 2026-09-14). Source: World Bank PIP API, Kenya 2015 national
consumption (KIHBS 2015/16; survey_year 2015.67), endpoint
https://api.worldbank.org/pip/v1/pip?country=KEN&year=2015; rural/urban cuts unpublished for Kenya.
PIP fields: mean 5.1342, median 3.7466, gini 0.4077 PPP$/day; deciles 0.0241, 0.0374, 0.0467,
0.0558, 0.0665, 0.0795, 0.0952, 0.1195, 0.1596, 0.3156. Reconstruction: monotone cubic Lorenz
curve through decile points with the derivative at u=0.5 fixed to reproduce the published median,
evaluated at the 25 midpoint levels; converted to nominal KSH with the study's constant factor
46.49 KSH per 2017-PPP dollar (ratio nondurables_exp_pc / nondurables_exp_pc_PPP, constant across
all 8,223 households, max deviation 2.7e-6); annualisation x365. Validation: mean integral 5.133687
(PIP 5.1342), median 3.7466 exact, gini 0.403779 (PIP 0.4077).
Benchmark medians (KSH pc per year): sample control 24,494; treated 26,256; national 63,575.
Raw distances and permutation p: control benchmark d0 22,241, d1 19,669, contrast -2,572 (p=0.374);
national benchmark d0 79,677, d1 74,990, contrast -4,687 (p=0.0075, 10,000 permutations).
Calibrated refit with the national benchmark (3 seeds, runs/external/): reference DR -5,535.6
(across-seed mean IF SE 2,235.0; seed SD 132.3; seed-0 plug-in -1.2). The other functionals in this
fit are unchanged from the primary per-seed fits (mean +888.9/+1030.0/+956.5 in both), so only the
reference functional responds to the benchmark; the table reports just that row. Shrinkage 0 all seeds.
Conclusion: with an external normative benchmark the reference contrast is negative in both layers
and nominally significant (calibrated t ~ -2.5; raw p=0.0075); this is the only reference-distance
result in the paper that combines an external benchmark with nominal significance.

