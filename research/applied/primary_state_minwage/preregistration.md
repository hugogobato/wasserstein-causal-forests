# Primary applied study: US state-year wage distributions and minimum-wage policy

Pre-registration frozen before any WCF fit. File paths are relative to the repository
root. This document is also copied into the saved-data manifest (`meta` field).

## Study object

Units are state-years (state s, calendar year t). The outcome for a unit is the K = 25
midpoint-grid quantile vector of the state-year hourly wage distribution of employed
wage earners, in 2016 dollars. Treatment is a large state minimum-wage increase during
year t. The benchmark `q_star` is a Nordic distribution used as a shape reference.

## Sources

1. CPS MORG annual microdata, NBER, https://data.nber.org/morg/annual/morgYY.dta for
   YY = 79..99, 00..24 (credential-free). Primary window uses 1979-2022; the 2023-2024
   files are cached but not used because the Vaghul-Zipperer policy panel ends in 2022.
2. Vaghul-Zipperer historical minimum wage v1.4.0 (annual, 1974-2022),
   https://github.com/benzipperer/historicalminwage/releases/download/v1.4.0/mw_state_stata.zip
   (file `mw_state_annual.dta`).
3. CPI-U-RS quarterly, base 2016 = 100, extracted from the Cengiz-Dube-Lindner-Zipperer
   (2019, QJE) replication package (Harvard Dataverse doi:10.7910/DVN/TJCTC7, CC0),
   file `data/state_panels_with3quant1979.dta` column `cpi`; stored as
   `research/applied/primary_state_minwage/cpi_urs_quarterly_2016base.csv` (152 quarters).
   For 2017-2022 the same index is spliced forward with FRED CPIAUCSL growth rates
   (https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL): index(y) =
   100 * CPI-U_annual(y) / CPI-U_annual(2016), anchored at the 2016 CPI-U-RS value.
4. Nordic benchmark `results/applied_study_exploration/nordic_benchmark/nordic_qstar_k25_grid.csv`,
   column `B_incY2024_4c_popw` (Eurostat ilc_di01 TC quantile thresholds in PPS, survey
   year 2025 / income year 2024, DK+FI+NO+SE population-weighted). Benchmark units are
   annual PPS; see rescaling below.

## Sample (frozen)

Individual inclusion criteria, applied identically to every annual MORG file:

- age 16-64;
- `earnwt > 0`;
- not self-employed: drop `class` in {5, 6} for year <= 1993, `class94` in {6, 7} for
  year >= 1994;
- `paidhre` in {1, 2};
- not imputed, following Cengiz et al. (2019) and Hirsch-Schumacher: for 1994 through
  August 1995 the I25 flags are treated as uninformative (imputation set to 0);
  for 1989-1993 the flags are reconstructed by comparing edited and unedited fields
  (`earnhr`, `uhours`, `uearnwk`); `imputed = 1` if paid-hourly and the hourly-wage flag
  I25c is positive, or if not paid-hourly and the hours flag I25a or the earnings flag
  I25d is positive; imputed records are dropped;
- hourly wage `wage = earnhre / 100` for `paidhre == 1`, and
  `wage = earnwke / uhourse` for `paidhre == 2` (requires `earnwke > 0` and
  `uhourse > 0`); records with non-finite or non-positive wage are dropped.

Nominal wages are deflated to 2016 dollars with the CPI-U-RS index above. The
construction matches `state_panels_cents_new_QJE.do` in the Cengiz et al. replication
package, minus the $30 bin top-code and with the age 16-64 restriction added.

State identifiers: `stfips` when present (1989+), otherwise the MORG CPS state code in
`state` mapped by the crosswalk in `build_dataset.py`, derived empirically from the
1989-1990 files that carry both variables (51 codes, bijective).

## Outcome Q

For every state-year with at least 20 sampled workers, weighted quantiles of the hourly
wage at the midpoint grid u_k = (k - 0.5)/25, k = 1..25, using `earnwt` as weights
(`weighted_quantiles` in the shared adapter). No top-code correction: at most ~0.1% of
records sit at the $99.99 earnhre cap.

## Treatment (primary pre-registered)

Policy source: `mw_state_annual.dta` (state-year, 1974-2022), `min_mw` (state effective
rate; equals the federal rate for states without their own higher rate) and
`min_fed_mw`. Real effective minimum wage `rmw_t = min_mw_t / (cpi_t/100)`.

- `A1` (PRIMARY): 1 if `rmw_t - rmw_{t-1} > 0.25` **and** `min_mw_t > min_mw_{t-1}`
  **and** not a federal-only increase. Federal-only for state s in year t means the
  federal rate rose (`min_fed_mw_t > min_fed_mw_{t-1}`), the state was at the federal
  floor before (`min_mw_{t-1} == min_fed_mw_{t-1}`), and the state is at the new federal
  floor after (`min_mw_t == min_fed_mw_t`); this mirrors the `fedincrease != 1` clause
  in Cengiz et al. (2019).
- `A2` (robustness): 1 if `rmw_t - rmw_{t-1} > 0.25` (federal-induced increases count).
- `A3` (robustness): 1 if `min_mw_t > min_fed_mw_t` (state above the federal floor).

## Covariates X and moderator (frozen)

All covariates are strictly lagged to t-1. Column 0 of X is the ecdf transform of the
raw moderator. Remaining columns are z-scored (documented deviation: tree splits are
invariant, the propensity logistic is not).

- Column 0 (moderator): lagged state median real hourly wage (own Q, u = 0.10 is
  q[:,2] and u = 0.50 is q[:,12]), transformed by `ecdf_moderator`. The forest bins
  column 0 at (-0.5, 0, 0.5), i.e. into pooled quartiles.
- Lagged log real effective minimum wage, `log(max(min_mw, min_fed_mw) / (cpi/100))`.
- Lagged state median real wage (same variable as the moderator, in levels).
- Lagged state p10 real wage (own Q, u = 0.10).
- Lagged below-share: weighted share of sampled workers with nominal wage below the
  state's effective minimum wage at t-1 (same real dollars).
- Lagged log CPS earner population: log of the weighted sum of `earnwt` in the state
  (the Cengiz panel's `population` is the same construct).
- Lagged share with high-school education or less (`hgradecp <= 12`, Cengiz mapping).
- Lagged female share (`sex == 2`).

## Benchmarks q_star (frozen)

- PRIMARY `q_star`: the Nordic vector rescaled by one documented scalar,
  `s = median(pooled US wage distribution in 2016) / Nordic(u = 0.5)`. The scalar maps
  the Nordic median to the pooled 2016 US median in 2016 dollars; the benchmark is
  treated as a shape reference.
- `reference`: adapter-declared distance to `q_star`.
- Extra functional `ref_mean_norm`: sqrt(sum_k w_k (q_k/mean(q) - q*_k/mean(q*))^2),
  which removes the level.
- Extra functional `ref_us2016`: same Euclidean grid distance to the employment-weighted
  national 2016 distribution built from the same MORG sample.

## Estimands

Run `run_wcf` once per seed with the frozen defaults (M = 10 particles, 100 trees,
learning rate 0.12, depth 4, min leaf 10, min arm leaf 5, epsilon 1e-3, contrast
candidates (0, 50, 500), 3 folds, arm shrinkage 5.0). Reported automatically: DR
marginals with influence-function SEs, four-bin DR contrasts, plug-in counterparts,
propensity/overlap diagnostics, and the selected contrast shrinkage. Declared
functionals: the five automatic ones (mean, sd, skew, upper-half mean `tail`, reference)
plus `p10` (q[:,2]), `p90` (q[:,22]), `ref_mean_norm`, `ref_us2016`.

## Scale, stability, placebo

- Confirmatory run: 2000-2016 (17 years x 51 states = 867 state-years).
- Extensions if time permits: 1980-2016 (Cengiz window) and 2017-2022.
- Stability: seeds 0, 1, 2 at the frozen defaults.
- Placebo: permutation of A, seeds 0, 1, 2 at the frozen defaults.
- Dependence caveat: state-year rows repeat states and the adapter's folds are random,
  not state-grouped, so IF SEs are optimistic; leave-one-state-out and a within-state
  first-difference variant are attempted if time permits.

## Trusted scalar (falsification check)

The minimum-wage literature (Cengiz et al. 2019; Lee 1999) implies a rise in the lower
tail and compression: expect DR marginal contrasts h_p10 > 0 and h_mean <= 0 (or small),
h_sd < 0, h_skew < 0. Any sign conflict is treated as a pipeline bug until checked
(time alignment, deflator, weights, event coding).

## Post hoc overlap trimming (added 2026-09-14, not pre-registered)

The paper reports fits after dropping units whose full-sample fitted propensity lies outside
[0.05,0.95] or [0.10,0.90], with the model refit on the retained sample. This is a robustness
exercise prompted by the DR/plug-in divergence, not part of the pre-registered analysis.
Reported variant: [0.05,0.95], seed-0 retained sample 760 (112/648). The restriction shrinks the
sd DR/plug-in gap from 0.134 to 0.051 and raises the plug-in mean, but the DR mean stays positive
and the pre-registered sign pattern is still not recovered; p10 stays positive.
Fits: results/fit_trim005_095_seed*.json. (The stricter [0.10,0.90] fits were computed but are no
longer reported; see EVIDENCE_LEDGER for their values.)
