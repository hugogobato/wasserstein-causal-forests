# Application design memo

Status: WP0 template. No application-design pass is claimed by this file.

This memo is completed by WP10 before any outcome modeling. It must document the unit, treatment, timing, outcome horizon, target population, causal DAG, adjustment set, interference definition, household sampling design, and the interpretation of the region-level distributional outcome.

## Required fields

| Field | Frozen value or status |
|---|---|
| Candidate application | Bolsa Família at the municipality level, provisional only |
| Outer unit | To be verified by WP10 |
| Treatment definition | To be verified as binary municipality-level exposure or adoption |
| Treatment timing | To be verified |
| Outcome horizon | To be verified |
| Target population | Equal-weight region population unless a documented alternative is chosen |
| Covariates \(X\) | To be specified from pre-treatment information |
| Confirmatory modifier \(V\) or subgroups | To be frozen independently of confirmation outcomes |
| Assignment mechanism | To be justified; arbitrary outcome-informed binarization is prohibited |
| Identification assumptions | A1–A4, plus application-specific design assumptions |
| Interference and migration | To be audited explicitly |
| Inner household sampling | To be documented as iid, survey-weighted, clustered, stratified, or another design |
| Effective inner sample sizes | To be reported by region and year |
| Missingness | To be documented before heterogeneous-effect analysis |
| Established scalar validation | At least one conventional average scalar result before ODCF |

## Required WP10 outputs

1. `municipality_feasibility.csv` with municipality, year, effective household sample, survey design cells, treatment exposure, and missingness.
2. A causal DAG and an adjustment-set justification.
3. A treatment-overlap report before outcome modeling.
4. A migration and interference assessment.
5. A frozen analysis protocol and a blinded or outcome-restricted pipeline dry run.
6. `applied_promise_report.md` ending in one of `NO-ADDED-VALUE`, `DESCRIPTIVE-VALUE`, `METHOD-VALUE`, or `SUBSTANTIVE-CAUSAL-VALUE`.

