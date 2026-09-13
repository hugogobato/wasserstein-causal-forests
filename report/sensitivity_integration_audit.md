# Independent audit: WCF sensitivity integration into `report/wcf_main.tex`

**Auditor:** independent re-computation from the frozen artifacts, no manuscript or table/script was modified.
**Date:** 2026-09-12
**Scope:** the three sensitivity fragments (`sensitivity_methods_fragment.tex`, `sensitivity_results_fragment.tex`, `sensitivity_appendix_fragment.tex`), the seven generated tables under `report/tables_generated/wcf_sensitivity_*.tex`, the surgical edits to `report/wcf_main.tex`, and every numeric claim in the associated text and captions.

## Verdict

The integration is **sound in substance**. Every headline number reproduces from the raw artifacts under `results/wcf_sensitivity/` (rebuilt from `merged/wcf_sensitivity_combined.parquet` and the per-stage manifests), the semantic mapping of every table column is the intended metric/target, the parity language is statistically defensible, the style constraints hold, the labels and citations resolve, and the paper compiles cleanly (23 pages, zero errors, zero undefined references or citations, zero overfull boxes).

**Result: 53 PASS / 6 FAIL.** The six failures comprise one substantive provenance gap, one root cause of double rounding that surfaces in two claims (one table cell pair and the matching text), one scope overstatement, and two git-baseline limitations. The one that requires attention:

1. **Most important discrepancy.** The gradient-boosting calibration numbers in `sensitivity_results_fragment.tex` (OOF predictions "reach 0.001 and 0.998", "4 to 12 percent ... clipped", propensity RMSE "0.22 to 0.23" vs "0.05 to 0.16" for logistic and "0.09 to 0.10" for the random forest) are **not recorded in any persisted result artifact**. The combined parquet stores only `diagnostic_ehat_mean`; no propensity min/max, clipping fraction, or propensity RMSE is written anywhere under `results/`. The values exist only in prose (`report/sensitivity_results_fragment.tex` and `results/wcf_sensitivity/analysis/final_decision.md`). This is the one failure that affects a substantive explanatory claim (that the degradation is a factory property).
2. **Double rounding in the fresh-seed confirmation block.** The table value `-43.8` for IC1 full-range law and `3.6` for the IC3 full-range reference-TCATE SE are produced by re-formatting values already rounded to 2 to 6 decimals in `confirm_paired.csv`. A single-rounding from the raw cell metrics gives `-43.7` and `3.7`. The corresponding text quote "43.8" inherits the same 0.1 discrepancy.
3. **M=25 scope.** "At most 1.2 percent" holds on the particle-budget panel (K=25, maximum 1.160 percent at IC0) but not across the full factorial grid tested in the same study (1.476 percent at K=5, IC1).
4. **Git regression evidence unavailable.** `report/wcf_main.tex` (and the three fragments and seven tables) are untracked, so `git diff` cannot verify that the integration consisted only of the seven replacements plus three `\input` insertions, and cannot exclude an accidental deletion against a baseline. The edit log, compile, and input checks are consistent with the intended edits.

None of these changes the qualitative conclusions, the retained primary specification `(K,M)=(25,10)`, or the ranking claims.

## 1. Number verification

All values below were recomputed from `results/wcf_sensitivity/merged/wcf_sensitivity_combined.parquet` using the frozen cell-aggregation semantics (status `ok`, `metric != cell_failure`, arm rows averaged within a cell, functional targets kept separate) and cross-checked against `cell_metrics.csv`, `one_factor_k.csv`, `one_factor_m.csv`, `confirm_paired.csv`, `sym_methods.csv`, `flex_rf_paired.csv`, `null_companions.csv`, `runtime_summary.csv`, `design_checks.json`, `decision.json`, and `final_decision.md`. The rebuilt cell table matches `cell_metrics.csv` exactly (58,000 rows, max value difference `4.55e-13`).

| # | Check | Result | Exact supporting value / computation |
|---|-------|--------|--------------------------------------|
| A1 | `cell_metrics.csv` reproducible from combined parquet | PASS | 58,000 rows built from 71,680 raw rows; csv-only 0, rebuilt-only 0; max abs diff `4.55e-13` |
| A2 | Stage manifests and cell inventory | PASS | primary 880, nulls 360, confirm 160, flex_rf 40, factorial 80 newly merged (280 declared, 200 one-factor cells already counted in primary); combined 1,520 cells, 71,680 rows, 0 failures, no duplicate keys |
| A3 | All fragments and table inputs exist | PASS | 3 fragments + 7 `wcf_sensitivity_*.tex` + all other `\input` targets present |
| A4 | Retention-rule percentages 15.5 / 14.1 / 19.1 | PASS | raw: K=5 ref-TCATE 15.47 %, K=49 ref-TCATE 14.08 %, K=5 law 19.09 %; `decision.json` 15.47 / 14.08 / 19.09 |
| A5 | Decision-rule outcome | PASS | condition A false (12/16 within tolerance), condition B any improvement false, `retain_primary=false`, `recommended_pair=null`, reason "at least one one-factor alternative leaves the relative-tolerance band and no larger pair proves an improvement" |
| A6 | K/M table, all 120 entries and incumbent duplication | PASS | Panel A + B exact string match for 4 regimes x 3 K x 5 columns and 4 x 3 M x 5 columns; IC1 `(25,10)` row identical in both panels (`0.0309(0.0026)`, `0.0605(0.0009)`, `0.0290(0.0028)`, `0.0195(0.0007)`, `0.0317(0.0030)`) |
| A7 | Resolution table, primary seeds 0-9, 20/20 cells | PASS | exact string match; e.g. IC1 full-range law `-43.4(1.2)`, IC1 full-range ref-TCATE `-16.4(6.9)`, IC3 ref-ATE common `-6.3(20.1)` |
| A8 | Resolution table, confirmation seeds 20-39, 20 cells | **FAIL** | 18/20 exact. IC1 full-range law: printed `-43.8(1.0)`; single-round of exact `-43.746 (1.035)` is `-43.7`. IC3 full-range ref-TCATE: printed `0.3(3.6)`; exact `0.320 (3.650)` gives SE `3.7`. Root cause: `confirm_paired.csv` stores `relative_pct` at 2 decimals and `mc_se`/`mean_a` at 6 decimals, and the table re-formats those. See correction F2 |
| A9 | Text: confirmation law reductions "46.5, 43.8, 44.1, 42.8" | **FAIL** | raw single-rounded values: `-46.5, -43.7, -44.1, -42.8`; only IC1 differs (`-43.74588` rounds to `-43.7`, not `-43.8`) |
| A10 | Text: confirmation interior law changes -1.7 / -3.4 / -1.3 / -1.3 | PASS | exact: `-1.681, -3.394, -1.261, -1.301` |
| A11 | Text: primary full-range ref-TCATE +4.2 / -16.4 / -3.5 / +5.7 | PASS | exact: `+4.20, -16.38, -3.53, +5.72` |
| A12 | Text: primary interior ref-TCATE +8.2 / -15.0 / +6.4 / +3.9 | PASS | exact: `+8.25, -15.00, +6.42, +3.94` |
| A13 | All paired interior-law `|t| < 1.4` on both seed samples | PASS | exact t: primary `-1.02, +0.16, -0.92, +0.37`; confirm `-0.76, -1.32, -0.64, -0.49` |
| A14 | IC1 full-range ref-TCATE t = -2.4 primary, -2.2 confirmation | PASS | exact `-2.391` (-16.379/6.850) and `-2.201` (-7.919/3.598) |
| A15 | IC3 native marginal reference-effect regression 20.8 % | PASS | `confirm_paired.csv` IC3 `REF-ATE-K`: `+20.83 %`, `t_like=2.85` |
| A16 | IC3 full-range common marginal reference-effect 12.1 % | PASS | exact `+12.133 %` (SE 6.1), matches table `12.1(6.1)` |
| A17 | M=25 common-grid law change <= 1.2 % on the particle-budget panel (K=25) | PASS | max `|change| = 1.1599 %` at IC0 (IC1 -0.41, IC2 +0.47, IC3 -0.37); rounds to 1.2 |
| A18 | M=25 common-grid law change <= 1.2 % across the full tested factorial | **FAIL** | `(K=5, M=25)` IC1 changes by `-1.4759 %`; the sentence couples this claim with a cost ratio that spans K=5/25/49, so the unqualified 1.2 % is over-scoped. See correction F3 |
| A19 | M=25 costs four to eight times M=10 | PASS | median ratios (M=25 / M=10): K=5 `4.19x`, K=25 `4.07x`, K=49 `7.70x` |
| A20 | M=5 interior ref-TCATE worse in IC0/IC1/IC2; 0.0320 vs 0.0317 in IC1 | PASS | IC0 `0.0297 vs 0.0292`, IC1 `0.0320 vs 0.0317`, IC2 `0.0330 vs 0.0315` |
| A21 | M=5 interior law essentially unchanged | PASS | `0.0173/0.0178`, `0.0202/0.0195`, `0.0192/0.0198`, `0.0226/0.0228` (IC0-IC3, M=5 vs M=10) |
| A22 | M=5 native ref-TCATE worse in IC0 (0.0298 vs 0.0286); indistinguishable in IC1 (0.0300 vs 0.0309) | PASS | exact means; IC1 paired `t=-0.57` |
| A23 | Factorial IC1 row minima M=10 -> M=25 -> M=5 as K rises | PASS | minima: K=5 M=10; K=25 M=25; K=49 M=5 |
| A24 | No larger pair improves both targets in both regimes by 2 SE | PASS | `decision.json` condition B `any_improves=false`; all 12 combination records have `improves=false` |
| A25 | (49,10) costs 1.6 times (25,10) | PASS | `524.4402 / 330.2631 = 1.5879` |
| A26 | Assignment table, 24 WCF/Causal-DRF/DRF cells | PASS | exact string match, e.g. n=500 WCF `0.0444/0.0452/0.0597/0.0461`, n=1000 WCF `0.0286/0.0350/0.0417/0.0287` |
| A27 | Assignment text values for all four mechanisms and both n | PASS | all 16 WCF and cited comparator values reproduced, e.g. Causal-DRF n=1000 `0.0313/0.0392/0.0814/0.0461`, DRF n=500 `0.0417/0.0428/0.0746/0.0501` |
| A28 | Margin under nonlinear mechanisms grows with n | PASS | SYM-NL WCF/Causal-DRF `0.0597/0.0889=0.67` at n=500 vs `0.0417/0.0814=0.51` at n=1000; SYM-MU `0.81` vs `0.62` |
| A29 | Alignment pair differences and p-values | PASS | `+0.00413 (p=0.478)` ref-TCATE and `-0.00169 (p=0.108)` law at n=500; `-0.00081 (p=0.877)` and `-0.00043 (p=0.725)` at n=1000 |
| A30 | Propensity table, 16 cells (logistic, RF, GB, oracle) | PASS | exact string match, e.g. SYM-NL n=1000 `0.0417/0.0383/0.1471/0.0339`; SYM-MU n=1000 `0.0287/0.0291/0.1057/0.0261` |
| A31 | Propensity text GB/logistic/RF values | PASS | GB/NL `0.1471` vs `0.0417`; GB/MU `0.1057` vs `0.0287`; RF/NL `0.0383` vs `0.0417` |
| A32 | Gradient boosting three to four times worse | PASS | `0.1471/0.0417 = 3.53x`; `0.1057/0.0287 = 3.68x` |
| A33 | RF paired differences -8.1 to +1.9 %, all `|t| < 1.4` | PASS | cells `-5.41 (t=-0.81)`, `+1.87 (t=0.52)`, `-8.07 (t=-1.31)`, `+1.41 (t=0.25)`; max `|t|=1.31` |
| A34 | Oracle upper bound 18.7 % (0.0339 vs 0.0417) | PASS | `flex_rf_paired.csv`: `mean_a=0.0417`, `mean_b=0.0339`, `relative=-18.66 %`, `t=-3.54` |
| A35 | GB calibration numbers 0.001/0.998, 4-12 % clipped, RMSE 0.22-0.23 / 0.05-0.16 / 0.09-0.10 | **FAIL** | no persisted artifact contains propensity ranges, clipping fractions, or propensity RMSE; only `diagnostic_ehat_mean` exists. See correction F1 |
| A36 | Design checks SYM-NL | PASS | min `0.050`, max `0.9320`, treated `0.4557`; SMDs `mu=1.131`, `sin(pi x1)=1.092`, `x1=0.819`, `x2x3=0.212` |
| A37 | Design checks SYM-MU | PASS | min `0.131`, max `0.8698`, treated `0.4992`; SMDs `mu=0.615`, `sin(pi x1)=0.574`, `x1=0.444`, `x5=0.258` |
| A38 | Grid skewness maximum 1.8e-15 | PASS | max abs over arms and regimes `1.7509e-15` |
| A39 | Alignment marginal max abs difference 0.0018 | PASS | `0.0018036` |
| A40 | Design checks on 200,000 draws per regime, all passed | PASS | `n_rows=200000`, `checks_passed=true` |
| A41 | Placebo table, 24 cells | PASS | exact match after pooling the six null designs (60 values per cell, mean and `sd/sqrt(60)`), e.g. n=1000 WCF `0.0295(0.0028)`, `0.0466(0.0025)`, `0.0135(0.0014)`, `0.0352(0.0016)`; the null DGP companions set `tau(x)=0` in `sensitivity_dgps.py`, so the caption's "exactly zero treatment effects" is supported |
| A42 | Placebo row/panel ordering | PASS | order is `TATE grid mean`, `TCATE grid mean`, `Ref. TATE`, `Ref. TCATE`, for n=500 then n=1000; caption's "first two rows grid-mean functional, last two reference-distance" is correct |
| A43 | Placebo text values | PASS | all fifteen cited values reproduced exactly to 4 decimals, including n=1000 ref-TCATE `0.0352` vs `0.0476`/`0.0519` and n=500 ref-TCATE `0.0449` vs `0.0504`/`0.0556` |
| A44 | WCF smallest on every placebo target | PASS | 8/8 target-by-sample cells |
| A45 | Cost table, 9 rows | PASS | medians and ratios exact against `runtime_summary.csv`, e.g. `(25,10)=330.2631 s`, `(49,10)=524.4402 s`, `(25,25)=1345.5619 s` |
| A46 | Cost text ratios 1.6, 4.1, 0.35 | PASS | `1.588x`, `4.074x`, `0.353x` |
| A47 | Seven concurrent single-threaded workers | PASS | seven shard logs (`shard_000` to `shard_006`) in the primary, factorial, and flex_rf stages; maximum concurrent overlap from `finished_at - wall_seconds` windows is 7; `research/run_wcf_sensitivity.py:287,693` pins each worker to one numerical thread |
| A48 | Factory, fold, and candidate settings | PASS | RF factory 200 trees, `min_samples_leaf=5`, `n_jobs=1`; GB `HistGradientBoostingClassifier` defaults; all variants `n_folds=3`, candidates `{0,50,500}`; factory predictions (and the oracle) clipped to `[0.02,0.98]` in `FunctionalAIPW` |
| A49 | Table captions' numeric claims | PASS | "mean(SE) over ten replications" verified (10 seeds), confirmation seeds 20-39 (20), placebo pooling over six designs (60), cost medians and `(25,10)` ratio, resolution percentage-change sign convention (K=49 minus K=25) all verified against the computed series |

## 2. Statistical-language check (parity)

Seed-paired WCF-minus-comparator differences, native `REF-TCATE-K`, ten seeds, paired t with 9 degrees of freedom.

| # | Check | Result | Exact supporting value / computation |
|---|-------|--------|--------------------------------------|
| B1 | "Parity" defensible for SYM-RANDOM and SYM-LIN at the 5 percent level, both comparators and sample sizes | PASS | eight paired tests, all `p >= 0.115` with the t distribution and `p >= 0.081` under the normal approximation; largest `|t| = 1.75` (SYM-LIN, n=1000, vs Causal-DRF). Details in the table below |

| Mechanism | n | WCF | Comparator | Paired diff | t | p |
|-----------|---|-----|------------|-------------|---|----|
| SYM-RANDOM | 500 | 0.0444 | Causal-DRF | +0.00219 | +0.36 | 0.729 |
| SYM-RANDOM | 500 | 0.0444 | DRF | +0.00272 | +0.56 | 0.587 |
| SYM-RANDOM | 1000 | 0.0286 | Causal-DRF | -0.00268 | -0.67 | 0.519 |
| SYM-RANDOM | 1000 | 0.0286 | DRF | +0.00106 | +0.27 | 0.797 |
| SYM-LIN | 500 | 0.0452 | Causal-DRF | -0.00961 | -1.28 | 0.234 |
| SYM-LIN | 500 | 0.0452 | DRF | +0.00242 | +0.33 | 0.747 |
| SYM-LIN | 1000 | 0.0350 | Causal-DRF | -0.00424 | -1.75 | 0.115 |
| SYM-LIN | 1000 | 0.0350 | DRF | +0.00103 | +0.38 | 0.714 |

- **Interpretation.** "Parity" is defensible at the 5 percent level for both comparators and both sample sizes: no paired difference rejects equality (`p >= 0.115` in all eight cells; even the normal-approximation p-value in `sym_methods.csv` is 0.081 at its smallest). WCF is also significantly better under SYM-NL at n=1000 against both baselines (`p < 0.001`) and under SYM-MU at n=1000 (`p = 0.000` and `0.002`), so "at parity or better" is accurate at the level of statistical evidence.
- **Caveat (not a FAIL).** On point estimates WCF is slightly worse than DRF under SYM-RANDOM and SYM-LIN at both sample sizes (paired means +0.0010 to +0.0027), and the abstract phrase "at parity or better" should not be read as "never worse in point estimate". The differences are not significant, and the paper reports the point estimates in the table, so the statement is not misleading.

## 3. Semantic check

| # | Check | Result | Exact supporting value / computation |
|---|-------|--------|--------------------------------------|
| C1 | Intended metric/target used by every column | PASS | recomputation used the stated pairs and reproduced every table cell: native = `reference_tcate_rmse`/`REF-TCATE-K`; Law common = `kernel_law_error`/`LAW-A-COMMON199`; Ref. TCATE common = `reference_tcate_rmse`/`REF-TCATE-COMMON199`; Law interior = `kernel_law_error`/`LAW-A-INTERIOR`; Ref. TCATE interior = `reference_tcate_rmse`/`REF-TCATE-INTERIOR`; Ref.-ATE common = `reference_effect_rmse`/`REF-ATE-COMMON199`; factorial, assignment, propensity tables = `reference_tcate_rmse`/`REF-TCATE-K`; placebo = `TATE-K-grid_mean`, `TCATE-K-grid_mean`, `REF-ATE-K`, `REF-TCATE-K`. A mislabeled column would have failed the exact-match reconstruction |
| C2 | Native, common, and interior targets never pooled | PASS | 29 distinct target ids; `REF-TCATE-K`, `REF-TCATE-COMMON199`, `REF-TCATE-INTERIOR` exist as separate series and each table cell was reconstructed from its own series |

## 4. Regression check

| # | Check | Result | Exact supporting value / computation |
|---|-------|--------|--------------------------------------|
| D1 | `git diff --stat` and `git diff report/wcf_main.tex` show only the seven replacements plus three `\input` insertions | **FAIL** | `git diff -- report/wcf_main.tex` produces empty output (exit 0) because `report/wcf_main.tex` is untracked (`?? report/wcf_main.tex`); all fragments, tables, and the manuscript itself are untracked, so git has no baseline. `git diff --stat` lists 18 tracked files (phase65 shards/logs, `research/run_phase65.py`, `src/.../dr_calibration.py`, `dgps.py`, `phase6_methods.py`, `runner.py`), none under `report/`. See correction F4 |
| D2 | No accidental deletion of existing content | **FAIL (not verifiable)** | no baseline exists in git history (`git show HEAD:report/wcf_main.tex` fails: path exists on disk but not in HEAD; no stashes). Indirect evidence is clean: all seven edit pairs from `sensitivity_edits.md` are applied consistently (each new block occurs exactly once; replacement edits leave zero old-block occurrences; append-style edits keep the old sentence inside the new block), the three `\input` lines occur exactly once at lines 315, 358, 382, every `\input` target exists, the full document compiles with zero undefined references, and all labels resolve. See correction F4 |
| D3 | Surgical-edit log consistency | PASS | 7 replacement pairs from `sensitivity_edits.md` (Edits 1, 2, 4, 5, 6, 7, 8; Edit 3 is a documented no-edit) each occur exactly once as the new text; `\input{sensitivity_methods_fragment.tex}`, `\input{sensitivity_results_fragment.tex}`, `\input{sensitivity_appendix_fragment.tex}` at lines 315, 358, 382 respectively |

## 5. Style, label, and citation checks

| # | Check | Result | Exact supporting value / computation |
|---|-------|--------|--------------------------------------|
| E1 | No em dash or `---`, no `itemize`, no `enumerate` | PASS | counts are 0 for the literal em dash, the `---` sequence, `\begin{itemize}`, and `\begin{enumerate}` in `wcf_main.tex` and all three fragments |
| E2 | Every label referenced in the new text resolves | PASS | 21 reference sites resolve in the fresh `wcf_main.aux`; the 7 labels defined by the fragments (`sec:resolution-design`, `sec:assignment-design`, `sec:results-resolution`, `sec:results-assignment`, `app:sensitivity-tables`, `eq:sens-retention-rule`, `eq:sens-common-grid`) and all 7 new table labels are present in both the fresh aux and `report/wcf_main.aux` |
| E3 | No new citation key outside `wcf_references.bib` | PASS | the new fragments cite only `DRF-paper` and `naf2026causaldrf`; both keys are in `wcf_references.bib`; the compile reports zero undefined citations |

## 6. Build check

| # | Check | Result | Exact supporting value / computation |
|---|-------|--------|--------------------------------------|
| G1 | Two-pass `pdflatex` build | PASS | compiled twice in an isolated copy (`/tmp/opencode/wcf_audit_build`, so the repository's build artifacts were not touched); exit 0 both passes; 23 pages; 0 errors; 0 undefined references; 0 undefined citations; 0 overfull `\hbox`; 0 overfull `\vbox`; 0 LaTeX warnings (the only "warning" string in the log is the infwarerr package banner) |

## FAIL corrections (minimal changes)

- **F1 (calibration numbers, A35, most important).** Persist the diagnostic that produced the prose numbers, for example add per-cell metrics `propensity_min`, `propensity_max`, `propensity_clip_fraction`, and `propensity_rmse_vs_truth` for `cwdb_dr_flex` and `cwdb_dr_flex_rf`, and regenerate the flex stage analysis tables. If re-running is not possible, replace the sentence's specific numbers with a statement that the factory's calibration diagnostic was recorded separately, or remove the numeric ranges. As written, "0.001 and 0.998", "4 to 12 percent", and the three RMSE ranges cannot be checked by a reader against any file in the repository.
- **F2 (double rounding, A8, A9).** Compute the confirmation-block percentages directly from the confirm-grid cell metrics (or store full-precision values in `confirm_paired.csv`), so that the text and table read `-43.7` for IC1 full-range law and `3.6` or `3.7` consistently for the IC3 full-range reference-TCATE SE. The discrepancy is at most 0.1 percentage point and changes no conclusion; correcting the text "43.8" to "43.7" is sufficient for consistency with the raw cells.
- **F3 (M=25 scope, A18).** Either restrict the sentence to the particle-budget panel explicitly ("at K=25, ... at most 1.2 percent") or widen the bound to "at most about 1.5 percent across the tested grids" because the factorial extension records 1.476 percent at `(K=5, M=25)` in IC1.
- **F4 (git baseline, D1, D2).** Track `report/wcf_main.tex` (and the three fragments and seven generated tables) in git before or with the integration commit, so that `git diff` and `git diff -- report/wcf_main.tex` can confirm the surgical-edit surface. Until then, the edit log, clean compile, and zero undefined references are the strongest available regression evidence, and no accidental deletion was detected by those means.

## Exact commands run

Repository state and regression evidence:

```bash
git status
git log --oneline -10
git diff --stat
git diff -- report/wcf_main.tex
git status --porcelain report/
git ls-files report/
git show HEAD:report/wcf_main.tex
git stash list
```

Independent numeric recomputation (scripts written only under `/tmp/opencode/`, not in the repository):

```bash
python3 /tmp/opencode/audit_numbers.py
python3 /tmp/opencode/audit_numbers_extra.py
```

The first script rebuilds the cell table from `merged/wcf_sensitivity_combined.parquet`, checks the stage inventories, and recomputes all table and text numbers (retention percentages, K/M table, resolution table and t statistics, M=5 and M=25 comparisons, assignment, alignment, propensity, placebo, cost, design checks, decision rule). The second script reproduces the focused checks reported above (double-rounding scan of `confirm_paired.csv`, exact t statistics, parity tests, full 120-value K/M table reconstruction, build-log parsing, and style/label/edit checks). The scripts print their own automated summary (39 PASS / 3 FAIL over the raw-recomputation checks, where the resolution-table check is counted once); the audit table above splits that check into A8 and A9, adds the M=25 scope determination (A18) and the two git-baseline checks (D1, D2), and yields 53 PASS / 6 FAIL in total.

Isolated build (the repository's `report/wcf_main.aux/.log/.pdf` were not modified):

```bash
mkdir -p /tmp/opencode/wcf_audit_build
cp -r report/. /tmp/opencode/wcf_audit_build/
cd /tmp/opencode/wcf_audit_build
pdflatex -interaction=nonstopmode wcf_main.tex
pdflatex -interaction=nonstopmode wcf_main.tex
pdfinfo wcf_main.pdf
```

Provenance scans:

```bash
grep -rlE "fraction_clipped|propensity_rmse|clip_fraction" results/
grep -rn "clip" results/wcf_sensitivity/
```

## Notes and limitations

- The repository's `report/wcf_main.aux`, `report/wcf_main.log`, and `report/wcf_main.pdf` already existed; to honor the instruction to write only the audit file, the fresh build was performed on a byte-for-byte copy of `report/` under `/tmp/opencode`. All 12 new labels are present in both the fresh aux and the existing `report/wcf_main.aux`.
- The calibration numbers are the only prose values with no persisted source. Everything else in the three fragments and the seven table captions was either recomputed exactly, shown to be a documented rounding artifact, or shown to be an operational claim corroborated by the execution logs (the seven workers).
- The audit did not re-run any estimator; all checks use the frozen artifacts as required.

## Re-audit after corrections

**Date:** 2026-09-12 (same session, follow-up). Scope: re-verify the four substantive FAILs (A8, A9, A18, A35) after the corrections, plus the style, label, and build checks. No file other than this audit was modified; table generation and the propensity-diagnostic run were executed into temporary directories, and the LaTeX build was executed on a refreshed copy under `/tmp/opencode`.

### Updated status of the corrected checks

| # | Check | Was | Now | Exact supporting value / computation |
|---|-------|-----|-----|--------------------------------------|
| A35 | Calibration numbers persisted and reproducible | FAIL | **PASS** | New artifact `results/wcf_sensitivity/analysis/propensity_calibration_diagnostic.json` from `research/checks/wcf_sensitivity_propensity_calibration.py`. GB raw min `0.001278` -> `0.001`, max `0.998250` -> `0.998` (text "reach 0.001 and 0.998" correct); GB clip totals `12.2 %`, `2.2 %`, `8.8 %` -> text "2 to 12 percent" correct; GB RMSE `0.219257`, `0.232880`, `0.229748` -> "0.22 to 0.23"; logistic RMSE `0.159536`, `0.107772`, `0.047291` -> "0.05 to 0.16"; random-forest RMSE `0.100475`, `0.093569`, `0.097530` -> "0.09 to 0.10". Unmodified script re-run in a sandbox produced a JSON **byte-identical** to the persisted artifact (SHA-256 `53d7296b426818b7...`); stdout matches the JSON |
| A8 | Resolution table confirmation cells single-rounded | FAIL | **PASS** | The builder now computes every cell with `_paired_pct` for both seed blocks (`confirm_paired.csv` is retained only as a cross-check). IC1 confirmation full-range law is `-43.7(1.0)` from exact `-43.745878 (1.035020)`; IC3 confirmation full-range ref-TCATE is `0.3(3.7)` from exact `0.320410 (3.650104)`. All 40 cells (4 regimes x 5 columns x 2 blocks) match a single-rounded reconstruction from the frozen cell metrics exactly |
| A9 | Text quote of confirmation reductions | FAIL | **PASS** | `sensitivity_results_fragment.tex` now reads "reductions of $46.5$, $43.7$, $44.1$, and $42.8$ percent"; `final_decision.md` confirmation row IC1 now shows `-43.7%` |
| A18 | M=25 bound scope | FAIL | **PASS** | `sensitivity_results_fragment.tex` now reads "At $K=25$, increasing to $M=25$ changes the common-grid law error by at most $1.2$ percent, and across the factorial grid by at most $1.5$ percent". Data: K=25 maximum `1.1599 %` (IC0), factorial maximum `1.4759 %` at (K=5, IC1), so both bounds hold. `final_decision.md` carries "at most 1.2 percent common-grid law error at K=25 and 1.5 percent across the tested grids" (two occurrences), and `final_decision.json` has `m25_vs_m10_common_law_error_pct_max = 1.5` and `m25_vs_m10_common_law_error_pct_max_at_k25 = 1.2` |
| E1 | Style: no `---`, em dash, `itemize`, or `enumerate` | PASS | **PASS** | counts remain 0 in `wcf_main.tex` and all three fragments after the text edits |
| E2 | Labels resolve | PASS | **PASS** | all 12 new labels present in the freshly compiled aux; zero undefined references |
| E3/G1 | Build, citations | PASS | **PASS** | two-pass `pdflatex` on the refreshed copy: 23 pages, 0 errors, 0 undefined references, 0 undefined citations, 0 overfull `\hbox`, 0 overfull `\vbox`. No citation was added by the corrected text |

Additional re-audit evidence:

- **Table regeneration.** `python3 report/build_wcf_sensitivity_assets.py --output-dir /tmp/opencode/tables_regen` regenerated all seven tables; every regenerated file is **byte-identical** (SHA-256) to the corresponding file in `report/tables_generated/`, confirming that the resolution table is the only one the fix changed and that the repository tables are consistent with the fixed builder.
- **Propensity diagnostic reproducibility.** The unmodified script was copied into `/tmp/opencode/propcheck/research/checks/` with `src/` symlinked to the repository, so it ran the exact production code (DGP builders, factories, fold helper) with only the output root redirected. The resulting JSON was byte-identical to the persisted artifact. The configuration recorded in the artifact matches the AIPW layer exactly: 1,000 rows, cell seed 0, `random_state = seed + 31 = 31`, 5 stratified folds, fold models seeded `random_state + fold`, clip `[0.02, 0.98]`. The code path was confirmed in `src/wasserstein_causal_forests/cwdb/dr_calibration.py` (`AIPW_N_FOLDS = 5`, `random_state=self.random_state + 31`, `factory(random_state + fold)`, `PROPENSITY_CLIP = (0.02, 0.98)`).
- **Pointer sentence.** `final_decision.md` line 143 now reads "... is reported as a documented negative finding (persisted in `analysis/propensity_calibration_diagnostic.json`)", and `final_decision.json` records `"calibration_diagnostic": "analysis/propensity_calibration_diagnostic.json"`.
- **Rounding note (not a discrepancy).** "2 to 12 percent" derives from clip totals of `2.2 %` (SYM-MU) and `12.2 %` (SYM-NL) at integer precision, and "0.001 and 0.998" derive from `0.001278` and `0.998250` at three decimals. Both are consistent roundings of the persisted values.

### D1/D2 (unchanged)

There is no change. `report/wcf_main.tex`, the three fragments, and the seven tables remain untracked (`??`), so `git diff -- report/wcf_main.tex` is still empty and there is still no baseline to prove the absence of accidental deletions. These two entries remain FAILs and are not new findings; the indirect evidence (edit-log consistency, byte-identical regenerated tables, zero undefined references, clean compile) remains clean.

### Updated tally

| State | PASS | FAIL | FAIL entries |
|-------|------|------|--------------|
| Original audit | 53 | 6 | A8, A9, A18, A35, D1, D2 |
| After corrections | **57** | **2** | D1, D2 (untracked git baseline only) |

**Newly found discrepancies: none.** All four corrected checks reproduce against the frozen artifacts, the corrected bounds and quotes are exact at the stated precision, the persisted calibration artifact is reproducible byte-for-byte, and the paper still compiles cleanly.

### Exact commands run for the re-audit

```bash
python3 /tmp/opencode/reaudit_corrections.py

mkdir -p /tmp/opencode/propcheck/research/checks
cp research/checks/wcf_sensitivity_propensity_calibration.py /tmp/opencode/propcheck/research/checks/
ln -sfn /home/hugo_souto/Stuff/Research/Wasserstein_Causal_Forests/src /tmp/opencode/propcheck/src
cp results/wcf_sensitivity/analysis/propensity_calibration_diagnostic.json /tmp/opencode/propensity_calibration_before.json
python3 /tmp/opencode/propcheck/research/checks/wcf_sensitivity_propensity_calibration.py

python3 report/build_wcf_sensitivity_assets.py --output-dir /tmp/opencode/tables_regen
sha256sum report/tables_generated/wcf_sensitivity_*.tex /tmp/opencode/tables_regen/wcf_sensitivity_*.tex

cp -r /home/hugo_souto/Stuff/Research/Wasserstein_Causal_Forests/report/. /tmp/opencode/wcf_audit_build/
cd /tmp/opencode/wcf_audit_build
pdflatex -interaction=nonstopmode wcf_main.tex
pdflatex -interaction=nonstopmode wcf_main.tex
pdfinfo wcf_main.pdf
```
