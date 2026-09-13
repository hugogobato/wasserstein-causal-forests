# Sensitivity and data-generating-process experiments

This document specifies the next simulation block. It is a design document, not a report of completed estimator runs. The existing results use $K=25$ and $M=10$; the sensitivity cells below must be run before those values are presented as stable choices.

## What the current suite already covers

The current implementation has two different types of nonlinear structure.

1. The abstract regimes `D0`--`D9` already contain nonlinear outcome surfaces. The prognostic surface contains \(\sin(\pi x_1)\) and the interaction \(x_2x_3\); `D3` also uses \(x_4^2\), `D6` uses a multimodal outer location law, `D7` changes the inner shape, and `D8` uses a stronger assignment index. These regimes are symmetric in the inner quantile shape except for `D7`, where the shape coefficient is nonzero. The `D2-imb`, `D7-imb`, and `D8-imb` variants add an imbalanced assignment mechanism.
2. The income regimes `IC0`--`IC3` have right-skewed inner quantile contours, six covariates, and logistic assignment indices. `IC0` is a placebo, `IC1` and `IC2` use moderate endogenous adoption, and `IC3` deteriorates overlap. The phase-6.5 `DAskew` ablation removes the inner shape, and `DArand` replaces the assignment probability by one half.

The current suite therefore tests nonlinear outcome regression and changing overlap, but it does not give the final WCF a dedicated nonlinear propensity surface. The existing assignment functions are logistic links of linear indices in the covariates, apart from the nonlinear link itself. The new block should keep the logistic propensity as a deliberate misspecification benchmark and add a flexible-propensity implementation for a sensitivity check.

The cheap probe in `research/checks/dgp_stress_probe.py` is a construction check only. It generated a symmetric inner law with a nonlinear outcome and assignment surface, mean propensity $0.457$, support $[0.05,0.93]$, treated fraction $0.454$, standardized mean difference $0.818$ on $X_1$, and grid skewness essentially zero in both arms. These values show that the proposed regime creates meaningful covariate imbalance without making a right-skewed outcome. They do not measure estimator performance.

## Existing execution path

The cell contract is `Cell(grid, dgp, n_train, n_grid, n_particles, method, seed)` in `src/wasserstein_causal_forests/g3/manifest.py`. The phase-6 launcher is `research/run_phase6.py`; phase 6.5 uses `research/run_phase65.py`. Both shard by replication, pin numerical libraries to one thread, write one parquet per shard, and merge only after cell-key validation. The Colab equivalents are `colab/00_setup.ipynb`, `colab/01_run_shard.ipynb`, and `colab/02_merge_and_analyze.ipynb`; phase-6.5 shard notebooks are under `colab/phase65_shards/`.

For the final estimator, use the `cwdb_dr` adapter in `src/wasserstein_causal_forests/g3/phase6_methods.py`, and label its output WCF in the paper. It fits the shared particle law and then applies the AIPW calibration to declared functionals and the reference distance. The calibration does not change the predicted particle law, `mean_quantile_rmse`, or `kernel_law_error`. The primary comparison should include `causal_drf` and `drf` on every cell that produces a law. `cwdb_zipt` should be included only in the zero-inflation block.

Do not reuse the old `particles` and `resolution` rows as evidence for the final WCF. Those rows use earlier method identifiers and are useful as a template for manifest construction, but the new manifest must contain `cwdb_dr` and record its exact adapter parameters. In particular, the existing frozen phase-6 manifest records the historical fold settings; the new manifest should be the source of truth for the rerun and should be kept beside, rather than mixed into, the historical parquet files.

## Grid and particle sensitivity

The first sensitivity block should use the income regimes `IC0`--`IC3`, $n=1{,}000$, the common test design of 1,000 rows, and seeds 0--9. The primary grid is

\[
K\in\{5,25,49\},\qquad M\in\{5,10,25\}.
\]

Run the one-factor slices before the interaction grid. For grid resolution, hold $M=10$ and run all three values of $K$. For particle resolution, hold $K=25$ and run all three values of $M$. If the one-factor slices show a material change, run the $3\times3$ factorial only on `IC1` and `IC3`, the regimes with a policy effect and the most difficult overlap. This reduces the factorial to 180 WCF cells before baselines, while preserving the interaction that could otherwise be missed.

For every cell record `mean_quantile_rmse`, `kernel_law_error`, `tate_functional_rmse`, `tcate_functional_rmse`, `reference_effect_rmse`, and `reference_tcate_rmse`, together with runtime, peak memory, selected contrast shrinkage, and the effective particle support. Aggregate by seed after first averaging arm-specific law metrics and functional-specific TATE or TCATE rows within a cell. Report the evaluator's definitions explicitly: the TATE and reference-ATE rows are absolute errors of the marginal contrast in each replication, whereas TCATE and reference-TCATE are RMSEs over moderator bins. Use the mean absolute replication error or a root mean square over replications consistently, and do not call the former a standard error of test-row RMSE.

The reference vector must be regenerated on the same $K$-grid as the truth. The finite-grid targets are `REF-ATE-K` and `REF-TCATE-K`; they should not be pooled across $K$ as if they were the same continuum estimand. The kernel bandwidth, zero-mass tolerance, boosting budget, propensity clipping, and test-seed rule stay fixed. Forest baselines do not use $M$; repeat their $K$-sensitivity rows once per $K$, and document that duplicating them over $M$ would create identical cells. To separate grid approximation from estimator error, generate a common dense evaluation design and evaluate every fitted law on the declared grid for that cell. If predictions are interpolated from a coarser grid for a common diagnostic, label that diagnostic separately and never mix it with `REF-ATE-K` or `REF-TCATE-K`.

The sensitivity decision rule should be fixed before inspecting outcomes. A practical rule is to retain $K=25,M=10$ if each one-factor change stays within 10 percent of the primary WCF reference-TCATE and law errors in IC1 and IC3, and if no larger grid or particle count improves both targets by more than two between-seed standard errors in both regimes. If the rule fails, report the selected pair and give the complete sensitivity table rather than silently changing the primary specification.

## Symmetric and nonlinear assignment regimes

Add a separate DGP family whose inner law is symmetric and whose assignment creates treated-control imbalance. Use six independent $U[-1,1]$ covariates and the finite-grid law

\[
q_k(Y^a)=\mu(X)+a\,\tau(X)+\xi+\exp\{s(X)\}z_k,
\]

where

\[
\begin{aligned}
\mu(x)&=0.70\sin(\pi x_1)+0.35x_2x_3+0.20\cos(\pi x_4),\\
\tau(x)&=0.25+0.15\sin(\pi x_2),\\
s(x)&=0.12+0.08x_5^2-0.05x_3,
\end{aligned}
\]

and $z_k=\Phi^{-1}\{(k-1/2)/K\}$. Set the inner shape coefficient to zero and use a symmetric outer location shock $\xi\sim N(0,0.25^2)$, with no outer log-scale shock in the strict symmetry block. This gives a nonlinear conditional mean and treatment effect while keeping the conditional quantile contour symmetric.

Use three assignment mechanisms with the same outcome law: `SYM-RANDOM`, $e(x)=0.5$; `SYM-LIN`, $e(x)=\operatorname{clip}\{\sigma(1.4x_1-1.1x_2+0.4x_3),0.05,0.95\}$; and `SYM-NL`,

\[
e(x)=\operatorname{clip}\{\sigma(1.6\sin(\pi x_1)+0.8x_2x_3-0.7x_4^2+0.35x_5),0.05,0.95\}.
\]

Add `SYM-MU`, which replaces the nonlinear index by $1.2\mu(x)+0.5x_5$, again clipped to $[0.05,0.95]$. This is the direct test of the distinction between ordinary observed-covariate confounding and a propensity that is a function of a baseline outcome surface. In all four cases the propensity is a function of pre-treatment $X$, not of the unobserved potential outcome. The `SYM-NL` and `SYM-MU` regimes should have their treated fraction, propensity range, and standardized covariate differences reported before estimator scores are interpreted.

Use $n\in\{500,1{,}000\}$, $K=25$, $M=10$, and ten seeds. Score the same six targets as in the income track. The primary scientific comparison is WCF against Causal-DRF and DRF under `SYM-LIN`, `SYM-NL`, and `SYM-MU`; `SYM-RANDOM` is a calibration control. The null version of each DGP, obtained by setting $\tau(x)=0$, should be included if the final paper makes a placebo claim.

## Propensity-model sensitivity

The current AIPW implementation in `src/wasserstein_causal_forests/cwdb/dr_calibration.py` fits cross-fitted `LogisticRegression` and clips predictions to $[0.02,0.98]$. This is an implementation choice, not a requirement of the WCF construction. Before running nonlinear propensity DGPs, refactor the AIPW class to accept a binary-probability estimator factory. Keep the default factory equal to the current logistic model, and add a flexible factory such as a cross-fitted random forest or gradient-boosted classifier. The factory must be fit inside each training fold, use only training-fold data for tuning, and expose the same clipped probability interface.

On `SYM-NL` and `SYM-MU`, run WCF with the logistic and flexible propensity estimators. Keep the outcome learner, $K$, $M$, seeds, and test designs fixed. The relevant contrast is the change in `REF-ATE-K`, `REF-TCATE-K`, TATE, and TCATE between the two propensity models. A logistic-only result on a nonlinear propensity DGP is a misspecification stress test; it cannot establish the doubly robust property by itself. The flexible-model result should be accompanied by its tuning rule, fold assignment, clipping rate, and effective support.

## Resource plan and reproducibility

The one-factor $K$ and $M$ slices are 4 regimes by 3 values by 10 seeds for WCF, plus one baseline row per $K$. Do not assume that this fits comfortably: run the existing cost pilot at one representative cell for each $(K,M)$ pair, inspect peak memory and any concurrent workload, and only then choose the worker count. Pin one numerical thread per worker and keep the total workers below the available physical-core and RAM budget. The $3\times3$ interaction block and the nonlinear DGP block are independent and can be split across Colab notebooks by DGP, $K$, or $M$. Each notebook must contain its own setup, manifest slice, shard runner, output parquet, and the safe Colab download fallback from the project instructions. Do not place new result files in `.gitignore`.

Before merging, validate that every cell key is unique, every method shares the same test seed within a replication, every DGP has the requested treatment support, and every law-producing method has all six metric families. Keep the probe JSON, frozen manifest, execution logs, merged parquet, and an analysis script together. Only after this audit should the selected $K$ and $M$ be moved into the paper's main implementation description.

## Concrete implementation work packages

### 1. Build an isolated manifest and launcher

Create `research/run_wcf_sensitivity.py`, using `research/run_phase6.py` as a template rather than changing its historical output directories. Its `freeze`, `run`, and `merge` commands should operate exclusively under `results/wcf_sensitivity/`. Import `wasserstein_causal_forests.g3.runner` first to register the DGPs and adapters. Construct `Cell` objects directly; the historical phase launchers enumerate restricted grids and will not discover new cells merely because a JSON file was edited.

The following illustrates actual existing interfaces for manifest construction, without executing any fits:

```python
from wasserstein_causal_forests.g3 import runner
from wasserstein_causal_forests.g3.manifest import Cell, METHOD_REGISTRY, BOOSTING_BUDGET

pairs = {(5, 10), (25, 10), (49, 10), (25, 5), (25, 25)}
cells = [
    Cell("wcf_sensitivity_v1_logit_f3", dgp, 1000, k, m, "cwdb_dr", seed)
    for dgp in ("IC0", "IC1", "IC2", "IC3")
    for k, m in sorted(pairs)
    for seed in range(10)
]
baseline_cells = [
    Cell("wcf_sensitivity_v1_logit_f3", dgp, 1000, k, 10, method, seed)
    for dgp in ("IC0", "IC1", "IC2", "IC3")
    for k in (5, 25, 49)
    for method in ("causal_drf", "drf")
    for seed in range(10)
]
assert len(cells) == 200
assert len(baseline_cells) == 240
assert len({c.key for c in cells + baseline_cells}) == 440
```

Before execution, explicitly set and serialize the `cwdb_dr` registry entry's `parameters` to three folds and contrast candidates `[0,50,500]`, matching the principal IC specification. Save the entire resolved registry entry, boosting budget, software revision, clipping interval, classifier parameters, and estimator implementation hash with the cells. `Cell.key` does not hash these additional settings, so use a distinct grid or method identifier for every different classifier/fold specification and refuse to merge conflicting configurations. Do not mutate historical manifests. The factorial adds 80 new WCF cells after reusing its 100 overlapping one-factor cells; there are 180 distinct WCF cells in the two-regime factorial itself.

The worker can call `runner.run_cell(cell, manifest_contract_id="WCF-SENSITIVITY-v1")` and `runner.write_rows(rows, output_path)` with a new path. Save the accumulated output after **each** completed cell, using a temporary file and atomic replacement. The existing `run_shard` saves every 20 cells, so its log is not by itself a durable per-cell checkpoint. On resume, read only successful cell keys from the actual parquet, retain the prior rows, and append new cells. Keep failure rows and their reasons. A failed cell is not a successful resume marker.

Planned commands after implementing this new launcher are:

```text
rtk python3 research/run_wcf_sensitivity.py freeze --output results/wcf_sensitivity
rtk python3 research/run_wcf_sensitivity.py pilot --workers 1
rtk python3 research/run_wcf_sensitivity.py run --workers <measured-safe-count>
rtk python3 research/run_wcf_sensitivity.py merge
```

These commands describe the new interface to implement; this revision does not create that launcher or run the 440-cell study.

### 2. Compare resolution on a common target as well as each native grid

Native-grid metrics alone change the estimand when K changes. Add a separate common-grid evaluation adapter with 199 midpoint levels. Interpolate each predicted quantile particle and each forest atom linearly in quantile level; use constant endpoint continuation outside the fitted grid and state that choice. Construct the true DGP curves and reference directly on the 199-level grid, without interpolating their coarse versions. Evaluate all methods against the same dense reference, truth quadrature, covariates, and kernel bandwidth per replication. Keep these metrics under distinct IDs such as `REF-TCATE-COMMON199`, never overwrite native-grid outputs. A second interior-grid diagnostic on levels between 0.1 and 0.9 avoids extrapolation for every proposed K and shows how much of the resolution change is due to tail coverage.

For common-grid calibrated functional effects, compute the out-of-fold nuisance means using these interpolated particles and evaluate observed functionals on the corresponding interpolation of the observed coarse curve. This evaluates the coarse-observation estimator against the dense target and includes discretization error; the AIPW identity for the dense target is not exact when the observed transformed outcome itself is approximated. A separate oracle-observed-functional diagnostic may use dense observed curves, but must be labeled as a different information setting. Do not silently reuse native-grid AIPW effects as dense-grid effects.

Use paired seed differences, with Monte Carlo SE computed from those differences, for sensitivity comparisons. If the proposed 10% rule is used to choose a new primary setting, confirm the choice on new seeds, for example 20--39, without reselecting it on those results. A ten-replication table is an initial sensitivity assessment, not a precise equivalence test. Also report each functional separately; averaging errors of skewness and monetary-scale functionals can conceal opposing changes.

### 3. Isolate assignment alignment

Add a constant-propensity control `SYM-RANDOM03` with $e(x)=0.3$. It directly tests the distinction between unequal allocation and confounding. For an alignment comparison, use the same outcome surfaces above and compare $e_{aligned}(x)=\sigma\{1.6\sin(\pi x_1)\}$ with $e_{irrelevant}(x)=\sigma\{1.6\sin(\pi x_6)\}$. The sixth coordinate does not enter any outcome surface. These assignment probabilities have the same marginal distribution because the covariates are independent and identically distributed. Their relation to prognosis differs. This is a cleaner alignment contrast than comparing unrelated logistic indices with different propensity spreads. Report overlap and balance for each pair, including balance of nonlinear prognostic features, since raw covariate mean differences alone can miss nonlinear selection.

Implement the new DGP definitions through `DGPSpec`, `DistributionalDGP`, and `register_specs`/`register_builders`, using the checked probe as a starting point. Register the builders inside every worker process and Colab notebook. Use a distinct name for each assignment mechanism and its null-effect companion. Add one flexible-propensity method identifier and an oracle-propensity diagnostic identifier after implementing the estimator factory. The oracle option should be labeled as a diagnostic, not a feasible competitor. Run design assertions before fits: finite and nondecreasing quantiles, pointwise inner symmetry, nonempty arm samples, and the intended propensity bounds.

### 4. Pilot and allocate local/Colab work

The energy-gradient computation has pairwise-particle work proportional to $n M^2 K$ before chunking; tree fitting also depends on $n M K d$. The cost of law scoring depends on truth nodes and predicted atoms and may dominate forest fits. Measure wall time and total worker memory on the largest $(K,M)$ and both representative sample sizes. Record free RAM and concurrent experiments first. Reserve memory for the operating system and other work, and use at most `floor(available_experiment_RAM / measured_peak_worker_RAM)` workers with a further margin for variability. Do not default to using every logical CPU.

Reuse the notebook setup, dependency installation, R comparator installation, output and resume patterns in `research/checks/phase65_make_colab_notebooks.py`, but replace its allocation function and fixed manifest with the new cells. Each notebook must install/import the exact same code, register all new DGPs, load its embedded manifest slice, and work independently. Bundle the exact revised source or a pinned revision containing it, rather than assuming uncommitted local files will exist in Colab. Allocate by measured replication cost, with a target below eight hours and a hard stop safely before ten. Use up to 48 independent shards only if the pilot justifies that many; keep workloads of roughly 30 minutes together rather than splitting them further. Respect the user's approximate 12 GB RAM per Colab environment and confirm the actual CPU allocation at runtime.

Each notebook should produce one ZIP containing its manifest, parquet, log, diagnostics, and configuration. Use the required safe download fallback:

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

output_file = "wcf_sensitivity_shard.zip"
with ZipFile(output_file, "w", ZIP_DEFLATED) as archive:
    for path in Path("shard_output").rglob("*"):
        if path.is_file():
            archive.write(path, arcname=path.relative_to("shard_output"))
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
```

## Deliverable and completion criteria

The completed study should include the versioned manifest and code, all successful and failed cell records, per-functional and common-grid tables with Monte Carlo uncertainty, paired sensitivity plots, runtime/memory summaries, and a written decision about whether K=25 and M=10 are adequate for the tested setting. No theoretical consistency or efficiency claim follows from this experiment alone. The current revision provides the plan and the successful DGP construction probe; full estimator sensitivity runs remain to be performed.
