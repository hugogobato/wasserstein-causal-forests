# Wasserstein Causal Forests (WCF)

**Wasserstein Causal Forests (WCF)** is a nonparametric framework for estimating heterogeneous and marginal causal effects when each observational unit carries a **distribution-valued outcome** represented on a $K$-point quantile grid $Q_i \in \mathcal{Q}_K \subset \mathbb{R}^K$.

Instead of collapsing each unit's outcome distribution to a single scalar summary before estimation—or re-weighting a fixed pool of training distributions that cannot extrapolate beyond observed support—WCF models the conditional law of the random quantile curve $Q_i(a) \mid X_i = x$ directly in Wasserstein space and pairs the fitted law with a doubly robust (AIPW) calibration layer for downstream functionals.

---

## Key Capabilities

1. **Causal Wasserstein Distribution Boosting (`CWDBRegressor`, `CrossFittedCWDBRegressor`)**
   - Represents each treatment arm's conditional law $\widehat{P}_{a,M}^K(x) = M^{-1}\sum_{m=1}^M \delta_{p_{a,m}(x)}$ by $M$ transported particle tracks in the monotone quantile cone $\mathcal{Q}_K$.
   - Updates particles via functional gradient boosting of the negative $\varepsilon$-smoothed Wasserstein energy score, balancing **attraction** to observed unit quantile curves and **repulsion** across particles so within-cell distributional variability is preserved rather than collapsed to a conditional barycenter.
   - Uses **arm-shared regression trees** (`ArmSharedTreeRegressor`) with out-of-fold selection of the treatment-contrast shrinkage penalty $\lambda^\star \in \{0, 50, 500\}$.

2. **Doubly Robust Functional Calibration (`DRCalibratedCWDB`, `FunctionalAIPW`)**
   - Debiases plug-in law functionals (mean, standard deviation, skewness, upper-half mean, quantiles $p_\tau$, and **reference Wasserstein distance** $d_{W,K}(q, q_\star^K)$ to an external benchmark distribution $q_\star^K$) using cross-fitted augmented inverse-propensity weighting (AIPW).
   - Delivers $\sqrt{n}$-consistent marginal treatment effects (TATE), moderator-stratum conditional effects (TCATE), and influence-function standard errors (`dr_if_se`).

3. **Two-Part Estimation for Structural Zeros (`ZIPTAdapter`)**
   - For outcomes with a point mass at the degenerate zero curve $Q_i = \mathbf{0}_K$ (e.g., earnings with non-employment), combines cross-fitted arm-specific positive-mass classifiers $\widehat{\pi}_a(x)$ with positive-stratum CWDB particle boosting into the mixture law $\widehat{P}_{a,\text{ZI}}^K(x) = (1 - \widehat{\pi}_a(x))\delta_{\mathbf{0}_K} + \widehat{\pi}_a(x)\widehat{P}_{a,+}(x)$, and calibrates functional effects via the exact two-part decomposition $h(Q_i) = h(\mathbf{0}_K) + I_i(h(Q_i) - h(\mathbf{0}_K))$.

4. **Applied Microdata Adapter (`AppliedDataset`, `run_wcf`, `run_placebo`)**
   - Converts hierarchical individual/household microdata grouped by experimental units (classrooms, villages, state-years) into validated $K=25$ midpoint quantile datasets and runs standardized primary, multi-seed replication, and permutation-placebo analyses.

---

## Installation

### Python Package

Requires Python $\ge 3.10$. From the repository root:

```bash
pip install -e .
```

To reproduce the exact Python environment used for the paper's experiments and asset generation:

```bash
pip install -r requirements-pinned.txt
pip install -e .
```

### Optional R Dependencies (Forest Baselines Only)

WCF itself is pure Python (`numpy`, `scipy`, `scikit-learn`, `pandas`, `pyarrow`, `matplotlib`). Running the external R forest baselines (`drf` and `causal_drf`) from scratch requires an R installation with `drf` (v1.3.1), `kernlab`, `Matrix`, and `jsonlite` (see `research/baselines/causal_drf_r/README.md`).

---

## Quickstart

### 1. Direct Estimator API (`DRCalibratedCWDB`)

```python
import numpy as np
from scipy.stats import norm
from wasserstein_causal_forests import (
    DRCalibratedCWDB,
    ecdf_moderator,
    grid_weights,
    make_functionals,
    midpoint_levels,
)

# 1. Prepare covariates X (moderator in column 0), binary treatment A, and K=25 quantile curves Q
rng = np.random.default_rng(42)
n, K = 300, 25
u = midpoint_levels(K)
z = norm.ppf(u)

X = rng.normal(size=(n, 4))
X[:, 0] = ecdf_moderator(X[:, 0])  # rank-transform moderator into (-1, 1)
A = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * X[:, 0])), size=n)

loc = 0.3 * X[:, 0] + A * (0.4 + 0.3 * (X[:, 0] > 0)) + rng.normal(0.0, 0.25, size=n)
scale = (1.1 - 0.2 * A) * np.exp(rng.normal(0.0, 0.12, size=n))
Q = np.maximum.accumulate(loc[:, None] + scale[:, None] * z[None, :], axis=1)

# 2. Declare reference benchmark q_star and functionals
q_star = 1.0 + 0.85 * z
weights = grid_weights(K)
functionals = make_functionals(q_star, weights)

# 3. Fit cross-fitted + AIPW-calibrated Wasserstein Causal Forest
model = DRCalibratedCWDB(
    functionals=functionals,
    contrast_candidates=(0.0, 50.0, 500.0),
    n_folds=3,
    n_particles=10,
    n_estimators=100,
    learning_rate=0.12,
    max_depth=4,
    min_samples_leaf=10,
    min_arm_leaf=5,
    random_state=0,
).fit(X, A, Q, weights)

# 4. Inspect doubly robust marginal TATE, IF standard errors, and predicted particle clouds
for name in ("mean", "sd", "skew", "tail", "reference"):
    print(f"{name:<10}: DR TATE = {model.dr_marginal(name):+.3f} (IF SE = {model.dr_if_se(name):.3f})")

particles_treated = model.predict_particles(X[:5], arm=1)  # shape (5, M=10, K=25)
```

### 2. Applied Microdata Pipeline (`AppliedDataset` & `run_wcf`)

```python
from wasserstein_causal_forests import (
    AppliedDataset,
    ecdf_moderator,
    midpoint_levels,
    quantile_matrix,
    run_placebo,
    run_wcf,
    weighted_quantiles,
)

# Aggregate individual outcomes (y_indiv, w_indiv) by unit_id into (n_units, K=25) quantile vectors
u = midpoint_levels(25)
kept_units, Q = quantile_matrix(y_indiv, w_indiv, unit_ids, levels=u)
q_star = weighted_quantiles(y_indiv[control_mask], w_indiv[control_mask], levels=u)

ds = AppliedDataset(
    study="my_study",
    X=X_units,                 # column 0 = ecdf_moderator(raw_moderator)
    A=A_units,                 # binary treatment in {0, 1}
    Q=Q,                       # shape (n_units, 25)
    moderator_raw=raw_moderator,
    q_star=q_star,             # shape (25,)
    feature_names=["moderator_ecdf", "cov1", "cov2"],
).validate()

model, summary = run_wcf(ds, random_state=0)
placebo_runs = run_placebo(ds, seeds=(0, 1, 2))
```

---

## Tutorials

The [`tutorials/`](tutorials/) directory contains both standalone Python scripts (`.py`) and Google Colab-ready Jupyter notebooks (`.ipynb`):

| Tutorial | Script | Colab Notebook | Description |
|---|---|---|---|
| **01. Synthetic Quickstart** | [`01_quickstart_synthetic.py`](tutorials/01_quickstart_synthetic.py) | [`01_quickstart_synthetic.ipynb`](tutorials/01_quickstart_synthetic.ipynb) | End-to-end simulation, fitting `DRCalibratedCWDB`, evaluating marginal TATE & moderator-quartile TCATE with IF SEs, predicting particle laws, and plotting results. |
| **02. Applied Workflow & Project STAR** | [`02_applied_workflow_and_project_star.py`](tutorials/02_applied_workflow_and_project_star.py) | [`02_applied_workflow_and_project_star.ipynb`](tutorials/02_applied_workflow_and_project_star.ipynb) | Building an `AppliedDataset` from hierarchical individual microdata (`quantile_matrix`, `weighted_quantiles`, `ecdf_moderator`), running `run_wcf` & `run_placebo`, and reproducing the Project STAR classroom analysis. |
| **03. Two-Part WCF for Structural Zeros** | [`03_two_part_structural_zeros.py`](tutorials/03_two_part_structural_zeros.py) | [`03_two_part_structural_zeros.ipynb`](tutorials/03_two_part_structural_zeros.ipynb) | Fitting `ZIPTAdapter` (`cwdb_zipt`) on zero-inflated distributional outcomes ($Q_i = \mathbf{0}_K$) and evaluating conditional-law and AIPW functional errors. |

---

## Repository Structure

```text
Wasserstein_Causal_Forests/
├── src/wasserstein_causal_forests/   # Core WCF Python package
│   ├── cwdb/                         # Particle booster, arm-shared trees, energy score, cross-fitting, AIPW calibration
│   ├── applied/                      # AppliedDataset interface, quantile_matrix, run_wcf, run_placebo
│   ├── g3/                           # Simulation DGPs (LS0-LS3, S1-S6, Z0-Z3, sensitivity), baselines, evaluation
│   └── pta_bcf/                      # Grid functional target definitions
├── tutorials/                        # Standalone .py scripts and Colab-ready .ipynb tutorials
├── paper/                            # Paper figure & table builders and generated assets
│   ├── build_figures.py              # Builds confirmatory simulation figures & tables
│   ├── build_sensitivity_figures.py  # Builds 50-seed sensitivity figures & tables
│   ├── build_sample_size_sensitivity.py # Builds sample-size sensitivity figures & tables
│   ├── confirmatory_data.py          # Confirmatory data loader & paired ledger builder
│   ├── sensitivity_r50_data.py       # Sensitivity 50-seed data loader
│   ├── sensitivity_r50_tables.py     # Sensitivity LaTeX table generator
│   ├── figures/                      # Generated PDF/PNG figures included in the paper
│   └── tables/                       # Generated LaTeX tables (.tex) and summary CSVs
├── research/                         # Simulation drivers, applied study pipelines, and verification checks
│   ├── run_wcf_confirmatory.py       # 50-seed confirmatory simulation driver (n=1000, 14 DGPs)
│   ├── run_wcf_sensitivity_r50.py    # 50-seed sensitivity simulation driver (K, M, assignment, propensity, nulls)
│   ├── run_wcf_sample_size_sensitivity.py # Sample-size sensitivity driver (n in {125, 250, 500, 1000}, 14 DGPs)
│   ├── applied/                      # Reproducible pipelines for the 3 applied studies & asset builders
│   │   ├── project_star/             # Study 1: Project STAR classroom randomization
│   │   ├── primary_state_minwage/    # Study 2: U.S. state-year minimum wages (CPS MORG)
│   │   ├── nordic_benchmark/         # Eurostat Nordic wage benchmark construction
│   │   ├── egger_kenya/              # Study 3: Kenyan cash transfers (Egger et al., 2022 & KIHBS benchmark)
│   │   ├── build_applied_assets.py   # Audits against EVIDENCE_LEDGER.md and writes paper/tables/applied_*.tex
│   │   └── build_applied_figures.py  # Builds density and forest-plot PDFs in paper/figures/applied_*.pdf
│   ├── baselines/                    # R driver scripts for DRF and Causal-DRF comparators
│   └── checks/                       # Proper-score verification and Colab shard builders/importers
├── results/                          # Frozen audited simulation outputs and applied study artifacts
│   ├── wcf_confirmatory_r50/         # 50-seed confirmatory manifest and merged parquet/CSV outputs
│   ├── wcf_sensitivity_r50/          # 50-seed sensitivity manifest and merged parquet/CSV outputs
│   ├── wcf_sample_size_sensitivity/  # Sample-size sensitivity manifest and merged parquet/CSV outputs
│   ├── merged_phase6/                # Reference n=500 LS and historical 10-seed comparison outputs
│   ├── merged_original_causal_drf/   # Reference 10-seed Causal-DRF outputs
│   ├── merged_original_drf/          # Reference 10-seed DRF outputs
│   └── applied_study_exploration/    # Audited datasets, fitted summaries, and EVIDENCE_LEDGER.md for the 3 applied studies
├── colab/                            # Self-contained Google Colab shard notebooks for all 3 simulation studies
│   ├── wcf_confirmatory_r50_26_shards/
│   ├── wcf_sensitivity_r50_120_shards/
│   └── wcf_sample_size_all_dgps_51_shards/
└── tests/                            # Unit and contract test suite (pytest)
```

---

## Reproducing All Paper Figures and Tables

All frozen simulation Parquets and audited applied-study JSON/NPZ artifacts are stored under `results/`. Running the five builder scripts regenerates every figure (`paper/figures/`) and LaTeX table (`paper/tables/`) in the manuscript:

```bash
# 1. Confirmatory 50-replication simulation figures and tables (LS0-LS3, S1-S6, Z0-Z3)
PYTHONPATH=src:paper:. python paper/build_figures.py

# 2. Sensitivity 50-replication figures and tables (K, M, factorial, assignment, propensity, nulls)
PYTHONPATH=src:paper:. python paper/build_sensitivity_figures.py

# 3. Sample-size sensitivity figures and tables (n in {125, 250, 500, 1000} across all 14 DGPs)
PYTHONPATH=src:paper:. python paper/build_sample_size_sensitivity.py

# 4. Applied-study LaTeX tables (cross-checked against results/applied_study_exploration/EVIDENCE_LEDGER.md)
PYTHONPATH=src:. python research/applied/build_applied_assets.py

# 5. Applied-study mixture density and estimate forest plots
PYTHONPATH=src:. python research/applied/build_applied_figures.py
```

---

## Simulation Studies

### 1. Confirmatory 50-Replication Comparison ($n = 1000$)
- **Design**: 14 DGPs across three families:
  - **Location-scale / income family (`LS0`–`LS3`)**: `LS0` (null effect), `LS1` (heterogeneous location + scale compression), `LS2` (variance/tail reshaping), `LS3` (strong nonlinear confounding).
  - **Structural mechanism family (`S1`–`S6`)**: `S1` (smooth baseline), `S2` (scale/shape heterogeneity), `S3` (normative reference-distance regime), `S4` (within-cell stochastic law variability), `S5` (unseen-functional transfer), `S6` (propensity overlap stress).
  - **Zero-inflated family (`Z0`–`Z3`)**: `Z0` (null with structural zeros), `Z1` (aligned extensive + intensive margin), `Z2` (intensive-margin only), `Z3` (opposing extensive vs. intensive margins).
- **Comparators**: `cwdb_dr` (WCF), `cwdb_zipt` (Two-part WCF on `Z0`–`Z3`), `causal_drf` (Causal-DRF), and `drf` (DRF), evaluated under both plug-in law integration and 3-fold AIPW calibration.
- **Commands**:
  ```bash
  python research/run_wcf_confirmatory.py status
  python research/run_wcf_confirmatory.py run --workers 4
  python research/run_wcf_confirmatory.py merge
  python research/run_wcf_confirmatory.py summarize
  ```

### 2. Sensitivity 50-Replication Analysis
- **Design**: Evaluates robustness to grid resolution $K \in \{5, 25, 49\}$, particle count $M \in \{5, 10, 25\}$, $3 \times 3$ $(K, M)$ factorial interaction, assignment mechanisms (`SYM-RANDOM`, `SYM-LIN`, `SYM-NL`, `SYM-MU`), moderator-confounder alignment (`SYM-ALIGN`, `SYM-IRREL`), propensity learners (`logistic`, `hist_gradient_boosting`, `random_forest`, `oracle`), and null-effect companions.
- **Commands**:
  ```bash
  python research/run_wcf_sensitivity_r50.py status
  python research/run_wcf_sensitivity_r50.py run --workers 4
  python research/run_wcf_sensitivity_r50.py merge
  python research/run_wcf_sensitivity_r50.py summarize
  ```

### 3. Sample-Size Sensitivity ($n \in \{125, 250, 500, 1000\}$)
- **Design**: 10 paired replications across all 14 DGPs (`LS0`–`LS3`, `S1`–`S6`, `Z0`–`Z3`) at $n \in \{125, 250, 500, 1000\}$.
- **Commands**:
  ```bash
  python research/run_wcf_sample_size_sensitivity.py status --suite all_dgps
  python research/run_wcf_sample_size_sensitivity.py run --suite all_dgps --workers 4
  python research/run_wcf_sample_size_sensitivity.py merge --suite all_dgps
  ```

---

## Applied Studies

The three empirical studies in the paper share the standardized `AppliedDataset` pipeline ($K = 25$ midpoint quantiles, $M = 10$ particles, 3-fold cross-fitting over $\lambda \in \{0, 50, 500\}$, and 3-fold AIPW calibration):

1. **Project STAR Classroom Randomization (`research/applied/project_star/`)**
   - **Sample**: $n = 1{,}333$ classroom-grades ($522$ small classes, $811$ regular/aide classes; $24{,}610$ student-grade math test observations across grades K–3).
   - **Moderator**: School free-lunch share quartiles.
   - **Reproduce**:
     ```bash
     PYTHONPATH=src:. python research/applied/project_star/run_star_study.py
     ```

2. **U.S. State-Year Minimum Wages (`research/applied/primary_state_minwage/`)**
   - **Sample**: $n = 867$ state-years (50 states + DC, 2000–2016) constructed from CPS MORG hourly wage microdata (`earnwt`), plus 1980–2016 and 2017–2022 window extensions, alternative treatment definitions (`A2`, `A3`), first-difference specifications, and $[0.05, 0.95]$ propensity-overlap trimming ($n = 760$).
   - **Benchmarks**: Scaled Eurostat Nordic wage distribution (`research/applied/nordic_benchmark/`) and the 2016 U.S. national wage distribution.
   - **Reproduce**:
     ```bash
     PYTHONPATH=src:. python research/applied/primary_state_minwage/03_run_wcf_estimations.py
     PYTHONPATH=src:. python research/applied/primary_state_minwage/06_run_trimmed_estimations.py
     ```

3. **Kenyan Cash Transfers (`research/applied/egger_kenya/`)**
   - **Sample**: $n = 653$ villages ($328$ treated, $325$ control; $8{,}226$ households) from Egger et al. (2022), *Econometrica*.
   - **Outcome & Benchmarks**: Village distribution of per-capita nondurable consumption ($K = 25$, household-weighted), evaluated against both the pooled control-village benchmark and the external KIHBS 2015/16 national consumption distribution reconstructed from the World Bank PIP API (`research/applied/egger_kenya/build_external_benchmark.py`).
   - **Reproduce**:
     ```bash
     PYTHONPATH=src:. python research/applied/egger_kenya/run_study.py
     PYTHONPATH=src:. python research/applied/egger_kenya/build_external_benchmark.py
     ```

---

## Running the Test Suite

Run all unit, geometry, proper-scoring-rule, AIPW calibration, and experiment-contract tests via `pytest`:

```bash
PYTHONPATH=src:. pytest -q
```
