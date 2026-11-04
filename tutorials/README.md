# Wasserstein Causal Forests (WCF) Tutorials

This directory provides self-contained Python scripts (`.py`) and Google Colab-ready Jupyter notebooks (`.ipynb`) illustrating how to use **Wasserstein Causal Forests (WCF)** in applied and methodological research.

## Contents

1. **`01_quickstart_synthetic.py` / `01_quickstart_synthetic.ipynb`**
   - **Topic**: End-to-end quickstart on synthetic distributional data.
   - **What it covers**:
     - Representing unit-level outcome distributions as $K$-point quantile curves $Q_i \in \mathcal{Q}_K$ on the uniform midpoint grid $u_k = (k - 0.5)/K$.
     - Declaring distributional functionals (mean, standard deviation, skewness, upper-half mean, $p_{10}$, $p_{90}$, and reference Wasserstein distance $d_{W,K}(q, q_\star^K)$ to an external benchmark law $q_\star^K$).
     - Fitting `DRCalibratedCWDB` ($M$-particle energy-score boosting with arm-shared trees, out-of-fold contrast shrinkage $\lambda^\star \in \{0, 50, 500\}$, and 3-fold AIPW calibration).
     - Extracting doubly robust marginal TATE and moderator-quartile TCATE estimates, influence-function standard errors, and predicted conditional particle clouds $\widehat{P}_{a,M}^K(x)$.

2. **`02_applied_workflow_and_project_star.py` / `02_applied_workflow_and_project_star.ipynb`**
   - **Topic**: Building an `AppliedDataset` from hierarchical microdata & reproducing the Project STAR classroom analysis.
   - **What it covers**:
     - Aggregating individual/household microdata into unit-level quantile matrices via `quantile_matrix` and `weighted_quantiles`.
     - Rank-transforming the primary effect moderator into $(-1, 1)$ with `ecdf_moderator` so that the four bins $(-1, -0.5, 0, 0.5, 1)$ correspond to moderator quartiles.
     - Running `run_wcf` and `run_placebo` (permutation falsification test).
     - Loading and inspecting the audited Project STAR dataset ($n = 1{,}333$ classroom-grades) from Section 4 of the paper.

3. **`03_two_part_structural_zeros.py` / `03_two_part_structural_zeros.ipynb`**
   - **Topic**: Two-part Wasserstein Causal Forests (`ZIPTAdapter`) for outcome distributions with structural zeros ($Q_i = \mathbf{0}_K$).
   - **What it covers**:
     - Combining cross-fitted arm-specific positive-mass classifiers $\widehat{\pi}_a(x)$ with positive-stratum CWDB particle boosting.
     - Assembling two-part mixture laws $\widehat{P}_{a,\text{ZI}}^K(x) = (1 - \widehat{\pi}_a(x))\delta_{\mathbf{0}_K} + \widehat{\pi}_a(x)\widehat{P}_{a,+}(x)$ and computing doubly robust functional and reference-distance treatment effects via $h(Q_i) = h(\mathbf{0}_K) + I_i(h(Q_i) - h(\mathbf{0}_K))$.

## Running Locally

From the repository root:

```bash
pip install -e .
python tutorials/01_quickstart_synthetic.py
python tutorials/02_applied_workflow_and_project_star.py
python tutorials/03_two_part_structural_zeros.py
```

## Running on Google Colab

Each `.ipynb` notebook includes an environment setup cell that installs `wasserstein-causal-forests` and a final cell with a safe `google.colab.files.download` fallback to automatically download generated figures and JSON summaries when executed on Colab.
