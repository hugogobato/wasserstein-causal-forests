# Google Colab Sharded Simulation Notebooks

This directory contains the self-contained, content-addressed Google Colab shard notebooks used to execute the three simulation studies reported in the paper:

1. **`wcf_confirmatory_r50_26_shards/` (26 notebooks)**
   - **Experiment**: 50-replication confirmatory comparison at $n = 1000$, $K = 25$, $M = 10$ across all 14 paper DGPs:
     - Location-scale/income family (`LS0`–`LS3`),
     - Structural family (`S1`–`S6`),
     - Zero-inflated family (`Z0`–`Z3`).
   - **Methods**: `cwdb_dr` (WCF), `cwdb_zipt` (Two-part WCF on `Z0`–`Z3`), `causal_drf` (Causal-DRF), and `drf` (DRF) under paired plug-in and 3-fold AIPW evaluation.
   - **Driver & Manifest**: `research/run_wcf_confirmatory.py`, `results/wcf_confirmatory_r50/manifest.json`.
   - **Frozen Merged Output**: `results/wcf_confirmatory_r50/merged/wcf_confirmatory_results.parquet`.

2. **`wcf_sensitivity_r50_120_shards/` (120 notebooks)**
   - **Experiment**: 50-replication sensitivity analysis varying:
     - Grid resolution $K \in \{5, 25, 49\}$ and particle count $M \in \{5, 10, 25\}$ (one-factor and full $3 \times 3$ factorial on `LS1` and `LS2`),
     - Assignment mechanisms (`SYM-RANDOM`, `SYM-LIN`, `SYM-NL`, `SYM-MU`),
     - Moderator-confounder alignment (`SYM-ALIGN`, `SYM-IRREL`),
     - Propensity learners (`logistic`, `hist_gradient_boosting`, `random_forest`, `oracle`),
     - Null-effect companions (`SYM-RANDOM-NULL`, `SYM-NL-NULL`, `SYM-IRREL-NULL`).
   - **Driver & Manifest**: `research/run_wcf_sensitivity_r50.py`, `results/wcf_sensitivity_r50/manifest.json`.
   - **Frozen Merged Output**: `results/wcf_sensitivity_r50/merged/wcf_sensitivity_r50_results.parquet`.

3. **`wcf_sample_size_all_dgps_51_shards/` (51 notebooks)**
   - **Experiment**: Sample-size sensitivity across $n \in \{125, 250, 500, 1000\}$ over all 14 DGPs (`LS0`–`LS3`, `S1`–`S6`, `Z0`–`Z3`) with 10 paired replications per cell.
   - **Driver & Manifest**: `research/run_wcf_sample_size_sensitivity.py`, `results/wcf_sample_size_sensitivity/all_dgps/manifest.json`.
   - **Frozen Merged Output**: `results/wcf_sample_size_sensitivity/all_dgps/merged/wcf_sample_size_all_dgps_results.parquet`.
