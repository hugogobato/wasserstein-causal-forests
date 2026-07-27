# Wasserstein Distributional Causal Forests

This repository contains the finite-grid ODCF prototype, its simulation DGPs, prespecified baselines, and executable WP9 validation gates for the Wasserstein distributional causal-forest project.

The simulation target is deliberately finite-dimensional: `K=49` quantile coordinates and `J=3` nonlinear functional coordinates (Gini, Theil, and Atkinson). The primary simulation package is WP9 from `Wasserstein_Distributional_Causal_Forests_Theory_Plan.md`.

## Simulation studies

The runnable studies are:

| Study | Purpose | Default regime |
|---|---|---|
| D0 | Null treatment effect and bias calibration | `oracle_latent` |
| D1 | Smooth heterogeneous quantile effect | `oracle_latent` |
| D2 | Covariate-dependent scale or shape heterogeneity | `oracle_latent` |
| D3 | Tail and nonlinear-quantile stress test | `oracle_latent` |
| D4 | Nonlinear-functional counterexample and Gini recovery | `oracle_latent` |
| D5 | High-dimensional or weak-signal stress test | `oracle_latent` |
| D8 | Inner-sampling error and the bootstrap-correction comparison | `feasible_growing_inner` and `empirical_proxy` |

The default WP9 pilot is 30 seeds at `n=500` and `n=1000`, with all 11 registered methods. The methods include ODCF composite, curve-only, MMD-score, bootstrap, specialized and multi-output forests, pointwise and scalar causal forests, the two-arm Fréchet direct-sum forest, a global doubly robust estimator, and an arm-specific MMD comparator. The latter two DRF-labelled comparators are explicitly provisional implementations, not claims of official Causal-DRF reproduction.

Run the construction gate first:

```bash
python3 research/sim/g1_checks.py
```

A small local smoke test is:

```bash
python3 research/sim/runner.py \
  --dgps D0 D1 D4 D8 --n 80 --seeds 2 --n_trees 20 \
  --workers 2 --out /tmp/wp9_smoke.json --claim WP9-smoke
```

Run the complete local study with checkpointing as follows. `--resume` can be added after an interruption.

```bash
python3 research/sim/runner.py \
  --dgps D0 D1 D2 D3 D4 D5 D8 \
  --n 500 1000 --seeds 30 --n_trees 200 \
  --workers 4 --out outputs/wp9_pilot.json --claim WP9-T3

python3 research/sim/g2_checks.py outputs/wp9_pilot.json
```

The runner parallelizes complete simulation cells with processes. Each inner scikit-learn model uses one thread, so `--workers` is the main parallelism control. On a 10-physical-core, approximately 15 GiB RAM machine, start at 4 workers and increase only when the machine is otherwise idle. Ten workers is a ceiling, not a default.

## Colab workflow

The `colab/` directory contains three notebooks:

1. `00_setup.ipynb` installs dependencies and checks the checkout.
2. `01_run_shard.ipynb` runs one deterministic shard of the WP9 task list.
3. `02_merge_and_analyze.ipynb` merges shard JSON files, runs G2, and produces compact summary tables and plots.

The shard notebook accepts `SHARD_INDEX` in `[0, 26]` and uses `NUM_SHARDS=27`. Open the same notebook in 27 Colab sessions, set a different shard index in each session, and write each result to a different Google Drive path or download it immediately. The shard assignment is based on a deterministic global task list, so every cell belongs to exactly one shard. Do not let two sessions write to the same output JSON file.

The notebooks use the following repository URL by default:

```text
https://github.com/hugogobato/wasserstein-causal-forests.git
```

Change it in the setup notebook if the repository is renamed or transferred.

## Result format and research guardrails

Results are long-format JSON rows following `research/simulation_results_schema.md`. Every row carries the DGP, observation regime, evaluation-manifest identifier, sample size, seed, method, metric, and value. `empirical_proxy` rows are compared with a deterministic high-inner-sample proxy truth and must not be interpreted as latent-truth performance.

The G2 checker refuses to issue a promise verdict until the required challenge cells have at least 30 seeds. A failed G2 gate is evidence for repair, pivot, or abandonment of a mechanism, not a reason to relabel a metric or change the prespecified comparison after seeing results.
