# Colab execution guide

## Second pilot (current)

Upload `shards_v2/wp9b_shard_00.ipynb` through the last shard, one file per
Colab session, and choose **Runtime → Run all**. The shard index is fixed in
each notebook. Each session clones the repository, installs dependencies, runs
its cells, and downloads `wp9b_shard_XX.json` automatically. If Drive mounts,
the checkpoint is written there, so a disconnected session resumes on rerun
instead of starting over.

Regenerate the shard notebooks after any change to the pilot configuration:

```bash
python3 colab/make_shard_notebooks.py --num-shards 30 --workers 2
```

After downloading every shard file, place them in one directory and run from
the repository root:

```bash
python3 research/sim/merge_results.py wp9b_shard_*.json --out wp9b_merged.json
python3 research/sim/g2_checks.py wp9b_merged.json
```

`merge_results.py` refuses to merge files from different evaluation contracts,
so a stray first-pilot `wp9_shard_*.json` in the glob is an error rather than a
silent contamination.

### What the second pilot changes and why

1. **D0–D5 now run under `feasible_growing_inner`, not `oracle_latent`.** In
   the first pilot those cells handed every method the exact latent quantile
   functions together with oracle nuisances, which makes the AIPW score
   noiseless. Measured errors were 1e-7 to 1e-16, with `rmse_functional_0` on
   D1 equal to 1.1e-16 for six methods, i.e. exact to machine precision. Those
   cells ranked floating-point behavior rather than statistical performance and
   could not support a promise verdict in either direction. D8 was the only
   cell in the first pilot with genuine sampling noise.
2. **D8 keeps an `oracle_latent` reference cell.** Gate G2 criterion 4 compares
   the feasible estimate against the oracle one. The first pilot never ran that
   cell, so `g2_checks.py` correctly returned `null` and the inner-sampling
   criterion was undecidable.
3. **Three prior-art incumbents are now in the tournament**: `causal_drf_port`,
   `focal_dr_meta_learner`, and `wasserstein_random_forest`. The first pilot
   compared ODCF only against internal ablations and homebrew comparators, so
   the closest published competitors named in WP9.2 and in the conditional G0
   decision had never been run. All three are ports, not the authors' code, and
   must be reported as such; see the provenance note at the top of
   `research/sim/incumbents.py`.
4. **`worst_standardized_error` uses a frozen standardizer.** It previously
   divided by the empirical standard deviation of the truth across units, which
   collapses to ~0 on D4 where most coordinates are constant, was floored at
   1e-8, and inflated the metric to 1e6–1e7. It is the declared primary metric
   for D5 and D8, so it could not stay realization-dependent. The frozen scales
   are in `sim.config.frozen_coordinate_scales`.
5. **`specialized_forest` stays in as the primary adversary.** It is separate
   per-block ODCF forests, i.e. the shared-partition ablation, and it matched or
   beat `odcf_composite` in essentially every first-pilot cell. The
   shared-partition claim stands or falls against it.

Because of item 4 the evaluation manifest is tagged `eval-v3-...-fixedscale`.
First-pilot (`eval-v2`) rows are not comparable on that metric.

### Cost

Cross-fitted nuisances are now computed once per cell and reused by every
method, instead of being refitted by each of a dozen methods. The MMD split
rule is quadratic in the node size, so `odcf_mmd_score` and `causal_drf_port`
dominate the cost at n=1000.

For a first Colab test, set `N_SEEDS=1`, `N_TREES=20`, and use D0, D1, and D8
only. Restore `N_SEEDS=30`, `N_TREES=200`, `N_REGIONS=(500, 1000)`, and all
DGPs before collecting evidence.

## First pilot (superseded)

`shards/wp9_shard_00.ipynb` through `shards/wp9_shard_26.ipynb` reproduce the
first pilot. They are kept for provenance. Their results are in the `eval-v2`
contract and fail Gate G2 on every decidable criterion.

## Reusable route

Run `00_setup.ipynb` once per session and then `01_run_shard.ipynb`, setting
`SHARD_INDEX` per session. Every session must use its own output file. The
notebooks are thin wrappers around `research/sim/runner.py`; the source of
truth remains the Python simulation package. The runner checkpoints after every
completed cell, and a rerun with the same output path and `--resume` skips
cells already present in that file.
