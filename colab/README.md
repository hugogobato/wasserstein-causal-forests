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
python3 colab/make_shard_notebooks.py --num-shards 27 --workers 2
```

With 27 shards each notebook runs exactly 20 of the 540 cells. The split by
sample size is 10/10 for most shards and 9/11 or 11/9 for the rest, because 27
does not divide the interleaved task list evenly. The resulting spread in shard
runtime is about ten minutes, so there is no straggler worth planning around and
all 27 can be launched at once.

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

Measured on an idle 13th-gen i9 with all 14 methods and 200 trees, one
simulation cell costs 237 s at n=500 and 573 s at n=1000. A 20-cell shard is
therefore 2.2 to 2.3 h serial. Colab CPU runtimes are roughly 1.75x slower per
core and the notebooks use `WORKERS=2` against 2 vCPUs, so **expect about 2.0 h
per notebook**, and about 53 h of total wall time across 27 parallel sessions.
That sits well under the eight-hour shard cap in WP9.5.

The three most expensive methods are `causal_drf_port` (57 s / 150 s),
`odcf_mmd_score` (53 s / 113 s), and `drf_inspired_arm_mmd` (39 s / 117 s);
together they are roughly two thirds of the cost. `specialized_forest` is 32 s /
71 s because it fits one forest per coordinate block.

Two repairs made this grid runnable at all. Cross-fitted nuisances are now
computed once per cell and reused by every method, instead of being refitted by
each of a dozen methods. And the MMD split rule rebuilds a full node-sized
kernel for every candidate threshold, which makes tree growth quadratic in node
size; candidate thresholds are now scored on a fixed per-node subsample of at
most `DEFAULT_MMD_MAX_NODE_SAMPLE` points.

To re-measure on a different machine, run `research/sim/second_pilot_checks.py`
first, then time a single cell with `sim.runner.run_simulation_cell`. Measure on
an idle machine: contention from another job silently inflates every number.

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
