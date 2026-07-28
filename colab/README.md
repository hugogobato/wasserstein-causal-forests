# Colab execution guide

The easiest route is to upload `shards/wp9_shard_00.ipynb` through `shards/wp9_shard_26.ipynb`, one file per Colab session, and choose **Runtime → Run all**. The shard index is fixed in each notebook. Each session clones the public repository, installs dependencies, runs its 17 or 18 cells, and downloads `wp9_shard_XX.json` automatically.

The reusable route is to run `00_setup.ipynb` once per session and then open `01_run_shard.ipynb` in 27 sessions, setting `SHARD_INDEX` to `0, 1, ..., 26`. Every session must use its own output file. Finally run `02_merge_and_analyze.ipynb` once with the 27 files.

The notebooks are intentionally thin wrappers around `research/sim/runner.py`; the source of truth remains the Python simulation package. The runner has a checkpoint after every completed cell, and a rerun with the same output path and `--resume` skips cells already present in that file.

After downloading all files, place them in one local directory and run from the repository root:

```bash
python3 research/sim/merge_results.py wp9_shard_*.json --out wp9_merged.json
python3 research/sim/g2_checks.py wp9_merged.json
```

For a first Colab test, set `N_SEEDS=1`, `N_TREES=20`, and use D0, D1, and D8 only. Restore the WP9 values before collecting evidence: `N_SEEDS=30`, `N_TREES=200`, `N_REGIONS=(500, 1000)`, and all DGPs.
