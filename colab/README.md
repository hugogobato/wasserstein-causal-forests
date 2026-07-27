# Colab execution guide

Run `00_setup.ipynb` once per Colab session. Then open `01_run_shard.ipynb` in 27 sessions, setting `SHARD_INDEX` to `0, 1, ..., 26`. Every session must use its own output file, for example `shard_00.json` through `shard_26.json` in Google Drive. Finally run `02_merge_and_analyze.ipynb` once with the 27 files.

The notebooks are intentionally thin wrappers around `research/sim/runner.py`; the source of truth remains the Python simulation package. The runner has a checkpoint after every completed cell, and a rerun with the same output path and `--resume` skips cells already present in that file.

For a first Colab test, set `N_SEEDS=1`, `N_TREES=20`, and use D0, D1, and D8 only. Restore the WP9 values before collecting evidence: `N_SEEDS=30`, `N_TREES=200`, `N_REGIONS=(500, 1000)`, and all DGPs.
