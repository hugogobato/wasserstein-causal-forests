# WCF sensitivity Colab shards

4 self-contained notebooks covering 40 cells of the frozen sensitivity study exactly once, over the requested methods: `cwdb_dr_flex_rf`. The rough reference total is 1.7 hours.

## How to run

1. Upload one `wcf_sensitivity_shard_XX.ipynb` per Colab session.
2. Choose Runtime, then Run all. The first cell pins every numerical library to one thread before NumPy is imported.
3. Wait; the notebook checkpoints `shard_output/wcf_sensitivity_parquet.parquet` after every cell and prints progress. An interrupted session resumes from that parquet when the run cell is executed again.
4. Run the final cell; it writes the single `wcf_sensitivity_shard.zip` and downloads it. Send back exactly that one zip per shard.

Cost estimates are rough (from results/wcf_sensitivity/cost_pilot.json, linear in K and M); Colab is slower than the reference machine, so expect longer wall times. The shards are independent and can run concurrently.

The local tmux run (`python3 research/run_wcf_sensitivity.py run --workers N`) is the primary execution path. These notebooks are the overflow and contingency copy; results from both paths are written in the same parquet shape and merge the same way.

## Shards

| Notebook | Cells | Estimated minutes | Estimated hours | Methods |
|---|---|---|---|---|
| `wcf_sensitivity_shard_00.ipynb` | 10 | 25 | 0.42 | cwdb_dr_flex_rf |
| `wcf_sensitivity_shard_01.ipynb` | 10 | 25 | 0.42 | cwdb_dr_flex_rf |
| `wcf_sensitivity_shard_02.ipynb` | 10 | 25 | 0.42 | cwdb_dr_flex_rf |
| `wcf_sensitivity_shard_03.ipynb` | 10 | 25 | 0.42 | cwdb_dr_flex_rf |

## Notes

The notebooks require only Colab's preinstalled NumPy, SciPy, scikit-learn, pandas, and PyArrow. The setup writes the embedded source to a temporary directory and imports it from there; nothing is installed with pip unless a shard contains R forest methods, whose setup cell installs R and takes fifteen to twenty-five minutes.

Failed cells are preserved with their reason in `failure_rows.jsonl` and are never retried under a different seed. A failed cell is not a resume marker: re-running the run cell retries it and replaces its old rows before the next checkpoint.

To merge overflow bundles locally, unzip each bundle into `results/wcf_sensitivity/shards/`, rename `wcf_sensitivity_parquet.parquet` to `shard_colab_<index>.parquet` and `wcf_sensitivity_parquet.meta.json` to `shard_colab_<index>.meta.json` (the `colab_` infix avoids overwriting the local runner's numeric `shard_000.parquet` files), then run `python3 research/run_wcf_sensitivity.py merge`.

