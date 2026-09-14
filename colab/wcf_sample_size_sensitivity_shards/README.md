# WCF sample-size sensitivity Colab shards

These 16 self-contained notebooks cover 600 cells exactly once. The rough reference total is 20.2 hours (cost pilot).

Upload one notebook per Colab session, choose Run all, and wait for the run cell to finish. Numerical libraries are pinned to one thread. The run cell checkpoints one parquet after every cell; rerunning it resumes successful cells and retries failed cells under the same seed. The final cell downloads exactly one `wcf_sample_size_shard.zip`.

| Notebook | Cells | Estimated minutes | Methods |
|---|---:|---:|---|
| `wcf_sample_size_shard_00.ipynb` | 33 | 75 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_01.ipynb` | 33 | 75 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_02.ipynb` | 36 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_03.ipynb` | 36 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_04.ipynb` | 36 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_05.ipynb` | 36 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_06.ipynb` | 39 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_07.ipynb` | 39 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_08.ipynb` | 39 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_09.ipynb` | 39 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_10.ipynb` | 39 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_11.ipynb` | 39 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_12.ipynb` | 39 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_13.ipynb` | 39 | 76 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_14.ipynb` | 39 | 75 | causal_drf, cwdb_dr, drf |
| `wcf_sample_size_shard_15.ipynb` | 39 | 75 | causal_drf, cwdb_dr, drf |

Each bundle contains `wcf_sample_size_parquet.parquet`, its sidecar, the manifest slice, execution logs, failure rows when present, and dependency metadata. Rename the parquet and sidecar to `shard_colab_<index>.parquet` and `shard_colab_<index>.meta.json` in `results/wcf_sample_size_sensitivity/shards/`, then run:

```bash
rtk python3 research/run_wcf_sample_size_sensitivity.py merge
```

Do not edit the parquet, manifest slice, sidecar, or logs. The local merge checks the contract, source hash, manifest checksum, and cell keys. Failed cells remain explicit rather than being silently dropped.
