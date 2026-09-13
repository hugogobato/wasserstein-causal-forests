# WCF sensitivity Colab shards

48 self-contained notebooks covering 400 cells of the frozen sensitivity study exactly once, over the requested methods: `cwdb_dr`, `cwdb_dr_flex`, `cwdb_dr_oracle`. The rough reference total is 54.9 hours.

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
| `wcf_sensitivity_shard_00.ipynb` | 10 | 70 | 1.17 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_01.ipynb` | 10 | 70 | 1.17 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_02.ipynb` | 10 | 70 | 1.17 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_03.ipynb` | 10 | 70 | 1.17 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_04.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_05.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_06.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_07.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_08.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_09.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_10.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_11.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_12.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_13.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_14.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_15.ipynb` | 8 | 67 | 1.12 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_16.ipynb` | 7 | 69 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_17.ipynb` | 7 | 69 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_18.ipynb` | 7 | 69 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_19.ipynb` | 7 | 69 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_20.ipynb` | 7 | 70 | 1.16 | cwdb_dr |
| `wcf_sensitivity_shard_21.ipynb` | 7 | 70 | 1.16 | cwdb_dr |
| `wcf_sensitivity_shard_22.ipynb` | 7 | 70 | 1.16 | cwdb_dr |
| `wcf_sensitivity_shard_23.ipynb` | 7 | 70 | 1.16 | cwdb_dr |
| `wcf_sensitivity_shard_24.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_25.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_26.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_27.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_28.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_29.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_30.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_31.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_32.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_33.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_34.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_35.ipynb` | 8 | 69 | 1.16 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_36.ipynb` | 10 | 70 | 1.17 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_37.ipynb` | 10 | 70 | 1.17 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_38.ipynb` | 10 | 70 | 1.17 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_39.ipynb` | 10 | 70 | 1.17 | cwdb_dr, cwdb_dr_flex, cwdb_dr_oracle |
| `wcf_sensitivity_shard_40.ipynb` | 9 | 68 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_41.ipynb` | 9 | 68 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_42.ipynb` | 9 | 68 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_43.ipynb` | 9 | 68 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_44.ipynb` | 9 | 68 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_45.ipynb` | 9 | 68 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_46.ipynb` | 9 | 68 | 1.14 | cwdb_dr |
| `wcf_sensitivity_shard_47.ipynb` | 9 | 68 | 1.14 | cwdb_dr |

## Notes

The notebooks require only Colab's preinstalled NumPy, SciPy, scikit-learn, pandas, and PyArrow. The setup writes the embedded source to a temporary directory and imports it from there; nothing is installed with pip unless a shard contains R forest methods, whose setup cell installs R and takes fifteen to twenty-five minutes.

Failed cells are preserved with their reason in `failure_rows.jsonl` and are never retried under a different seed. A failed cell is not a resume marker: re-running the run cell retries it and replaces its old rows before the next checkpoint.

To merge overflow bundles locally, unzip each bundle into `results/wcf_sensitivity/shards/`, rename `wcf_sensitivity_parquet.parquet` to `shard_colab_<index>.parquet` and `wcf_sensitivity_parquet.meta.json` to `shard_colab_<index>.meta.json` (the `colab_` infix avoids overwriting the local runner's numeric `shard_000.parquet` files), then run `python3 research/run_wcf_sensitivity.py merge`.

