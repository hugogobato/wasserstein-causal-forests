# G3 tournament Colab shards

27 notebooks covering all 4110 cells of the frozen manifest `G3-MAIN-v1` exactly once.

## How to run

Upload each notebook to Colab and run all cells. They are independent: run as many concurrently as your account allows, in any order. Each notebook ends by downloading a `.zip`.

Then, in the repository:

```bash
# unzip every downloaded bundle into results/
for z in ~/Downloads/g3_shard_*.zip; do unzip -o "$z" -d results/main/; done
mv results/main/execution_log_*.jsonl results/manifests/ 2>/dev/null

python research/run_g3.py merge
python research/checks/g3_report.py
python research/checks/g3_gate_flags.py
python research/checks/g3_write_memo.py
```

The merge reconciles every cell key against the manifest and fails loudly on a duplicate, an unknown key, or a missing cell, so a forgotten shard cannot pass silently as a smaller tournament.

## Shards

| Notebook | Group | Cells | Reference minutes | Size |
|---|---|---|---|---|
| `g3_shard_00_core.ipynb` | core | 276 | 51 | 2.03 MB |
| `g3_shard_01_core.ipynb` | core | 277 | 51 | 2.03 MB |
| `g3_shard_02_core.ipynb` | core | 277 | 51 | 2.03 MB |
| `g3_shard_03_core.ipynb` | core | 277 | 51 | 2.03 MB |
| `g3_shard_04_core.ipynb` | core | 277 | 51 | 2.03 MB |
| `g3_shard_05_core.ipynb` | core | 278 | 51 | 2.03 MB |
| `g3_shard_06_core.ipynb` | core | 278 | 51 | 2.03 MB |
| `g3_shard_07_ptas.ipynb` | ptas | 89 | 49 | 1.99 MB |
| `g3_shard_08_ptas.ipynb` | ptas | 90 | 49 | 1.99 MB |
| `g3_shard_09_ptas.ipynb` | ptas | 89 | 49 | 1.99 MB |
| `g3_shard_10_ptas.ipynb` | ptas | 89 | 49 | 1.99 MB |
| `g3_shard_11_ptas.ipynb` | ptas | 89 | 49 | 1.99 MB |
| `g3_shard_12_ptas.ipynb` | ptas | 92 | 49 | 2.00 MB |
| `g3_shard_13_ptas.ipynb` | ptas | 92 | 49 | 2.00 MB |
| `g3_shard_14_forest.ipynb` | forest | 224 | 48 | 2.02 MB |
| `g3_shard_15_forest.ipynb` | forest | 224 | 48 | 2.02 MB |
| `g3_shard_16_forest.ipynb` | forest | 223 | 48 | 2.02 MB |
| `g3_shard_17_forest.ipynb` | forest | 223 | 48 | 2.02 MB |
| `g3_shard_18_forest.ipynb` | forest | 223 | 48 | 2.02 MB |
| `g3_shard_19_forest.ipynb` | forest | 223 | 48 | 2.02 MB |
| `g3_shard_20_ptaf.ipynb` | ptaf | 29 | 46 | 1.98 MB |
| `g3_shard_21_ptaf.ipynb` | ptaf | 29 | 46 | 1.98 MB |
| `g3_shard_22_ptaf.ipynb` | ptaf | 29 | 46 | 1.98 MB |
| `g3_shard_23_ptaf.ipynb` | ptaf | 29 | 46 | 1.98 MB |
| `g3_shard_24_ptaf.ipynb` | ptaf | 28 | 45 | 1.98 MB |
| `g3_shard_25_ptaf.ipynb` | ptaf | 28 | 45 | 1.98 MB |
| `g3_shard_26_ptaf.ipynb` | ptaf | 28 | 45 | 1.98 MB |

Reference minutes are single-threaded on the machine that ran the cost pilot. Colab cores are slower; allow two to three times that, plus install time for the `forest` and `ptaf` groups.

## Notes

Every notebook pins all numerical libraries to one thread in its first cell, before any import. This is not optional. OpenMP sizes its pool at initialisation, so pinning after `import numpy` does nothing, and an unpinned run on the reference machine lost roughly a factor of forty to oversubscription.

Failed cells are preserved with their reason and are never retried under a different seed. If a shard fails wholesale, for example because an R package would not build, re-run that notebook; do not substitute cells from another shard.
