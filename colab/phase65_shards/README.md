# Phase 6.5 Colab shards

17 notebooks covering all 900 cells of the frozen manifest `G3-PHASE65-v1` exactly once. Every notebook clones `https://github.com/hugogobato/wasserstein-causal-forests.git` at the generating commit `bfe99cb3f305` and verifies the frozen manifest checksum before running anything.

## How to run

Upload each notebook to Colab and run all cells. They are independent: run as many concurrently as your accounts allow, in any order. Each notebook ends by downloading a `.zip`.

Then, in the repository:

```bash
# unzip every downloaded bundle into the phase directory
for z in ~/Downloads/p65_shard_*.zip; do unzip -o "$z" -d results/; done

python research/run_phase65.py merge
```

The merge reconciles every cell key against the manifest and fails loudly on a duplicate, an unknown key, or a missing cell.

## Shards

| Notebook | Group | Cells | Reference minutes | Size |
|---|---|---|---|---|
| `p65_shard_00_core65.ipynb` | core65 | 30 | 84 | 15 kB |
| `p65_shard_01_core65.ipynb` | core65 | 30 | 84 | 15 kB |
| `p65_shard_02_core65.ipynb` | core65 | 30 | 84 | 14 kB |
| `p65_shard_03_core65.ipynb` | core65 | 30 | 84 | 14 kB |
| `p65_shard_04_core65.ipynb` | core65 | 30 | 84 | 14 kB |
| `p65_shard_05_core65.ipynb` | core65 | 30 | 84 | 14 kB |
| `p65_shard_06_core65.ipynb` | core65 | 30 | 85 | 14 kB |
| `p65_shard_07_core65.ipynb` | core65 | 30 | 85 | 14 kB |
| `p65_shard_08_core65.ipynb` | core65 | 29 | 83 | 14 kB |
| `p65_shard_09_core65.ipynb` | core65 | 29 | 83 | 14 kB |
| `p65_shard_10_core65.ipynb` | core65 | 31 | 84 | 15 kB |
| `p65_shard_11_core65.ipynb` | core65 | 31 | 84 | 15 kB |
| `p65_shard_12_forest65.ipynb` | forest65 | 108 | 87 | 35 kB |
| `p65_shard_13_forest65.ipynb` | forest65 | 108 | 87 | 35 kB |
| `p65_shard_14_forest65.ipynb` | forest65 | 108 | 87 | 35 kB |
| `p65_shard_15_forest65.ipynb` | forest65 | 108 | 87 | 35 kB |
| `p65_shard_16_forest65.ipynb` | forest65 | 108 | 87 | 35 kB |

Reference minutes are single-threaded estimates on the reference machine, from the audited Phase 6 cost medians. Colab cores are slower; allow two to three times that, plus install time for the `forest65` group.

## Notes

Every notebook pins all numerical libraries to one thread in its first cell, before any import. OpenMP sizes its pool at initialisation, so pinning afterwards does nothing.

Shards containing `causal_drf_retn` cells run the preregistered bandwidth-selection pilot first (pilot seeds 100 and 101, held-out energy score, candidate grid {0.25, 0.5, 1, 2, 4}) and freeze the multipliers document before any decisive cell.

Failed cells are preserved with their reason and are never retried under a different seed. If a shard fails wholesale, for example because an R package would not build, re-run that notebook; do not substitute cells from another shard.
