# WCF sensitivity r50 Colab shards

120 self-contained notebooks covering all 6800 cells and 3000 paired replications of the frozen sensitivity roster exactly once, each cell evaluated with the confirmatory protocol.

Run one notebook per Colab session with **Runtime -> Run all**. The shard index is fixed in each notebook. Each session installs the pinned stack, runs the numerical work in a fresh child process, verifies the embedded source and the authors' Causal-DRF repository, checkpoints after every cell, validates coverage, writes a hash inventory, and downloads one ZIP file.

With 48 concurrent sessions, upload the first 48 notebooks, then upload the rest as sessions free up. Every shard is independent and stores its own `shard_output/` checkpoint, so an interrupted session resumes when the run cell is executed again.

## Merging

Unzip every bundle and copy each `sensitivity_r50_results.parquet` into `results/wcf_sensitivity_r50/shards/` as `shard_colab_<NN>.parquet` (the `colab_` infix keeps overflow shards from overwriting local numeric shards), then run:

```bash
python3 research/run_wcf_sensitivity_r50.py merge
python3 research/run_wcf_sensitivity_r50.py summarize
```

| Notebook | Cells | Paired replications | Conservative hours |
|---|---:|---:|---:|
| `wcf_sensitivity_r50_shard_00.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_01.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_02.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_03.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_04.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_05.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_06.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_07.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_08.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_09.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_10.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_11.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_12.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_13.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_14.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_15.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_16.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_17.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_18.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_19.ipynb` | 53 | 24 | 6.96 |
| `wcf_sensitivity_r50_shard_20.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_21.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_22.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_23.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_24.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_25.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_26.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_27.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_28.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_29.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_30.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_31.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_32.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_33.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_34.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_35.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_36.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_37.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_38.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_39.ipynb` | 54 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_40.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_41.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_42.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_43.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_44.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_45.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_46.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_47.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_48.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_49.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_50.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_51.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_52.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_53.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_54.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_55.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_56.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_57.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_58.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_59.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_60.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_61.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_62.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_63.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_64.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_65.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_66.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_67.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_68.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_69.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_70.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_71.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_72.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_73.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_74.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_75.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_76.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_77.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_78.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_79.ipynb` | 60 | 26 | 6.99 |
| `wcf_sensitivity_r50_shard_80.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_81.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_82.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_83.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_84.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_85.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_86.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_87.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_88.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_89.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_90.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_91.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_92.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_93.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_94.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_95.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_96.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_97.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_98.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_99.ipynb` | 58 | 25 | 6.99 |
| `wcf_sensitivity_r50_shard_100.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_101.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_102.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_103.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_104.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_105.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_106.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_107.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_108.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_109.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_110.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_111.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_112.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_113.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_114.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_115.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_116.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_117.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_118.ipynb` | 55 | 24 | 6.93 |
| `wcf_sensitivity_r50_shard_119.ipynb` | 55 | 24 | 6.93 |
