# WCF sample-size sensitivity

This experiment checks how the manuscript's synthetic designs behave when the
training sample is reduced. It reuses the frozen DGPs and estimator registry,
and therefore does not duplicate the numerical implementation. The original
five-design manifest remains the default `main` roster. The complete roster is
written under a separate stage so it cannot overwrite that manifest.

The default `main` manifest contains 200 cells per method family: five paper
designs, four training sizes, and ten paired seeds. The full `all` manifest
contains LS0--LS3 (IC0--IC3), S1--S6 (D0, D2, D5--D8), and Z0--Z3
(ZI0--ZI3), for 1680 cells in total. LS and S use `cwdb_dr` (ordinary WCF),
`causal_drf`, and `drf`. Z uses `cwdb_zipt` (the two-part WCF), `causal_drf`,
and `drf`; ordinary WCF is rejected for every Z cell. The fixed representation
is `K=25`, `M=10`; the test sample has 1000 units and uses the same test seed
for every method at a given design and replication.

The default sample-size grid is `n in {125, 250, 500, 1000}`. The two larger
values overlap the existing study, so the comparison is anchored to its
reported settings. The two smaller values probe treated and control arms with
roughly 60 and 125 observations under balanced assignment. Ten seeds are
used, matching the current simulation study. The raw rows preserve every
metric emitted by the evaluation contract; `summarize` reports the mean,
Monte Carlo standard deviation, and Monte Carlo standard error separately for
each metric and target.

## Local execution

From the repository root:

```bash
rtk python3 research/run_wcf_sample_size_sensitivity.py freeze
rtk python3 research/run_wcf_sample_size_sensitivity.py status
rtk python3 research/run_wcf_sample_size_sensitivity.py run --workers 4
rtk python3 research/run_wcf_sample_size_sensitivity.py merge
rtk python3 research/run_wcf_sample_size_sensitivity.py summarize
```

The default freeze declares 600 cells. `--workers 4` is intentionally
conservative for the 20-thread workstation because each worker is pinned to a
single numerical thread. Increase it only after checking memory and the
status of other experiments. A smaller smoke run can be executed in a separate
stage directory, for example:

```bash
rtk python3 research/run_wcf_sample_size_sensitivity.py --stage smoke freeze \
  --n-values 125 --seeds 0,1,2
```

The runner refuses to overwrite an existing manifest once shard or merged
result files are present. To run only a subset of a frozen manifest, use
filters such as `--dgps LS1,S4`, `--n-values 125,250`, or `--methods cwdb_dr`.

To freeze the complete manuscript roster in its own versioned results
directory, use:

```bash
rtk python3 research/run_wcf_sample_size_sensitivity.py \
  --stage all_dgps_v1 freeze --roster all
rtk python3 research/run_wcf_sample_size_sensitivity.py \
  --stage all_dgps_v1 run --workers 4
rtk python3 research/run_wcf_sample_size_sensitivity.py \
  --stage all_dgps_v1 merge
rtk python3 research/run_wcf_sample_size_sensitivity.py \
  --stage all_dgps_v1 summarize
```

## Colab overflow

Use `research/checks/wcf_sample_size_make_colab_notebooks.py` to generate
self-contained notebooks after freezing the manifest. For the complete
roster, the exact 51-shard command is:

```bash
rtk python3 research/checks/wcf_sample_size_make_colab_notebooks.py \
  --manifest results/wcf_sample_size_sensitivity/all_dgps_v1/manifest.json \
  --output-dir colab/wcf_sample_size_sensitivity_all_dgps_v1_shards \
  --shards 51
```

The generator embeds the exact source tree and manifest slice in each
notebook. Each notebook checkpoints after every cell and downloads exactly one
`wcf_sample_size_shard.zip`, containing the parquet, sidecar, manifest slice,
logs, and dependency information. The generated README explains how to rename
the parquet and sidecar into `results/wcf_sample_size_sensitivity/shards/` and
merge them with the local runner. The generator accepts up to 64 shards, so
the 51-shard all-DGP layout remains within the available Colab account layout.

Do not delete or gitignore raw parquet, JSONL failure logs, manifests, or
merged summaries. Failed cells remain visible and are retried with the same
seed and configuration.
