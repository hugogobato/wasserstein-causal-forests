"""Regenerate the standalone Colab shard notebooks for the WP9 second pilot.

Run from the repository root:

    python3 colab/make_shard_notebooks.py --num-shards 30

The shard notebooks are generated rather than hand-edited so that the pilot
configuration (regime grid, method list, seed count, worker count) cannot drift
between sessions.  Changing the configuration means regenerating every shard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NOTEBOOK_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}

HEADER = """# WP9 second pilot, shard {index:02d} of {total}

Upload this standalone notebook to Colab and choose **Runtime -> Run all**. Its
shard index is fixed, so no cell editing is required. The run checkpoints after
every completed cell and downloads `wp9b_shard_{index:02d}.json` when it finishes.

**What changed from the first pilot.** D0-D5 now run under
`feasible_growing_inner`, so each region is observed through a finite inner
sample and nuisances are cross-fitted. The first pilot ran them under
`oracle_latent`, where the scores are effectively noiseless and every method's
error collapsed to 1e-7..1e-16. D8 keeps an `oracle_latent` reference cell,
without which the feasible-oracle gap in Gate G2 criterion 4 is undefined.
Three prior-art incumbents (Causal-DRF, FOCaL-style, Wasserstein Random Forest)
have been added, and `worst_standardized_error` now uses a frozen standardizer.

**Checkpointing.** If Google Drive is mounted the checkpoint is written there,
so a disconnected session resumes instead of restarting. Re-running the
notebook with the same Drive file skips the cells already completed.
"""

SETUP = '''from pathlib import Path
import json
import os
import subprocess
import sys

SHARD_INDEX = {index}
NUM_SHARDS = {total}
REPO_URL = os.environ.get("WDCF_REPO_URL", "https://github.com/hugogobato/wasserstein-causal-forests.git")
REPO_DIR = Path("/content/wasserstein-causal-forests")
DGPS = {dgps}
N_REGIONS = {n_regions}
N_SEEDS = {n_seeds}
N_TREES = {n_trees}
N_EVAL = {n_eval}
WORKERS = {workers}
CLAIM_ID = "{claim_id}"
OUTPUT_NAME = f"wp9b_shard_{{SHARD_INDEX:02d}}.json"

# Prefer a Drive checkpoint so a disconnect resumes instead of restarting.
CHECKPOINT_DIR = Path("/content")
try:
    from google.colab import drive

    drive.mount("/content/drive")
    CHECKPOINT_DIR = Path("/content/drive/MyDrive/wdcf_wp9b")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("Checkpointing to Drive:", CHECKPOINT_DIR)
except Exception as exc:
    print("(No Drive; checkpointing to local /content):", exc)

OUTPUT_PATH = CHECKPOINT_DIR / OUTPUT_NAME

if not (REPO_DIR / "research/sim/runner.py").exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-r",
    str(REPO_DIR / "requirements-colab.txt"),
], check=True)
sys.path.insert(0, str(REPO_DIR / "research"))
from sim.runner import build_simulation_tasks

tasks = build_simulation_tasks(
    dgp_names=DGPS, n_regions_list=N_REGIONS, n_seeds=N_SEEDS,
    n_trees=N_TREES, n_eval=N_EVAL, claim_id=CLAIM_ID,
    shard_index=SHARD_INDEX, num_shards=NUM_SHARDS,
)
print(f"Running shard {{SHARD_INDEX:02d}}/{{NUM_SHARDS}}: {{len(tasks)}} cells")
for task in tasks:
    print(f"  {{task[0]}} n={{task[1]}} {{task[2]}} seed={{task[3]}}")
print(f"Output: {{OUTPUT_PATH}}")

command = [
    sys.executable, str(REPO_DIR / "research/sim/runner.py"),
    "--dgps", *DGPS,
    "--n", *(str(n) for n in N_REGIONS),
    "--seeds", str(N_SEEDS),
    "--n_trees", str(N_TREES),
    "--n_eval", str(N_EVAL),
    "--workers", str(WORKERS),
    "--shard-index", str(SHARD_INDEX),
    "--num-shards", str(NUM_SHARDS),
    "--claim", CLAIM_ID,
    "--out", str(OUTPUT_PATH),
    "--resume",
]
subprocess.run(command, cwd=REPO_DIR, check=True)

rows = json.loads(OUTPUT_PATH.read_text())
cells = {{(r["dgp_id"], r["n_regions"], r["observation_regime"], r["seed"]) for r in rows}}
methods = sorted({{r["method"] for r in rows}})
manifests = sorted({{r["evaluation_manifest_id"].split("-")[1] for r in rows}})
print(f"Completed {{len(cells)}} cells, {{len(rows)}} rows, {{len(methods)}} methods")
print("Evaluation manifest versions:", manifests)
assert manifests == ["v3"], "this shard mixed evaluation contracts; do not merge it"

local_copy = Path("/content") / OUTPUT_NAME
if local_copy != OUTPUT_PATH:
    local_copy.write_text(OUTPUT_PATH.read_text())

try:
    from google.colab import files
    files.download(str(local_copy))
    print("Downloaded:", local_copy)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''


def build_notebook(index: int, total: int, params: dict) -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": HEADER.format(index=index, total=total).splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": SETUP.format(index=index, total=total, **params).splitlines(keepends=True),
            },
        ],
        "metadata": NOTEBOOK_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--claim-id", default="WP9-T3-colab-v2")
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parent / "shards_v2")
    )
    args = parser.parse_args()

    sys_path = Path(__file__).resolve().parent.parent / "research"
    import sys

    sys.path.insert(0, str(sys_path))
    from sim.runner import MAX_SHARDS, build_simulation_tasks

    if not 1 <= args.num_shards <= MAX_SHARDS:
        raise SystemExit(f"num-shards must lie in [1, {MAX_SHARDS}]")

    params = {
        "dgps": ("D0", "D1", "D2", "D3", "D4", "D5", "D8"),
        "n_regions": (500, 1000),
        "n_seeds": args.n_seeds,
        "n_trees": args.n_trees,
        "n_eval": args.n_eval,
        "workers": args.workers,
        "claim_id": args.claim_id,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("wp9b_shard_*.ipynb"):
        stale.unlink()

    total_cells = 0
    for index in range(args.num_shards):
        notebook = build_notebook(index, args.num_shards, params)
        path = out_dir / f"wp9b_shard_{index:02d}.ipynb"
        path.write_text(json.dumps(notebook, indent=1) + "\n")
        cells = build_simulation_tasks(
            dgp_names=params["dgps"], n_regions_list=params["n_regions"],
            n_seeds=args.n_seeds, n_trees=args.n_trees, n_eval=args.n_eval,
            claim_id=args.claim_id, shard_index=index, num_shards=args.num_shards,
        )
        total_cells += len(cells)
        print(f"wrote {path.name}: {len(cells)} cells")
    print(f"{args.num_shards} shards, {total_cells} cells, workers={args.workers}")


if __name__ == "__main__":
    main()
