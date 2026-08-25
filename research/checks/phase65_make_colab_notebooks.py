#!/usr/bin/env python3
"""Generate self-contained Colab notebooks for the Phase 6.5 shards.

Run from the repository root:

    python research/checks/phase65_make_colab_notebooks.py

Writes `colab/phase65_shards/p65_shard_XX_<group>.ipynb`, one per shard, plus
a README. Each notebook is tiny: it clones this repository from GitHub at the
exact commit that generated it, verifies the frozen manifest's checksum, and
runs. Nothing is embedded, so the browser renders kilobytes instead of the
megabytes an embedded-archive notebook costs.

Generation refuses to run unless the working tree is clean apart from this
output directory and the Phase 6.5 manifest is committed: a pinned commit that
does not contain today's code and manifest would produce shards that fail
loudly on Colab rather than quietly run something else.

Two dependency groups:

* `core65`   pure NumPy/SciPy/scikit-learn, preinstalled on Colab;
* `forest65` adds R, the pinned CRAN `drf` 1.3.1, and the authors'
  causal-clean `drf` built from GitHub at the frozen commit, installed into a
  notebook-local library that `WCF_CAUSAL_DRF_R_LIB` selects.

Shards that contain `causal_drf_retn` cells first run the preregistered
bandwidth-selection pilot (seeds 100 and 101, outside every decisive range)
and freeze `results/manifests/phase65_bandwidth_selection.json` before any
decisive cell executes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.phase65 import (  # noqa: E402
    PHASE65_CONTRACT_ID,
    enumerate_phase65_cells,
)

OUTPUT = ROOT / "colab" / "phase65_shards"
MANIFEST = ROOT / "results" / "manifests" / "phase65_manifest.json"

TOTAL_NOTEBOOKS = 17

#: Resolved by `_resolve_pinned_context` before anything is written.
REMOTE_URL: str = ""
PINNED_COMMIT: str = ""
PINNED_MANIFEST_CHECKSUM: str = ""


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise SystemExit(f"git {' '.join(arguments)} failed:\n{completed.stderr}")
    return completed.stdout.strip()


def _resolve_pinned_context() -> None:
    """Pin remote, commit, and manifest checksum, refusing unsafe states."""

    global REMOTE_URL, PINNED_COMMIT, PINNED_MANIFEST_CHECKSUM

    REMOTE_URL = _git("remote", "get-url", "origin")
    PINNED_COMMIT = _git("rev-parse", "HEAD")

    tracked = _git(
        "ls-files", "--error-unmatch", "results/manifests/phase65_manifest.json"
    )
    if not tracked:
        raise SystemExit(
            "results/manifests/phase65_manifest.json is not committed, so a "
            "pinned clone could not reconstruct the frozen grid. Commit and "
            "push, then regenerate."
        )

    dirty = [
        line for line in _git("status", "--porcelain").splitlines()
        if line.strip() and "colab/phase65_shards/" not in line
    ]
    if dirty:
        preview = "\n".join(dirty[:8])
        raise SystemExit(
            "the working tree is dirty outside colab/phase65_shards/, so "
            f"HEAD does not contain what the shards would run:\n{preview}\n"
            "Commit and push, then regenerate the notebooks."
        )

    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    PINNED_MANIFEST_CHECKSUM = document["manifest_checksum"]

#: Method to dependency group.
GROUPS: dict[str, str] = {
    "cwdb_r3_cvridge": "core65",
    "cwdb_dr": "core65",
    "cwdb_zipt": "core65",
    "causal_drf_log": "forest65",
    "drf_log": "forest65",
    "causal_drf_retn": "forest65",
    "causal_drf": "forest65",
    "drf": "forest65",
}

GROUP_ORDER = ("core65", "forest65")

#: Single-threaded local seconds per fit, carried over from the audited Phase 6
#: cost medians where they exist and from linear-in-n interpolation for the
#: scaling grid. Only relative balance matters here.
BASE_COSTS = {
    ("cwdb_r3_cvridge", 500): 85.0,
    ("cwdb_r3_cvridge", 1000): 141.0,
    ("cwdb_r3_cvridge", 2000): 280.0,
    ("cwdb_r3_cvridge", 4000): 560.0,
    ("cwdb_dr", 500): 197.0,
    ("cwdb_dr", 1000): 209.0,
    ("cwdb_zipt", 500): 105.0,
    ("cwdb_zipt", 1000): 130.0,
}
FOREST_METHODS = ("causal_drf", "causal_drf_log", "drf", "drf_log")
for _method in FOREST_METHODS:
    BASE_COSTS[(_method, 500)] = 38.0
    BASE_COSTS[(_method, 1000)] = 40.0
    BASE_COSTS[(_method, 2000)] = 78.0
    BASE_COSTS[(_method, 4000)] = 156.0

CAUSAL_CLEAN_COMMIT = "0a1a508444176b5b1553f13e832be93a374b0af2"

SETUP: dict[str, str] = {
    "core65": (
        "# This group needs only NumPy, SciPy, scikit-learn and PyArrow, all\n"
        "# preinstalled on Colab. Nothing to install.\n"
        "!python -c \"import numpy, scipy, sklearn, pyarrow; "
        "print('numpy', numpy.__version__, '| sklearn', sklearn.__version__)\"\n"
    ),
    "forest65": (
        "# This group runs the R forest baselines, including Causal-DRF\n"
        "# through the authors' causal-clean package at the frozen commit.\n"
        "# The causal-clean repository is a monorepo whose R package sits in\n"
        "# r-package/drf, so the installer fetches the exact-commit tarball\n"
        "# from codeload (no GitHub API, hence no shared-IP rate limit) and\n"
        "# runs R CMD INSTALL on that subdirectory. Expect fifteen to twenty-\n"
        "# five minutes for this cell.\n"
        "%%bash\n"
        "set -e\n"
        "apt-get -qq update > /dev/null 2>&1\n"
        "apt-get -qq install -y r-base r-base-dev libcurl4-openssl-dev "
        "libssl-dev libxml2-dev curl > /dev/null 2>&1\n"
        "Rscript -e 'options(Ncpus=2); "
        "install.packages(c(\"Rcpp\",\"RcppEigen\",\"jsonlite\",\"remotes\","
        "\"transport\",\"fastDummies\",\"kernlab\"), "
        "repos=\"https://cloud.r-project.org\", quiet=TRUE)'\n"
        "# CRAN drf 1.3.1 drives the paper-DRF and W-DRF-T drivers; the\n"
        "# causal-clean library below shadows it only for Causal-DRF cells.\n"
        "Rscript -e 'options(Ncpus=2); "
        "if (!requireNamespace(\"drf\", quietly=TRUE)) "
        "install.packages(\"drf\", repos=\"https://cloud.r-project.org\", "
        "quiet=TRUE); cat(\"CRAN drf\", as.character(packageVersion(\"drf\")), "
        "\"ready\\n\")'\n"
        "mkdir -p results/Rlib/causal_drf\n"
        "CAUSAL_SHA=\"" + CAUSAL_CLEAN_COMMIT + "\"\n"
        "if [ ! -d results/Rlib/causal_drf/drf ]; then\n"
        "  TARBALL=\"/tmp/causal_clean_${CAUSAL_SHA:0:12}.tar.gz\"\n"
        "  curl -sL \"https://codeload.github.com/herbps10/drf/tar.gz/"
        "${CAUSAL_SHA}\" -o \"$TARBALL\"\n"
        "  EXTRACT=\"/tmp/causal_clean_src\"\n"
        "  rm -rf \"$EXTRACT\"; mkdir -p \"$EXTRACT\"\n"
        "  tar -xzf \"$TARBALL\" -C \"$EXTRACT\"\n"
        "  PKG_DIR=$(find \"$EXTRACT\" -maxdepth 3 -type d -path \"*r-package/drf\""
        " | head -1)\n"
        "  echo \"installing causal-clean drf from $PKG_DIR\"\n"
        "  R CMD INSTALL --library=results/Rlib/causal_drf \"$PKG_DIR\" \\\n"
        "    || Rscript -e 'options(Ncpus=2); "
        ".libPaths(c(\"results/Rlib/causal_drf\",.libPaths())); "
        "remotes::install_github(\"herbps10/drf\", ref=\"" + CAUSAL_CLEAN_COMMIT
        + "\", subdir=\"r-package/drf\", "
        "lib=\"results/Rlib/causal_drf\", upgrade=\"never\", quiet=TRUE)'\n"
        "fi\n"
        "Rscript -e '.libPaths(c(\"results/Rlib/causal_drf\",.libPaths())); "
        "stopifnot(requireNamespace(\"drf\", quietly=TRUE)); "
        "cat(\"causal-clean drf\", as.character(packageVersion(\"drf\")), "
        "\"ready\\n\")'\n"
        "echo 'setup complete'\n"
    ),
}

GROUP_NOTE: dict[str, str] = {
    "core65": (
        "This shard runs the pure-Python methods: the cross-fitted R3 "
        "booster, the doubly-robust calibration layer, and the two-part "
        "assembly. There is no install step, so computing starts immediately."
    ),
    "forest65": (
        "This shard runs the R forest baselines, including the two adversarial "
        "controls (log geometry and bandwidth retune). The setup cell installs "
        "R, the pinned `drf` 1.3.1, and the authors' causal-clean package at "
        "the frozen commit; fifteen to twenty-five minutes."
    ),
}


def cell_cost(cell) -> float:
    return BASE_COSTS.get((cell.method, cell.n_train), 60.0)


def allocate() -> list[tuple[str, list, float]]:
    """Split cells into TOTAL_NOTEBOOKS shards, grouped by dependency."""

    grouped: dict[str, list] = defaultdict(list)
    for cell in enumerate_phase65_cells():
        grouped[GROUPS[cell.method]].append(cell)

    group_cost = {
        name: sum(cell_cost(c) for c in cells)
        for name, cells in grouped.items()
    }
    counts = {name: 1 for name in grouped}
    for _ in range(TOTAL_NOTEBOOKS - len(grouped)):
        pressure = {name: group_cost[name] / counts[name] for name in grouped}
        counts[max(pressure, key=pressure.get)] += 1

    shards: list[tuple[str, list, float]] = []
    for name in GROUP_ORDER:
        cells = grouped.get(name, [])
        if not cells:
            continue
        bins: list[list] = [[] for _ in range(counts[name])]
        loads = [0.0] * counts[name]
        for cell in sorted(cells, key=cell_cost, reverse=True):
            index = loads.index(min(loads))
            bins[index].append(cell)
            loads[index] += cell_cost(cell)
        for bin_cells, load in zip(bins, loads):
            shards.append((name, bin_cells, load))
    return shards


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": text,
    }


def build_notebook(index: int, group: str, cells: list, load: float) -> dict:
    shard_name = f"p65_shard_{index:02d}_{group}"
    methods = sorted({c.method for c in cells})
    estimate_minutes = load / 60.0
    has_retune = any(c.method == "causal_drf_retn" for c in cells)

    notebook_cells = [
        markdown(
            f"# Phase 6.5 shard {index:02d} ({group})\n"
            "\n"
            f"Runs **{len(cells)} cells** of the frozen Phase 6.5 manifest "
            f"(`{PHASE65_CONTRACT_ID}`), covering: "
            f"{', '.join('`' + m + '`' for m in methods)}.\n"
            "\n"
            f"{GROUP_NOTE[group]}\n"
            "\n"
            f"Estimated single-threaded compute on the reference machine is "
            f"about **{estimate_minutes:.0f} minutes**. Colab cores are "
            "slower, so allow two to three times that, plus any install time "
            "above. This fits comfortably inside a nine hour session.\n"
            "\n"
            "**Run every cell in order.** The last cell downloads a `.zip`; "
            "collect every shard's zip into `results/phase65/colab_shards/` "
            "(logs into `results/manifests/`) and run "
            "`python research/run_phase65.py merge`.\n"
        ),
        code(
            "# Thread pinning MUST happen before NumPy or SciPy are imported.\n"
            "# OpenMP sizes its pool at initialisation, so setting these\n"
            "# afterwards is silently ineffective.\n"
            "import os\n"
            "for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',\n"
            "           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',\n"
            "           'VECLIB_MAXIMUM_THREADS', 'R_NUM_THREADS'):\n"
            "    os.environ[_v] = '1'\n"
            "print('threads pinned to 1')"
        ),
        markdown("## 1. Clone the repository at the pinned commit\n"
                 "\n"
                 f"Remote `{REMOTE_URL}`, commit `{PINNED_COMMIT[:12]}`. After "
                 "checkout the notebook asserts the frozen manifest checksum, "
                 "so a clone of anything but the generating commit fails here "
                 "rather than mid-run."
                 ),
        code(make_clone_cell(group)),
        markdown("## 2. Dependencies"),
        code(SETUP[group]),
        markdown("## 3. This shard's cells"),
        code(
            "import json, collections\n"
            "SHARD_INDEX = " + str(index) + "\n"
            "CELLS = json.loads('''"
            + json.dumps([c.to_dict() for c in cells]) + "''')\n"
            "print(f'{len(CELLS)} cells in this shard')\n"
            "for key, count in sorted(collections.Counter(\n"
            "        (c['grid'], c['dgp'], c['method'])\n"
            "        for c in CELLS).items()):\n"
            "    print(f'  {key[0]:12s} {key[1]:8s} {key[2]:18s} {count}')"
        ),
    ]
    if has_retune:
        notebook_cells.append(
            markdown(
                "## 4. Bandwidth-selection pilot (preregistered)\n"
                "\n"
                "This shard contains `causal_drf_retn` cells, so it first "
                "runs the selection pilot on seeds 100 and 101, outside every "
                "decisive range, and freezes the multipliers document. The "
                "rule picks the candidate with the best held-out energy score; "
                "oracle truth is never read."
            )
        )
        notebook_cells.append(code(RETUNE_CELL))
        notebook_cells.append(markdown("## 5. Run"))
    else:
        notebook_cells.append(markdown("## 4. Run"))
    notebook_cells.append(
        code(
            "import time\n"
            "from pathlib import Path\n"
            "from wasserstein_causal_forests.g3.manifest import Cell\n"
            "from wasserstein_causal_forests.g3.runner import run_shard\n"
            "\n"
            "cells = [Cell(**{k: v for k, v in item.items()\n"
            "                 if k not in ('cell_key', 'test_seed')})\n"
            "         for item in CELLS]\n"
            "\n"
            "out = Path('/content/wcf/results/phase65/colab_shards')\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "log = Path(f'/content/wcf/results/manifests/"
            "phase65_execution_log_{SHARD_INDEX:03d}.jsonl')\n"
            "log.parent.mkdir(parents=True, exist_ok=True)\n"
            "cache = Path('/content/wcf/results/rcpp_cache')\n"
            "cache.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "started = time.time()\n"
            "summary = run_shard(\n"
            "    cells,\n"
            "    out / f'shard_{SHARD_INDEX:03d}.parquet',\n"
            "    cache_directory=cache,\n"
            "    log_path=log,\n"
            "    manifest_contract_id='" + PHASE65_CONTRACT_ID + "',\n"
            ")\n"
            "print(json.dumps(summary, indent=2))\n"
            "print(f'elapsed {(time.time() - started) / 60:.1f} min')"
        )
    )
    notebook_cells.append(
        markdown(
            "## Check\n"
            "\n"
            "Every cell must appear exactly once, as a success or as a "
            "failure. Failures are kept and reported at merge time; a seed is "
            "never silently replaced."
        )
    )
    notebook_cells.append(
        code(
            "import collections\n"
            "records = [json.loads(line) for line in\n"
            "           open(log, encoding='utf-8') if line.strip()]\n"
            "status = collections.Counter(r['status'] for r in records)\n"
            "print('cells logged:', len(records), '| expected:', len(CELLS))\n"
            "print('status:', dict(status))\n"
            "assert len(records) == len(CELLS), 'shard did not finish every cell'\n"
            "for record in records:\n"
            "    if record['status'] != 'ok':\n"
            "        print('  FAILED', record['dgp'], record['method'],\n"
            "              record['seed'])\n"
            "slowest = sorted(records, key=lambda r: -r['wall_seconds'])[:5]\n"
            "print('slowest cells:', [(r['method'], round(r['wall_seconds'], 1))\n"
            "                         for r in slowest])"
        )
    )
    notebook_cells.append(markdown("## Download the results"))
    notebook_cells.append(
        code(
            "import shutil\n"
            f"bundle = '/content/{shard_name}'\n"
            "staging = Path('/content/bundle')\n"
            "if staging.exists():\n"
            "    shutil.rmtree(staging)\n"
            "staging.mkdir(parents=True)\n"
            "shutil.copy(out / f'shard_{SHARD_INDEX:03d}.parquet', staging)\n"
            "if log.exists():\n"
            "    shutil.copy(log, staging)\n"
            "output_file = shutil.make_archive(bundle, 'zip', staging)\n"
            "print('bundle:', output_file,\n"
            "      f'({os.path.getsize(output_file) / 1e6:.2f} MB)')\n"
            "\n"
            "try:\n"
            "    from google.colab import files\n"
            "    files.download(output_file)\n"
            "    print('Downloaded:', output_file)\n"
            "except Exception as e:\n"
            "    print('(Not on Colab / download skipped):', e)"
        )
    )

    return {
        "cells": [
            {**cell, "source": cell["source"].splitlines(keepends=True)}
            for cell in notebook_cells
        ],
        "metadata": {
            "colab": {"provenance": [], "name": shard_name},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def make_clone_cell(group: str) -> str:
    """Shallow-fetch exactly the pinned commit, verify, and wire the paths."""

    library_line = (
        "os.environ['WCF_CAUSAL_DRF_R_LIB'] = "
        "'/content/wcf/results/Rlib/causal_drf'\n"
        if group == "forest65"
        else ""
    )
    return (
        "import subprocess, pathlib, os, sys, json, hashlib\n"
        "\n"
        f"REPO = {REMOTE_URL!r}\n"
        f"COMMIT = {PINNED_COMMIT!r}\n"
        f"EXPECTED_CHECKSUM = {PINNED_MANIFEST_CHECKSUM!r}\n"
        "\n"
        "workdir = pathlib.Path('/content/wcf')\n"
        "if not workdir.exists():\n"
        "    subprocess.run(['git', 'init', '-q', str(workdir)], check=True)\n"
        "    subprocess.run(\n"
        "        ['git', '-C', str(workdir), 'remote', 'add', 'origin', REPO],\n"
        "        check=True,\n"
        "    )\n"
        "# A shallow fetch of the exact commit: nothing else is downloaded.\n"
        "    subprocess.run(\n"
        "        ['git', '-C', str(workdir), 'fetch', '-q', '--depth', '1',\n"
        "         'origin', COMMIT], check=True,\n"
        "    )\n"
        "    subprocess.run(\n"
        "        ['git', '-C', str(workdir), 'checkout', '-q', 'FETCH_HEAD'],\n"
        "        check=True,\n"
        "    )\n"
        "os.chdir(workdir)\n"
        "sys.path.insert(0, str(workdir / 'src'))\n"
        + library_line +
        "\n"
        "manifest = json.load(open(\n"
        "    'results/manifests/phase65_manifest.json', encoding='utf-8'\n"
        "))\n"
        "checksum = hashlib.sha256(\n"
        "    json.dumps(manifest['cells'], sort_keys=True).encode('utf-8')\n"
        ").hexdigest()\n"
        "assert checksum == EXPECTED_CHECKSUM, (\n"
        "    'the cloned manifest does not match the frozen grid: '\n"
        "    f'{checksum} != {EXPECTED_CHECKSUM}'\n"
        ")\n"
        "print('repo ready at commit ' + COMMIT[:12] + '; '\n"
        "      + str(manifest['n_cells']) + ' frozen cells verified')"
    )


RETUNE_CELL = (
    "from pathlib import Path\n"
    "import json, numpy as np\n"
    "from wasserstein_causal_forests.g3.dgps import build_dgp\n"
    "from wasserstein_causal_forests.g3.phase65_methods import (\n"
    "    BANDWIDTH_CANDIDATES, SELECTION_SEEDS, select_bandwidth_multiplier,\n"
    ")\n"
    "\n"
    "keys = sorted({(c['dgp'], c['n_train']) for c in CELLS\n"
    "               if c['method'] == 'causal_drf_retn'})\n"
    "multipliers = {}\n"
    "cache = Path('/content/wcf/results/rcpp_cache')\n"
    "cache.mkdir(parents=True, exist_ok=True)\n"
    "for dgp_name, n_train in keys:\n"
    "    dgp = build_dgp(dgp_name, 25)\n"
    "    best, means = select_bandwidth_multiplier(\n"
    "        dgp, n_train, seeds=SELECTION_SEEDS,\n"
    "        candidates=BANDWIDTH_CANDIDATES, cache_directory=cache,\n"
    "    )\n"
    "    multipliers[f'{dgp_name}|{n_train}'] = best\n"
    "    scores = {str(k): round(v, 5) for k, v in means.items()}\n"
    "    print(f'{dgp_name} n={n_train}: multiplier {best}  scores {scores}',\n"
    "          flush=True)\n"
    "\n"
    "document = {\n"
    "    'rule': 'held-out energy score, pilot seeds 100 and 101, '\n"
    "            'candidates ' + repr(BANDWIDTH_CANDIDATES),\n"
    "    'multipliers': multipliers,\n"
    "}\n"
    "path = Path('/content/wcf/results/manifests/'\n"
    "            'phase65_bandwidth_selection.json')\n"
    "path.parent.mkdir(parents=True, exist_ok=True)\n"
    "path.write_text(json.dumps(document, indent=2))\n"
    "print('froze', path)"
)


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit("freeze the manifest first")
    _resolve_pinned_context()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT.glob("*.ipynb"):
        stale.unlink()

    print(f"pinned: {REMOTE_URL} @ {PINNED_COMMIT[:12]}")

    shards = allocate()
    total_cells = 0
    rows = []
    for index, (group, cells, load) in enumerate(shards):
        notebook = build_notebook(index, group, cells, load)
        path = OUTPUT / f"p65_shard_{index:02d}_{group}.ipynb"
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        total_cells += len(cells)
        rows.append((index, group, len(cells), load / 60.0,
                     path.stat().st_size / 1e3))
        print(f"  {path.name:34s} {len(cells):4d} cells  "
              f"{load / 60:6.0f} min  {path.stat().st_size / 1e3:6.0f} kB")

    expected = len(enumerate_phase65_cells())
    if total_cells != expected:
        raise SystemExit(f"allocated {total_cells} cells, expected {expected}")

    readme = OUTPUT / "README.md"
    readme.write_text(_readme(rows, expected), encoding="utf-8")
    print(f"\n{len(shards)} notebooks, {total_cells} cells, wrote {readme}")
    return 0


def _readme(rows, expected: int) -> str:
    lines = [
        "# Phase 6.5 Colab shards",
        "",
        f"{len(rows)} notebooks covering all {expected} cells of the frozen "
        f"manifest `G3-PHASE65-v1` exactly once. Every notebook clones "
        f"`{REMOTE_URL}` at the generating commit `{PINNED_COMMIT[:12]}` and "
        "verifies the frozen manifest checksum before running anything.",
        "",
        "## How to run",
        "",
        "Upload each notebook to Colab and run all cells. They are "
        "independent: run as many concurrently as your accounts allow, in any "
        "order. Each notebook ends by downloading a `.zip`.",
        "",
        "Then, in the repository:",
        "",
        "```bash",
        "# unzip every downloaded bundle into the phase directory",
        "for z in ~/Downloads/p65_shard_*.zip; do unzip -o \"$z\" -d results/; done",
        "",
        "python research/run_phase65.py merge",
        "```",
        "",
        "The merge reconciles every cell key against the manifest and fails "
        "loudly on a duplicate, an unknown key, or a missing cell.",
        "",
        "## Shards",
        "",
        "| Notebook | Group | Cells | Reference minutes | Size |",
        "|---|---|---|---|---|",
    ]
    for index, group, count, minutes, size in rows:
        lines.append(
            f"| `p65_shard_{index:02d}_{group}.ipynb` | {group} | {count} | "
            f"{minutes:.0f} | {size:.0f} kB |"
        )
    lines += [
        "",
        "Reference minutes are single-threaded estimates on the reference "
        "machine, from the audited Phase 6 cost medians. Colab cores are "
        "slower; allow two to three times that, plus install time for the "
        "`forest65` group.",
        "",
        "## Notes",
        "",
        "Every notebook pins all numerical libraries to one thread in its "
        "first cell, before any import. OpenMP sizes its pool at "
        "initialisation, so pinning afterwards does nothing.",
        "",
        "Shards containing `causal_drf_retn` cells run the preregistered "
        "bandwidth-selection pilot first (pilot seeds 100 and 101, held-out "
        "energy score, candidate grid {0.25, 0.5, 1, 2, 4}) and freeze the "
        "multipliers document before any decisive cell.",
        "",
        "Failed cells are preserved with their reason and are never retried "
        "under a different seed. If a shard fails wholesale, for example "
        "because an R package would not build, re-run that notebook; do not "
        "substitute cells from another shard.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
