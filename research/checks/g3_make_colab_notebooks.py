#!/usr/bin/env python3
"""Generate self-contained Colab notebooks for the G3 tournament shards.

Run from the repository root:

    python research/checks/g3_make_colab_notebooks.py

Writes `colab/g3_shards/g3_shard_XX_<group>.ipynb`, one per shard, plus a
README. Each notebook carries the source tree and the frozen manifest as an
embedded base64 archive, so it needs no upload beyond itself and no network
access to this repository.

Shards are grouped by dependency rather than split blindly. Installing R and
compiling `drf` costs roughly twenty minutes, and `mvbcf` must be built from
source; a notebook that runs only the pure-NumPy methods should not pay either.
The four groups are therefore:

* `core`   pure NumPy and SciPy, preinstalled on Colab, no setup;
* `ptas`   adds the `stochtree` wheel;
* `forest` adds R with the pinned `drf` 1.3.1, Rcpp, and RcppEigen;
* `ptaf`   adds R with `mvbcf` built from source.

Notebook count is allocated in proportion to measured work within each group,
using the cost pilot, so no shard runs much longer than any other.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.manifest import enumerate_cells  # noqa: E402

OUTPUT = ROOT / "colab" / "g3_shards"
MANIFEST = ROOT / "results" / "manifests" / "main_manifest.json"
COST_PILOT = ROOT / "results" / "manifests" / "cost_pilot.json"

TOTAL_NOTEBOOKS = 27

#: Method to dependency group.
GROUPS: dict[str, str] = {
    "cwdb_v1": "core",
    "cwdb_v0": "core",
    "cwdb_v1_noshrink": "core",
    "sqw2_booster": "core",
    "pta_s": "ptas",
    "wdrft": "forest",
    "causal_drf": "forest",
    "pta_f": "ptaf",
}

GROUP_ORDER = ("core", "ptas", "forest", "ptaf")

SETUP: dict[str, str] = {
    "core": (
        "# This group needs only NumPy, SciPy and PyArrow, all preinstalled on\n"
        "# Colab. Nothing to install.\n"
        "!python -c \"import numpy, scipy, pyarrow; "
        "print('numpy', numpy.__version__, '| scipy', scipy.__version__, "
        "'| pyarrow', pyarrow.__version__)\"\n"
    ),
    "ptas": (
        "# PTA-S fits one stochtree BCF head per target coordinate.\n"
        "!pip -q install 'stochtree==0.4.5' 2>&1 | tail -2\n"
        "!python -c \"import stochtree; print('stochtree ready')\"\n"
    ),
    "forest": (
        "# W-DRF-T uses the pinned CRAN drf 1.3.1; Causal-DRF compiles an Rcpp\n"
        "# translation unit. Expect fifteen to twenty-five minutes here.\n"
        "%%bash\n"
        "set -e\n"
        "apt-get -qq update > /dev/null 2>&1\n"
        "apt-get -qq install -y r-base r-base-dev libcurl4-openssl-dev "
        "libssl-dev libxml2-dev > /dev/null 2>&1\n"
        "Rscript -e 'options(Ncpus=2); "
        "install.packages(c(\"Rcpp\",\"RcppEigen\",\"jsonlite\"), "
        "repos=\"https://cloud.r-project.org\", quiet=TRUE)'\n"
        "Rscript -e 'options(Ncpus=2); "
        "install.packages(\"https://cran.r-project.org/src/contrib/Archive/drf/drf_1.3.1.tar.gz\", "
        "repos=NULL, type=\"source\", quiet=TRUE)' "
        "|| Rscript -e 'options(Ncpus=2); install.packages(\"drf\", "
        "repos=\"https://cloud.r-project.org\", quiet=TRUE)'\n"
        "Rscript -e 'cat(\"drf\", as.character(packageVersion(\"drf\")), \"ready\\n\")'\n"
    ),
    "ptaf": (
        "# PTA-F needs the mvbcf package, which is built from source.\n"
        "%%bash\n"
        "set -e\n"
        "apt-get -qq update > /dev/null 2>&1\n"
        "apt-get -qq install -y r-base r-base-dev libcurl4-openssl-dev "
        "libssl-dev libxml2-dev git > /dev/null 2>&1\n"
        "Rscript -e 'options(Ncpus=2); "
        "install.packages(c(\"Rcpp\",\"RcppArmadillo\",\"jsonlite\",\"remotes\"), "
        "repos=\"https://cloud.r-project.org\", quiet=TRUE)'\n"
        "Rscript -e 'options(Ncpus=2); "
        "remotes::install_github(\"Nathan-McJames/mvbcf\", quiet=TRUE, "
        "upgrade=\"never\")' || echo 'MVBCF INSTALL FAILED - see the note below'\n"
        "Rscript -e 'cat(\"mvbcf available:\", "
        "requireNamespace(\"mvbcf\", quietly=TRUE), \"\\n\")'\n"
    ),
}

GROUP_NOTE: dict[str, str] = {
    "core": (
        "This shard runs the C-WDB variants and the squared-$W_2$ comparator. "
        "They are pure NumPy and SciPy, so there is no install step and the "
        "notebook starts computing immediately."
    ),
    "ptas": (
        "This shard runs PTA-S, which fits $D = K + J + 1$ independent scalar "
        "BCF heads through `stochtree`. Its cost is linear in $D$, measured at "
        "1.14 s per head."
    ),
    "forest": (
        "This shard runs the two R forest baselines. The setup cell installs R "
        "and the pinned `drf` 1.3.1 and takes fifteen to twenty-five minutes; "
        "the Causal-DRF Rcpp unit is compiled once before the shard fans out, "
        "not per cell."
    ),
    "ptaf": (
        "This shard runs PTA-F, the forced-shared MVBCF endpoint, at $K = 5$ "
        "so its target dimension is $D = 8$. If the `mvbcf` install fails, the "
        "cells will be recorded as failures with that reason, which is a valid "
        "result: the merge keeps failures explicit rather than dropping them."
    ),
}


def build_archive() -> str:
    """Base64 tar.gz of everything a shard needs to run."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(ROOT / "src" / "wasserstein_causal_forests",
                    arcname="src/wasserstein_causal_forests")
        archive.add(ROOT / "research" / "baselines", arcname="research/baselines")
        archive.add(MANIFEST, arcname="results/manifests/main_manifest.json")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pilot_costs() -> dict[tuple, float]:
    document = json.loads(COST_PILOT.read_text(encoding="utf-8"))
    by_shape: dict[tuple, float] = {}
    worst: dict[str, float] = {}
    for row in document["measurements"]:
        if row["status"] != "ok":
            continue
        by_shape[(row["method"], row["n_train"], row["n_grid"],
                  row["n_particles"])] = float(row["wall_seconds"])
        worst[row["method"]] = max(worst.get(row["method"], 0.0),
                                   float(row["wall_seconds"]))
    return by_shape, worst


def allocate() -> list[tuple[str, list[dict]]]:
    """Split cells into TOTAL_NOTEBOOKS shards, grouped by dependency."""

    by_shape, worst = pilot_costs()

    def cost(cell) -> float:
        return by_shape.get(
            (cell.method, cell.n_train, cell.n_grid, cell.n_particles),
            worst.get(cell.method, 30.0),
        )

    grouped: dict[str, list] = defaultdict(list)
    for cell in enumerate_cells():
        grouped[GROUPS[cell.method]].append(cell)

    group_cost = {
        name: sum(cost(c) for c in cells) for name, cells in grouped.items()
    }
    total_cost = sum(group_cost.values())

    # Give every group at least one notebook, then hand out the rest in
    # proportion to measured work so no shard is much longer than the others.
    counts = {name: 1 for name in grouped}
    for _ in range(TOTAL_NOTEBOOKS - len(grouped)):
        pressure = {
            name: group_cost[name] / counts[name] for name in grouped
        }
        counts[max(pressure, key=pressure.get)] += 1

    shards: list[tuple[str, list[dict]]] = []
    for name in GROUP_ORDER:
        cells = grouped.get(name, [])
        if not cells:
            continue
        # Longest cell first into the currently lightest bin: a greedy
        # partition keeps the slowest shard close to the average.
        bins: list[list] = [[] for _ in range(counts[name])]
        loads = [0.0] * counts[name]
        for cell in sorted(cells, key=cost, reverse=True):
            index = loads.index(min(loads))
            bins[index].append(cell)
            loads[index] += cost(cell)
        for bin_cells, load in zip(bins, loads):
            shards.append((name, [c.to_dict() for c in bin_cells], load))
    return shards


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": text,
    }


def build_notebook(index: int, group: str, cells: list[dict], load: float,
                   archive: str) -> dict:
    shard_name = f"g3_shard_{index:02d}_{group}"
    methods = sorted({c["method"] for c in cells})
    estimate_minutes = load / 60.0

    notebook_cells = [
        markdown(
            f"# G3 tournament shard {index:02d} ({group})\n"
            "\n"
            f"Runs **{len(cells)} cells** of the frozen Phase G3 manifest "
            f"(`G3-MAIN-v1`), covering: {', '.join('`' + m + '`' for m in methods)}.\n"
            "\n"
            f"{GROUP_NOTE[group]}\n"
            "\n"
            f"Measured compute for this shard on the reference machine is about "
            f"**{estimate_minutes:.0f} minutes** single-threaded. Colab cores are "
            "slower, so allow two to three times that, plus any install time "
            "above. This fits comfortably inside a nine hour session.\n"
            "\n"
            "**Run every cell in order.** The last cell downloads a `.zip` "
            "containing this shard's results; collect all shards' zips into "
            "`results/main/` in the repository and run the merge.\n"
            "\n"
            "The shard is idempotent: re-running the notebook recomputes the "
            "same cells from the same seeds and produces identical rows."
        ),
        code(
            "# Thread pinning MUST happen before NumPy, SciPy or stochtree are\n"
            "# imported. OpenMP sizes its pool at initialisation, so setting\n"
            "# these afterwards is silently ineffective and costs roughly a\n"
            "# factor of forty in throughput.\n"
            "import os\n"
            "for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',\n"
            "           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',\n"
            "           'VECLIB_MAXIMUM_THREADS', 'R_NUM_THREADS'):\n"
            "    os.environ[_v] = '1'\n"
            "print('threads pinned to 1')"
        ),
        markdown("## 1. Unpack the source tree"),
        code(
            "import base64, io, tarfile, pathlib\n"
            "\n"
            "ARCHIVE = '''" + archive + "'''\n"
            "\n"
            "workdir = pathlib.Path('/content/wcf')\n"
            "workdir.mkdir(parents=True, exist_ok=True)\n"
            "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE))) as tar:\n"
            "    tar.extractall(workdir)\n"
            "os.chdir(workdir)\n"
            "import sys\n"
            "sys.path.insert(0, str(workdir / 'src'))\n"
            "print('source ready at', workdir)\n"
            "print(sorted(p.name for p in (workdir / 'src' / "
            "'wasserstein_causal_forests').iterdir())[:12])"
        ),
        markdown("## 2. Dependencies"),
        code(SETUP[group]),
        markdown("## 3. This shard's cells"),
        code(
            "import json\n"
            "SHARD_INDEX = " + str(index) + "\n"
            "GROUP = " + repr(group) + "\n"
            "CELLS = json.loads('''" + json.dumps(cells) + "''')\n"
            "print(f'{len(CELLS)} cells in this shard')\n"
            "import collections\n"
            "for key, count in sorted(collections.Counter(\n"
            "        (c['grid'], c['method']) for c in CELLS).items()):\n"
            "    print(f'  {key[0]:12s} {key[1]:18s} {count}')"
        ),
        markdown("## 4. Run"),
        code(
            "import time\n"
            "from pathlib import Path\n"
            "from wasserstein_causal_forests.g3.manifest import Cell\n"
            "from wasserstein_causal_forests.g3.runner import run_shard\n"
            "\n"
            + ("from wasserstein_causal_forests.g3 import r_bridge\n"
               "cache = Path('/content/wcf/results/rcpp_cache')\n"
               "cache.mkdir(parents=True, exist_ok=True)\n"
               "# Compile the Causal-DRF Rcpp unit once, before any cell runs.\n"
               "try:\n"
               "    r_bridge.warm_rcpp_cache(cache)\n"
               "    print('Rcpp cache warm')\n"
               "except Exception as error:\n"
               "    print('cache warm failed, cells will report it:', error)\n"
               if group in ("forest", "ptaf") else
               "cache = None\n") +
            "\n"
            "cells = [Cell(**{k: v for k, v in item.items()\n"
            "                 if k not in ('cell_key', 'test_seed')})\n"
            "         for item in CELLS]\n"
            "\n"
            "out = Path('/content/wcf/results/main')\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "log = Path(f'/content/wcf/results/manifests/"
            "execution_log_{SHARD_INDEX:03d}.jsonl')\n"
            "log.parent.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "started = time.time()\n"
            "summary = run_shard(\n"
            "    cells,\n"
            "    out / f'shard_{SHARD_INDEX:03d}.parquet',\n"
            "    cache_directory=cache,\n"
            "    log_path=log,\n"
            ")\n"
            "print(json.dumps(summary, indent=2))\n"
            "print(f'elapsed {(time.time() - started) / 60:.1f} min')"
        ),
        markdown(
            "## 5. Check\n"
            "\n"
            "Every cell must appear exactly once, as a success or as a failure. "
            "Failures are kept: the merge reconciles them against the manifest "
            "and reports them, and a seed is never silently replaced."
        ),
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
        ),
        markdown("## 6. Download the results"),
        code(
            "import shutil\n"
            "bundle = f'/content/{shard_name}'\n".replace(
                "{shard_name}", shard_name
            ) +
            "staging = Path('/content/bundle')\n"
            "if staging.exists():\n"
            "    shutil.rmtree(staging)\n"
            "staging.mkdir(parents=True)\n"
            "shutil.copy(out / f'shard_{SHARD_INDEX:03d}.parquet', staging)\n"
            "shutil.copy(log, staging)\n"
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
        ),
    ]

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


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit("freeze the manifest first")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT.glob("*.ipynb"):
        stale.unlink()

    archive = build_archive()
    print(f"embedded archive: {len(archive) / 1e6:.2f} MB base64")

    shards = allocate()
    total_cells = 0
    rows = []
    for index, (group, cells, load) in enumerate(shards):
        notebook = build_notebook(index, group, cells, load, archive)
        path = OUTPUT / f"g3_shard_{index:02d}_{group}.ipynb"
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        total_cells += len(cells)
        rows.append((index, group, len(cells), load / 60.0,
                     path.stat().st_size / 1e6))
        print(f"  {path.name:34s} {len(cells):5d} cells  "
              f"{load / 60:6.1f} min  {path.stat().st_size / 1e6:.2f} MB")

    expected = len(enumerate_cells())
    if total_cells != expected:
        raise SystemExit(f"allocated {total_cells} cells, expected {expected}")

    readme = OUTPUT / "README.md"
    readme.write_text(_readme(rows, expected), encoding="utf-8")
    print(f"\n{len(shards)} notebooks, {total_cells} cells, wrote {readme}")
    return 0


def _readme(rows, expected: int) -> str:
    lines = [
        "# G3 tournament Colab shards",
        "",
        f"{len(rows)} notebooks covering all {expected} cells of the frozen "
        "manifest `G3-MAIN-v1` exactly once.",
        "",
        "## How to run",
        "",
        "Upload each notebook to Colab and run all cells. They are independent: "
        "run as many concurrently as your account allows, in any order. Each "
        "notebook ends by downloading a `.zip`.",
        "",
        "Then, in the repository:",
        "",
        "```bash",
        "# unzip every downloaded bundle into results/",
        "for z in ~/Downloads/g3_shard_*.zip; do unzip -o \"$z\" -d results/main/; done",
        "mv results/main/execution_log_*.jsonl results/manifests/ 2>/dev/null",
        "",
        "python research/run_g3.py merge",
        "python research/checks/g3_report.py",
        "python research/checks/g3_gate_flags.py",
        "python research/checks/g3_write_memo.py",
        "```",
        "",
        "The merge reconciles every cell key against the manifest and fails "
        "loudly on a duplicate, an unknown key, or a missing cell, so a "
        "forgotten shard cannot pass silently as a smaller tournament.",
        "",
        "## Shards",
        "",
        "| Notebook | Group | Cells | Reference minutes | Size |",
        "|---|---|---|---|---|",
    ]
    for index, group, count, minutes, size in rows:
        lines.append(
            f"| `g3_shard_{index:02d}_{group}.ipynb` | {group} | {count} | "
            f"{minutes:.0f} | {size:.2f} MB |"
        )
    lines += [
        "",
        "Reference minutes are single-threaded on the machine that ran the cost "
        "pilot. Colab cores are slower; allow two to three times that, plus "
        "install time for the `forest` and `ptaf` groups.",
        "",
        "## Notes",
        "",
        "Every notebook pins all numerical libraries to one thread in its first "
        "cell, before any import. This is not optional. OpenMP sizes its pool "
        "at initialisation, so pinning after `import numpy` does nothing, and "
        "an unpinned run on the reference machine lost roughly a factor of "
        "forty to oversubscription.",
        "",
        "Failed cells are preserved with their reason and are never retried "
        "under a different seed. If a shard fails wholesale, for example "
        "because an R package would not build, re-run that notebook; do not "
        "substitute cells from another shard.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
