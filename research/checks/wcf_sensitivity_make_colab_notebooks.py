#!/usr/bin/env python3
"""Generate self-contained Colab shard notebooks for the WCF sensitivity study.

Run from the repository root:

    python3 research/checks/wcf_sensitivity_make_colab_notebooks.py

Writes `colab/wcf_sensitivity_shards/wcf_sensitivity_shard_XX.ipynb`, one per
shard, plus a README. The working tree that contains the sensitivity modules is
dirty and uncommitted, so nothing can be cloned: each notebook carries a
base64-encoded ZIP of `src/wasserstein_causal_forests/` (plus
`research/baselines/` and `code/drfinference-main/` when the optional R forest
methods are requested), unpacks it to a temporary directory, and inserts the
extracted `src` on `sys.path`. Each shard also embeds its manifest slice, so no
repository file is needed at run time.

The allocation is a rough cost model over the frozen cells. With
`results/wcf_sensitivity/cost_pilot.json` present it interpolates the measured
wall seconds over K and M; otherwise it scales the documented base estimate for
`cwdb_dr` (about 200 s at n=1000, K=25, M=10, dense common-grid evaluation
included) by `(K/25) * (M/10)^2` with a floor. Work is packed into shards of
roughly thirty minutes, never splitting a replication, and capped at 48 shards.

The local tmux run (`python3 research/run_wcf_sensitivity.py run`) is the
primary execution path; these notebooks are the overflow/contingency copy.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.wcf_sensitivity import (  # noqa: E402
    WCF_BLOCKS,
    WCF_PRIMARY_BLOCKS,
    WCF_SENSITIVITY_CONTRACT_ID,
    build_wcf_sensitivity_cells,
)

MANIFEST_PATH = ROOT / "results" / "wcf_sensitivity" / "manifest.json"
DEFAULT_COST_PILOT = ROOT / "results" / "wcf_sensitivity" / "cost_pilot.json"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "colab" / "wcf_sensitivity_shards"

DEFAULT_BLOCKS = ",".join(WCF_PRIMARY_BLOCKS)
DEFAULT_METHODS = "cwdb_dr,cwdb_dr_flex,cwdb_dr_oracle"
PURE_METHODS = ("cwdb_dr", "cwdb_dr_flex", "cwdb_dr_oracle")
FOREST_METHODS = ("causal_drf", "drf")

TARGET_SHARD_SECONDS = 1800.0
MAX_SHARDS = 48
MINIMUM_SECONDS = 30.0

#: Rough single-threaded seconds on the reference machine for one cell at
#: n_train=1000, K=25, M=10, with the dense common-grid evaluation included.
#: The 200 s figure for `cwdb_dr` is the documented historical record; the
#: other three are declared as the same order of magnitude because no separate
#: measurement exists. The cost pilot replaces all of this when present.
REFERENCE_SECONDS: dict[str, float] = {
    "cwdb_dr": 200.0,
    "cwdb_dr_flex": 200.0,
    "cwdb_dr_oracle": 200.0,
    "causal_drf": 40.0,
    "drf": 40.0,
}
REFERENCE_N_TRAIN = 1000
REFERENCE_N_GRID = 25
REFERENCE_N_PARTICLES = 10

R_LIBRARY = "/content/Rlib/causal_drf"
CAUSAL_CLEAN_COMMIT = "0a1a508444176b5b1553f13e832be93a374b0af2"

FOREST_SETUP = """\
# This shard contains R forest cells. The setup installs R, the pinned CRAN
# `drf` 1.3.1, and the authors' causal-clean package at the frozen commit into
# a notebook-local library that WCF_CAUSAL_DRF_R_LIB selects. Expect fifteen
# to twenty-five minutes for this cell.
%%bash
set -e
apt-get -qq update > /dev/null 2>&1
apt-get -qq install -y r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev curl > /dev/null 2>&1
Rscript -e 'options(Ncpus=2); install.packages(c("Rcpp","RcppEigen","jsonlite","remotes","transport","fastDummies","kernlab"), repos="https://cloud.r-project.org", quiet=TRUE)'
# CRAN drf 1.3.1 drives the paper-DRF baseline; the causal-clean library
# below shadows it only for Causal-DRF cells.
Rscript -e 'options(Ncpus=2); if (!requireNamespace("drf", quietly=TRUE)) install.packages("drf", repos="https://cloud.r-project.org", quiet=TRUE); cat("CRAN drf", as.character(packageVersion("drf")), "ready\\n")'
mkdir -p /content/Rlib/causal_drf
CAUSAL_SHA="__CAUSAL_SHA__"
if [ ! -d /content/Rlib/causal_drf/drf ]; then
  TARBALL="/tmp/causal_clean_${CAUSAL_SHA:0:12}.tar.gz"
  curl -sL "https://codeload.github.com/herbps10/drf/tar.gz/${CAUSAL_SHA}" -o "$TARBALL"
  EXTRACT="/tmp/causal_clean_src"
  rm -rf "$EXTRACT"; mkdir -p "$EXTRACT"
  tar -xzf "$TARBALL" -C "$EXTRACT"
  PKG_DIR=$(find "$EXTRACT" -maxdepth 3 -type d -path "*r-package/drf" | head -1)
  echo "installing causal-clean drf from $PKG_DIR"
  R CMD INSTALL --library=/content/Rlib/causal_drf "$PKG_DIR" \\
    || Rscript -e 'options(Ncpus=2); .libPaths(c("/content/Rlib/causal_drf",.libPaths())); remotes::install_github("herbps10/drf", ref="__CAUSAL_SHA__", subdir="r-package/drf", lib="/content/Rlib/causal_drf", upgrade="never", quiet=TRUE)'
fi
Rscript -e '.libPaths(c("/content/Rlib/causal_drf",.libPaths())); stopifnot(requireNamespace("drf", quietly=TRUE)); cat("causal-clean drf", as.character(packageVersion("drf")), "ready\\n")'
echo 'setup complete'
""".replace("__CAUSAL_SHA__", CAUSAL_CLEAN_COMMIT)

FOREST_ENVIRONMENT = """\
# The R bridge subprocesses inherit this variable from the notebook process, so
# it must be set in Python rather than in the install shell below.
import os
os.environ['WCF_CAUSAL_DRF_R_LIB'] = '/content/Rlib/causal_drf'
print('WCF_CAUSAL_DRF_R_LIB =', os.environ['WCF_CAUSAL_DRF_R_LIB'])
"""

THREAD_PIN_CELL = """\
# Thread pinning MUST happen before NumPy or SciPy are imported. OpenMP sizes
# its pool at initialisation, so setting these afterwards is silently
# ineffective.
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'R_NUM_THREADS'):
    os.environ[_v] = '1'
print('threads pinned to 1')"""

RUNNER_CELL = """\
# The runner is a per-cell checkpoint loop: after every cell the accumulated
# rows are rewritten to a temporary parquet and atomically renamed, so an
# interrupted session resumes from the parquet rather than from the start.
import json
import os
import time
from pathlib import Path

import pyarrow  # noqa: F401  (runner.write_rows falls back to JSONL without it)
import pyarrow.parquet as pq
from wasserstein_causal_forests.g3 import common_grid, runner
from wasserstein_causal_forests.g3.manifest import Cell

OUTPUT_DIRECTORY = Path('shard_output')
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = OUTPUT_DIRECTORY / 'wcf_sensitivity_parquet.parquet'
ROWS_PATH = OUTPUT_DIRECTORY / 'rows.jsonl'
LOG_PATH = OUTPUT_DIRECTORY / 'execution_log.jsonl'
FAILURE_PATH = OUTPUT_DIRECTORY / 'failure_rows.jsonl'
CACHE_DIRECTORY = OUTPUT_DIRECTORY / 'cache'
CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _read_rows(path):
    if not path.exists():
        return []
    return pq.read_table(path).to_pandas().to_dict('records')


def _replication_key(item):
    return (item['grid'], item['dgp'], item['n_train'],
            item['n_grid'], item['n_particles'], item['seed'])


rows = _read_rows(PARQUET_PATH)
by_cell = {}
for row in rows:
    by_cell.setdefault(row['cell_key'], []).append(row)
successful = {key for key, group in by_cell.items()
              if not any(row.get('metric') == 'cell_failure' for row in group)}
failed = {key for key, group in by_cell.items()
          if any(row.get('metric') == 'cell_failure' for row in group)}
if failed:
    # A failed cell is not a resume marker: drop its old rows and retry it.
    rows = [row for row in rows if row['cell_key'] not in failed]
    print(f'resuming: {len(successful)} successful cells, '
          f'retrying {len(failed)} failed cells')

method_order = {name: position for position, name in enumerate(METHOD_ORDER)}
ordered_cells = sorted(
    CELLS,
    key=lambda item: (_replication_key(item), method_order[item['method']]),
)

n_ok = 0
n_failed = 0
n_skipped = 0
started = time.time()
try:
    for position, item in enumerate(ordered_cells, start=1):
        if item['cell_key'] in successful:
            n_skipped += 1
            continue
        cell = Cell(**{key: value for key, value in item.items()
                       if key not in ('cell_key', 'test_seed')})
        assert cell.key == item['cell_key'], 'cell key mismatch in the slice'
        cell_rows = runner.run_cell(
            cell,
            cache_directory=CACHE_DIRECTORY,
            manifest_contract_id=CONTRACT_ID,
            extra_evaluators=(common_grid.evaluate_common_grid,),
        )
        rows.extend(cell_rows)
        status = 'failed' if cell_rows[0].get('status') == 'failed' else 'ok'
        n_ok += status == 'ok'
        n_failed += status == 'failed'
        temporary = PARQUET_PATH.with_suffix('.tmp')
        runner.write_rows(rows, temporary)
        os.replace(temporary, PARQUET_PATH)
        wall_seconds = float(cell_rows[0].get('wall_seconds', 0.0))
        with ROWS_PATH.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(
                {'cell_key': cell.key, 'rows': cell_rows}, default=str) + '\\n')
        with LOG_PATH.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'cell_key': cell.key, 'grid': cell.grid, 'dgp': cell.dgp,
                'n_train': cell.n_train, 'n_grid': cell.n_grid,
                'n_particles': cell.n_particles, 'method': cell.method,
                'seed': cell.seed, 'status': status, 'n_rows': len(cell_rows),
                'wall_seconds': round(wall_seconds, 3),
                'finished_at': time.time(),
            }) + '\\n')
        if status == 'failed':
            with FAILURE_PATH.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(
                    {'cell_key': cell.key, 'rows': cell_rows},
                    default=str) + '\\n')
        print(f'[{position}/{len(ordered_cells)}] {cell.dgp} '
              f'n={cell.n_train} K={cell.n_grid} M={cell.n_particles} '
              f'{cell.method} seed={cell.seed}: {status} '
              f'{wall_seconds:.1f}s', flush=True)
except KeyboardInterrupt:
    print('interrupted; re-run this cell to resume from the parquet '
          'checkpoint', flush=True)

print(f'shard finished: {n_ok} ok, {n_failed} failed, {n_skipped} already '
      f'present, {len(rows)} rows, {(time.time() - started) / 60.0:.1f} min')"""

DIAGNOSTICS_CELL = """\
# Diagnostics for the shard bundle: exact code identity, shard slice, and the
# dependency versions that Colab actually provided.
import json
import platform
import sys
import time
from pathlib import Path

versions = {'python': sys.version.split()[0], 'platform': platform.platform()}
for _name in ('numpy', 'scipy', 'sklearn', 'pandas', 'pyarrow'):
    try:
        _module = __import__(_name)
        versions[_name] = getattr(_module, '__version__', 'unknown')
    except ImportError:
        versions[_name] = None

config = {
    'shard_index': SHARD_INDEX,
    'shard_total': SHARD_TOTAL,
    'contract_id': CONTRACT_ID,
    'manifest_checksum': MANIFEST_SLICE['manifest_checksum'],
    'estimator_source_hash': MANIFEST_SLICE.get('estimator_source_hash'),
    'source_archive_sha256': SOURCE_ARCHIVE_SHA256,
    'blocks': MANIFEST_SLICE['blocks'],
    'methods': METHOD_ORDER,
    'n_cells': len(CELLS),
    'cell_keys': [item['cell_key'] for item in CELLS],
    'estimated_seconds_reference': ESTIMATED_SECONDS,
    'versions': versions,
}
Path('shard_output').mkdir(parents=True, exist_ok=True)
Path('shard_output/shard_config.json').write_text(
    json.dumps(config, indent=2), encoding='utf-8')
Path('shard_output/manifest_slice.json').write_text(
    json.dumps(MANIFEST_SLICE, indent=2), encoding='utf-8')
# The sidecar uses the keys the local merge verifies, so an overflow shard can
# be renamed into results/wcf_sensitivity/shards/ without inventing metadata.
Path('shard_output/wcf_sensitivity_parquet.meta.json').write_text(
    json.dumps({
        'manifest_checksum': MANIFEST_SLICE['manifest_checksum'],
        'estimator_source_hash': MANIFEST_SLICE.get('estimator_source_hash'),
        'contract_id': CONTRACT_ID,
        'updated_at': time.time(),
    }, indent=2), encoding='utf-8')
print(json.dumps(config, indent=2))"""

DOWNLOAD_CELL = """\
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
output_file = "wcf_sensitivity_shard.zip"
with ZipFile(output_file, "w", ZIP_DEFLATED) as archive:
    for path in Path("shard_output").rglob("*"):
        if path.is_file(): archive.write(path, arcname=path.relative_to("shard_output"))
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)"""

FINAL_MARKDOWN = """\
## What to send back

Download the single `wcf_sensitivity_shard.zip`. It contains this shard's
`wcf_sensitivity_parquet.parquet`, its execution log, its failure rows (if
any), the embedded manifest slice, and `shard_config.json` with the embedded
source hash and dependency versions. Do not hand-edit the parquet, the log, or
the failure rows: the merge reconciles every cell key against the frozen
manifest, keeps failures explicit, and refuses duplicates. The local tmux run
remains the primary path; this notebook is the contingency copy.

To feed an overflow bundle into the local merge, unzip it into
`results/wcf_sensitivity/shards/`, rename the parquet to
`shard_colab_<index>.parquet`, and rename `wcf_sensitivity_parquet.meta.json` to
`shard_colab_<index>.meta.json` (for example `shard_colab_00.parquet`); the
sidecar already carries the contract, checksum, and estimator hash the merge
verifies. The `colab_` infix keeps overflow shards from overwriting the local
runner's numeric `shard_000.parquet` files."""


class RoughCostModel:
    """Wall-second estimates per (method, n_train, K, M).

    With a cost pilot the measured points are used directly; missing shapes are
    filled by linear interpolation over K and over M, with power-law
    continuation outside the measured range. Without a pilot the documented
    base estimate is scaled by `(n/1000) * (K/25) * (M/10)^2` and floored.
    Every number is a rough allocation device, not a prediction.
    """

    def __init__(self, measurements: dict[tuple, float] | None) -> None:
        self.measurements = measurements or {}
        self.source = "base estimates" if not measurements else "cost pilot"

    def seconds(
        self, method: str, n_train: int, n_grid: int, n_particles: int
    ) -> float:
        exact = self.measurements.get(
            (method, n_train, n_grid, n_particles)
        )
        if exact is not None:
            return max(MINIMUM_SECONDS, float(exact))
        if self.measurements:
            interpolated = self._interpolate(
                method, n_train, n_grid, n_particles
            )
            if interpolated is not None:
                return max(MINIMUM_SECONDS, interpolated)
        return max(
            MINIMUM_SECONDS,
            _base_seconds(method, n_train, n_grid, n_particles),
        )

    def _interpolate(
        self, method: str, n_train: int, n_grid: int, n_particles: int
    ) -> float | None:
        points = [
            (key[1], key[2], key[3], value)
            for key, value in self.measurements.items()
            if key[0] == method
        ]
        if not points:
            return None
        sample_sizes = sorted({point[0] for point in points})
        reference_n = min(
            sample_sizes, key=lambda value: abs(math.log(n_train / value))
        )
        by_particles: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for n_value, k_value, m_value, wall in points:
            if n_value == reference_n:
                by_particles[m_value].append((k_value, wall))
        particles = sorted(by_particles)
        if len(particles) == 1:
            grid_estimate = _linear_axis(
                by_particles[particles[0]], n_grid, exponent=1.0
            )
            estimate = grid_estimate * (
                n_particles / particles[0]
            ) ** 2
        else:
            estimates = [
                _linear_axis(by_particles[m], n_grid, exponent=1.0)
                for m in particles
            ]
            estimate = _linear_axis(
                list(zip(particles, estimates)),
                n_particles,
                exponent=2.0,
            )
        return estimate * (n_train / reference_n)


def _linear_axis(
    points: list[tuple[float, float]], target: float, *, exponent: float
) -> float:
    """Linear interpolation in range, power-law continuation outside it."""

    ordered = sorted(points)
    if len(ordered) == 1:
        coordinate, value = ordered[0]
        return value * (target / coordinate) ** exponent
    for (lower, lower_value), (upper, upper_value) in zip(
        ordered, ordered[1:]
    ):
        if lower <= target <= upper:
            fraction = (target - lower) / (upper - lower)
            return lower_value + fraction * (upper_value - lower_value)
    coordinate, value = (
        ordered[0] if target < ordered[0][0] else ordered[-1]
    )
    return value * (target / coordinate) ** exponent


def _base_seconds(
    method: str, n_train: int, n_grid: int, n_particles: int
) -> float:
    reference = REFERENCE_SECONDS.get(method, 200.0)
    return (
        reference
        * (n_train / REFERENCE_N_TRAIN)
        * (n_grid / REFERENCE_N_GRID)
        * (n_particles / REFERENCE_N_PARTICLES) ** 2
    )


def load_measurements(path: Path | None) -> dict[tuple, float] | None:
    if path is None or not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"warning: cannot read {path} ({error}); using base estimates")
        return None
    measurements: dict[tuple, float] = {}
    for run in document.get("runs", []):
        if run.get("status") != "ok":
            continue
        try:
            key = (
                str(run["method"]),
                int(run["n_train"]),
                int(run["n_grid"]),
                int(run["n_particles"]),
            )
            wall = float(run["wall_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(wall) and wall > 0.0:
            measurements[key] = wall
    if not measurements:
        print(f"warning: {path} has no successful runs; using base estimates")
        return None
    return measurements


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"frozen manifest not found at {path}; run "
            "`python3 research/run_wcf_sensitivity.py freeze` first"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    for field in ("cells", "method_registry", "manifest_checksum", "blocks"):
        if field not in document:
            raise SystemExit(f"manifest {path} is missing {field!r}")
    return document


def block_membership(blocks: tuple[str, ...]) -> dict[str, list[str]]:
    """Reconstruct block tags by intersecting rebuilt cell keys."""

    membership: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        for cell in build_wcf_sensitivity_cells(block):
            membership[cell.key].append(block)
    return dict(membership)


def select_cells(
    document: dict,
    blocks: tuple[str, ...],
    methods: tuple[str, ...],
    membership: dict[str, list[str]],
) -> tuple[list[dict], dict[str, list[str]]]:
    selected: list[dict] = []
    tags: dict[str, list[str]] = {}
    for cell in document["cells"]:
        if cell["method"] not in methods:
            continue
        blocks_of_cell = membership.get(cell["cell_key"])
        if not blocks_of_cell:
            continue
        selected.append(dict(cell))
        tags[cell["cell_key"]] = blocks_of_cell
    return selected, tags


def replication_key(cell: dict) -> tuple:
    return (
        cell["grid"],
        cell["dgp"],
        cell["n_train"],
        cell["n_grid"],
        cell["n_particles"],
        cell["seed"],
    )


def group_replications(cells: list[dict]) -> list[list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for cell in cells:
        groups[replication_key(cell)].append(cell)
    return list(groups.values())


def allocate(
    cells: list[dict],
    model: RoughCostModel,
    shard_count: int,
    method_order: tuple[str, ...],
) -> tuple[list[list[dict]], list[float]]:
    """Greedy balanced packing; a replication is never split across shards."""

    groups = group_replications(cells)

    def group_seconds(group: list[dict]) -> float:
        return sum(
            model.seconds(
                cell["method"],
                cell["n_train"],
                cell["n_grid"],
                cell["n_particles"],
            )
            for cell in group
        )

    groups.sort(key=group_seconds, reverse=True)
    positions = {name: index for index, name in enumerate(method_order)}
    bins: list[list[dict]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for group in groups:
        index = loads.index(min(loads))
        bins[index].extend(group)
        loads[index] += group_seconds(group)
    for shard in bins:
        shard.sort(
            key=lambda item: (
                replication_key(item),
                positions[item["method"]],
            )
        )
    return bins, loads


def build_source_archive(include_forest: bool) -> tuple[str, str]:
    """Base64 ZIP of the package (plus the R assets when requested).

    The working tree carries uncommitted sensitivity code, so a pinned clone
    would not reproduce it. The archive keeps the exact generating bytes,
    including whichever uncommitted edits are on disk, and the extracted layout
    reproduces the repository-relative paths the R bridge expects.
    """

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        package = ROOT / "src" / "wasserstein_causal_forests"
        for path in sorted(package.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
        if include_forest:
            for name in ("research/baselines", "code/drfinference-main"):
                directory = ROOT / name
                if not directory.is_dir():
                    continue
                for path in sorted(directory.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(directory)
                    if "__pycache__" in path.parts:
                        continue
                    if "results" in relative.parts:
                        continue
                    if path.name.endswith(":Zone.Identifier"):
                        continue
                    archive.write(
                        path, arcname=path.relative_to(ROOT).as_posix()
                    )
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    return base64.b64encode(payload).decode("ascii"), digest


def chunk_base64(payload: str, width: int = 96) -> str:
    return "\n".join(
        payload[index : index + width]
        for index in range(0, len(payload), width)
    )


def setup_cell(
    archive_base64: str, source_sha256: str, checksum: str, include_forest: bool
) -> str:
    forest_note = ""
    if include_forest:
        forest_note = (
            "print('R assets embedded:', "
            "(workdir / 'research/baselines').is_dir())\n"
        )
    return (
        "import base64\n"
        "import pathlib\n"
        "import sys\n"
        "import tempfile\n"
        "import zipfile\n"
        "\n"
        f"SOURCE_ARCHIVE_SHA256 = {source_sha256!r}\n"
        "SOURCE_ARCHIVE_B64 = '''\\\n"
        + chunk_base64(archive_base64)
        + "\n"
        "'''\n"
        "\n"
        "workdir = pathlib.Path(tempfile.mkdtemp(prefix='wcf_sensitivity_'))\n"
        "archive_path = workdir / 'wcf_source.zip'\n"
        "archive_path.write_bytes(base64.b64decode(SOURCE_ARCHIVE_B64))\n"
        "with zipfile.ZipFile(archive_path) as archive:\n"
        "    archive.extractall(workdir)\n"
        "source_directory = workdir / 'src'\n"
        "sys.path.insert(0, str(source_directory))\n"
        "import wasserstein_causal_forests\n"
        "print('embedded source:', wasserstein_causal_forests.__file__)\n"
        "print('source archive sha256:', SOURCE_ARCHIVE_SHA256)\n"
        f"print('manifest checksum: {checksum}')\n"
        + forest_note
    )


def registration_cell(
    index: int,
    total: int,
    estimated_seconds: float,
    slice_document: dict,
    method_order: tuple[str, ...],
) -> str:
    return (
        "import json\n"
        "from wasserstein_causal_forests.g3 import runner\n"
        "from wasserstein_causal_forests.g3.sensitivity_dgps import (\n"
        "    register_sensitivity_dgps,\n"
        ")\n"
        "from wasserstein_causal_forests.g3.sensitivity_methods import (\n"
        "    register_sensitivity_methods,\n"
        ")\n"
        "from wasserstein_causal_forests.g3.wcf_sensitivity import (\n"
        "    apply_method_registry,\n"
        ")\n"
        "\n"
        f"SHARD_INDEX = {index}\n"
        f"SHARD_TOTAL = {total}\n"
        f"ESTIMATED_SECONDS = {estimated_seconds!r}\n"
        "MANIFEST_SLICE = json.loads('''"
        + json.dumps(slice_document)
        + "''')\n"
        "CELLS = MANIFEST_SLICE['cells']\n"
        f"METHOD_ORDER = {list(method_order)!r}\n"
        "CONTRACT_ID = MANIFEST_SLICE['contract_id']\n"
        "\n"
        "register_sensitivity_dgps()\n"
        "register_sensitivity_methods()\n"
        "apply_method_registry("
        "{'method_registry': MANIFEST_SLICE['method_registry']})\n"
        "print('shard', SHARD_INDEX, 'of', SHARD_TOTAL, '|', len(CELLS), "
        "'cells')\n"
        "print('registered methods:', "
        "sorted(MANIFEST_SLICE['method_registry']))"
    )


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


def build_notebook(
    index: int,
    total: int,
    shard_cells: list[dict],
    estimated_seconds: float,
    cost_description: str,
    archive_base64: str,
    source_sha256: str,
    slice_document: dict,
    method_order: tuple[str, ...],
    block_names: tuple[str, ...],
    include_forest: bool,
) -> dict:
    contract_id = slice_document["contract_id"]
    checksum = slice_document["manifest_checksum"]
    methods = sorted({cell["method"] for cell in shard_cells})
    shard_blocks = sorted(
        {block for item in shard_cells for block in slice_document.get("cell_blocks", {}).get(item["cell_key"], [])}
    )
    if not shard_blocks:
        shard_blocks = sorted(set(block_names))
    methods_text = ", ".join(f"`{name}`" for name in methods)
    blocks_text = ", ".join(f"`{name}`" for name in shard_blocks)
    has_forest = any(name in FOREST_METHODS for name in methods)

    cells = [
        markdown(
            f"# WCF sensitivity shard {index:02d}\n"
            "\n"
            f"Shard **{index}** of {total} for the frozen sensitivity study "
            f"(`{contract_id}`, manifest checksum `{checksum[:16]}...`).\n"
            "\n"
            f"This shard runs **{len(shard_cells)} cells** over blocks "
            f"{blocks_text} with methods {methods_text}.\n"
            "\n"
            f"Estimated reference cost is about **{estimated_seconds / 60.0:.0f}"
            f" minutes** single-threaded ({cost_description}). The estimate is "
            "rough and only balances shards; Colab cores are slower, so allow "
            "two to three times that.\n"
            "\n"
            "Every notebook is self-contained: it unpacks an embedded source "
            "archive, embeds its own manifest slice, checkpoints the parquet "
            "after every cell, and ends by downloading one zip. The local tmux "
            "run is the primary path; this shard is the contingency copy.\n"
        ),
        code(THREAD_PIN_CELL),
        markdown(
            "## 1. Embedded source\n"
            "\n"
            "The working tree contains uncommitted sensitivity code, so no "
            "pinned clone can reproduce it. This cell decodes a base64 ZIP of "
            "`src/wasserstein_causal_forests/` (plus the R drivers when this "
            "shard contains forest methods), extracts it to a temporary "
            "directory, and puts the extracted `src` on `sys.path`.\n"
        ),
        code(
            setup_cell(archive_base64, source_sha256, checksum, include_forest)
        ),
    ]
    if include_forest and has_forest:
        cells.append(
            markdown(
                "## 2. R forest dependencies (15-25 minutes)\n"
                "\n"
                "The R bridge subprocesses read `WCF_CAUSAL_DRF_R_LIB` from "
                "the notebook environment. The first cell sets it; the second "
                "installs R, the pinned CRAN `drf` 1.3.1, and the authors' "
                "causal-clean package at the frozen commit. Cells that cannot "
                "run are recorded as failures with their reason, which is a "
                "valid result, not a silent drop.\n"
            )
        )
        cells.append(code(FOREST_ENVIRONMENT))
        cells.append(code(FOREST_SETUP))
    cells += [
        markdown(
            "## 3. Registration and this shard's manifest slice\n"
            "\n"
            "The frozen registry is applied after the sensitivity "
            "registration, so the exact frozen estimator settings win over the "
            "static defaults.\n"
        ),
        code(
            registration_cell(
                index,
                total,
                estimated_seconds,
                slice_document,
                method_order,
            )
        ),
        markdown(
            "## 4. Run\n"
            "\n"
            "Cells are ordered so one replication's methods stay adjacent and "
            "share the cached dense oracle truth. Re-running this cell resumes "
            "from the parquet checkpoint; a failed cell is retried rather than "
            "treated as done.\n"
        ),
        code(RUNNER_CELL),
        markdown(
            "## 5. Diagnostics\n"
            "\n"
            "Writes the shard configuration, the embedded manifest slice, "
            "dependency versions, and the embedded source hash into "
            "`shard_output/` so the bundle is self-describing.\n"
        ),
        code(DIAGNOSTICS_CELL),
        markdown("## 6. Download the results"),
        code(DOWNLOAD_CELL),
        markdown(FINAL_MARKDOWN),
    ]

    return {
        "cells": [
            {**cell, "source": cell["source"].splitlines(keepends=True)}
            for cell in cells
        ],
        "metadata": {
            "colab": {"provenance": [], "name": f"wcf_sensitivity_shard_{index:02d}"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def write_notebook(path: Path, document: dict) -> None:
    try:
        import nbformat

        notebook = nbformat.from_dict(document)
        nbformat.validate(notebook)
    except ImportError:
        pass
    path.write_text(json.dumps(document, indent=1), encoding="utf-8")


def readme_text(
    rows: list[tuple],
    total_cells: int,
    shard_count: int,
    cost_description: str,
    method_order: tuple[str, ...],
    include_forest: bool,
) -> str:
    total_seconds = sum(row[2] for row in rows)
    lines = [
        "# WCF sensitivity Colab shards",
        "",
        f"{shard_count} self-contained notebooks covering {total_cells} cells "
        "of the frozen sensitivity study exactly once, over the requested "
        f"methods: {', '.join('`' + name + '`' for name in method_order)}. "
        f"The rough reference total is {total_seconds / 3600.0:.1f} hours.",
        "",
        "## How to run",
        "",
        "1. Upload one `wcf_sensitivity_shard_XX.ipynb` per Colab session.",
        "2. Choose Runtime, then Run all. The first cell pins every numerical "
        "library to one thread before NumPy is imported.",
        "3. Wait; the notebook checkpoints `shard_output/"
        "wcf_sensitivity_parquet.parquet` after every cell and prints "
        "progress. An interrupted session resumes from that parquet when the "
        "run cell is executed again.",
        "4. Run the final cell; it writes the single "
        "`wcf_sensitivity_shard.zip` and downloads it. Send back exactly that "
        "one zip per shard.",
        "",
        f"Cost estimates are rough ({cost_description}); Colab is slower than "
        "the reference machine, so expect longer wall times. The shards are "
        "independent and can run concurrently.",
        "",
        "The local tmux run (`python3 research/run_wcf_sensitivity.py run "
        "--workers N`) is the primary execution path. These notebooks are the "
        "overflow and contingency copy; results from both paths are written "
        "in the same parquet shape and merge the same way.",
        "",
        "## Shards",
        "",
        "| Notebook | Cells | Estimated minutes | Estimated hours | Methods |",
        "|---|---|---|---|---|",
    ]
    for index, count, seconds, methods in rows:
        lines.append(
            f"| `wcf_sensitivity_shard_{index:02d}.ipynb` | {count} | "
            f"{seconds / 60.0:.0f} | {seconds / 3600.0:.2f} | "
            f"{', '.join(methods)} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "The notebooks require only Colab's preinstalled NumPy, SciPy, "
        "scikit-learn, pandas, and PyArrow. The setup writes the embedded "
        "source to a temporary directory and imports it from there; nothing is "
        "installed with pip unless a shard contains R forest methods, whose "
        "setup cell installs R and takes fifteen to twenty-five minutes.",
        "",
        "Failed cells are preserved with their reason in `failure_rows.jsonl` "
        "and are never retried under a different seed. A failed cell is not a "
        "resume marker: re-running the run cell retries it and replaces its "
        "old rows before the next checkpoint.",
        "",
        "To merge overflow bundles locally, unzip each bundle into "
        "`results/wcf_sensitivity/shards/`, rename "
        "`wcf_sensitivity_parquet.parquet` to `shard_colab_<index>.parquet` and "
        "`wcf_sensitivity_parquet.meta.json` to `shard_colab_<index>.meta.json` "
        "(the `colab_` infix avoids overwriting the local runner's numeric "
        "`shard_000.parquet` files), then "
        "run `python3 research/run_wcf_sensitivity.py merge`.",
        "",
    ]
    if include_forest:
        lines += [
            "The optional R group installs R, the pinned CRAN `drf` 1.3.1, and "
            "the authors' causal-clean package at commit "
            f"`{CAUSAL_CLEAN_COMMIT[:12]}`. If the causal-clean install fails, "
            "`causal_drf` cells are recorded as failures with that reason.",
            "",
        ]
    return "\n".join(lines) + "\n"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks",
        default=DEFAULT_BLOCKS,
        help="comma-separated WCF sensitivity blocks (default: the primary "
        "non-null roster)",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=0,
        help="shard count; 0 computes ceil(total/1800) capped at 48",
    )
    parser.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help="comma-separated methods to run; the R forest baselines need "
        "--include-forest",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY.relative_to(ROOT)),
        help="output directory for the notebooks and README",
    )
    parser.add_argument(
        "--include-forest",
        action="store_true",
        help="embed the R assets and add the R setup cell for causal_drf/drf",
    )
    parser.add_argument(
        "--cost-pilot",
        default=str(DEFAULT_COST_PILOT),
        help="optional cost-pilot JSON; missing means base estimates",
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH.relative_to(ROOT)),
        help="manifest to slice; defaults to the frozen primary manifest",
    )
    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="select every manifest cell of the requested methods, ignoring "
        "block reconstruction; use for addendum stages with custom blocks",
    )
    parser.add_argument(
        "--keys-file",
        default="",
        help="optional JSON list of cell keys restricting the selection",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)

    manifest_path = Path(arguments.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    blocks = tuple(
        part.strip() for part in arguments.blocks.split(",") if part.strip()
    )
    if not arguments.all_cells:
        unknown_blocks = [name for name in blocks if name not in WCF_BLOCKS]
        if unknown_blocks:
            raise SystemExit(
                f"unknown blocks {unknown_blocks}; choose from {list(WCF_BLOCKS)} "
                "or pass --all-cells for a custom stage manifest"
            )
    methods = tuple(
        part.strip() for part in arguments.methods.split(",") if part.strip()
    )
    if not methods:
        raise SystemExit("--methods must name at least one method")
    forest_requested = [name for name in methods if name in FOREST_METHODS]
    if forest_requested and not arguments.include_forest:
        raise SystemExit(
            f"methods {forest_requested} run through R and need "
            "--include-forest; pass it to embed the R assets and the setup cell"
        )

    manifest = load_manifest(manifest_path)
    contract_id = manifest.get(
        "manifest_contract_id", WCF_SENSITIVITY_CONTRACT_ID
    )
    missing = [name for name in methods if name not in manifest["method_registry"]]
    if missing:
        raise SystemExit(
            f"methods {missing} are absent from {manifest_path}; the frozen "
            f"registry has {sorted(manifest['method_registry'])}"
        )

    if arguments.all_cells:
        blocks = tuple(manifest.get("blocks", ()))
        selected = [
            dict(cell)
            for cell in manifest["cells"]
            if cell["method"] in methods
        ]
        cell_blocks = {
            cell["cell_key"]: list(blocks) for cell in selected
        }
    else:
        membership = block_membership(blocks)
        selected, cell_blocks = select_cells(
            manifest, blocks, methods, membership
        )
    if getattr(arguments, "keys_file", ""):
        keys_path = Path(arguments.keys_file)
        if not keys_path.is_absolute():
            keys_path = ROOT / keys_path
        if not keys_path.exists():
            raise SystemExit(f"keys file {keys_path} does not exist")
        wanted_keys = set(json.loads(keys_path.read_text(encoding="utf-8")))
        available_keys = {cell["cell_key"] for cell in selected}
        unknown_keys = wanted_keys - available_keys
        if unknown_keys:
            raise SystemExit(
                f"{len(unknown_keys)} keys in {keys_path} are not selected by "
                f"the manifest/block/method filters"
            )
        selected = [
            cell for cell in selected if cell["cell_key"] in wanted_keys
        ]
        cell_blocks = {
            key: value
            for key, value in cell_blocks.items()
            if key in wanted_keys
        }
    if not selected:
        raise SystemExit(
            f"no manifest cells match blocks {list(blocks)} and methods "
            f"{list(methods)}; the frozen manifest covers "
            f"{list(manifest['blocks'])}"
        )

    pilot_path = Path(arguments.cost_pilot)
    if not pilot_path.is_absolute():
        pilot_path = ROOT / pilot_path
    measurements = load_measurements(pilot_path)
    model = RoughCostModel(measurements)
    if measurements:
        try:
            pilot_label = pilot_path.relative_to(ROOT)
        except ValueError:
            pilot_label = pilot_path
        cost_description = (
            f"from {pilot_label}, linear in K and M"
        )
    else:
        cost_description = (
            "base estimates scaled by (K/25) * (M/10)^2 from the historical "
            "~200 s at K=25, M=10, dense evaluation included"
        )

    total_seconds = sum(
        model.seconds(
            cell["method"], cell["n_train"], cell["n_grid"], cell["n_particles"]
        )
        for cell in selected
    )
    if arguments.shards:
        shard_count = arguments.shards
        if shard_count > MAX_SHARDS:
            print(
                f"warning: {shard_count} shards exceeds the documented cap of "
                f"{MAX_SHARDS}"
            )
    else:
        shard_count = min(
            MAX_SHARDS,
            max(1, math.ceil(total_seconds / TARGET_SHARD_SECONDS)),
        )
    groups = group_replications(selected)
    shard_count = max(1, min(shard_count, len(groups)))

    bins, loads = allocate(selected, model, shard_count, methods)

    output_directory = Path(arguments.output_dir)
    if not output_directory.is_absolute():
        output_directory = ROOT / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    for stale in output_directory.glob("*.ipynb"):
        stale.unlink()

    archive_base64, source_sha256 = build_source_archive(
        arguments.include_forest
    )
    print(
        f"embedded source archive: {len(archive_base64) / 1e6:.2f} MB base64 "
        f"(sha256 {source_sha256[:12]})"
    )
    print(
        f"selected {len(selected)} cells from blocks {list(blocks)} over "
        f"methods {list(methods)}; rough total {total_seconds / 3600.0:.1f} h"
    )

    rows: list[tuple] = []
    total_written = 0
    for index, (shard_cells, load) in enumerate(zip(bins, loads)):
        slice_document = {
            "contract_id": contract_id,
            "manifest_checksum": manifest["manifest_checksum"],
            "estimator_source_hash": manifest.get("estimator_source_hash"),
            "blocks": list(blocks),
            "cell_blocks": {
                item["cell_key"]: cell_blocks[item["cell_key"]]
                for item in shard_cells
            },
            "method_registry": {
                name: manifest["method_registry"][name]
                for name in sorted({item["method"] for item in shard_cells})
            },
            "cells": shard_cells,
        }
        notebook = build_notebook(
            index=index,
            total=len(bins),
            shard_cells=shard_cells,
            estimated_seconds=load,
            cost_description=cost_description,
            archive_base64=archive_base64,
            source_sha256=source_sha256,
            slice_document=slice_document,
            method_order=methods,
            block_names=blocks,
            include_forest=arguments.include_forest,
        )
        path = output_directory / f"wcf_sensitivity_shard_{index:02d}.ipynb"
        write_notebook(path, notebook)
        total_written += len(shard_cells)
        rows.append(
            (
                index,
                len(shard_cells),
                load,
                sorted({item["method"] for item in shard_cells}),
            )
        )
        print(
            f"  {path.name:36s} {len(shard_cells):3d} cells  "
            f"{load / 60.0:7.1f} min  {path.stat().st_size / 1e6:5.2f} MB"
        )

    if total_written != len(selected):
        raise SystemExit(
            f"allocated {total_written} cells but selected {len(selected)}"
        )

    readme = output_directory / "README.md"
    readme.write_text(
        readme_text(
            rows,
            len(selected),
            len(bins),
            cost_description,
            methods,
            arguments.include_forest,
        ),
        encoding="utf-8",
    )
    print(
        f"\n{len(bins)} notebooks, {total_written} cells; wrote {readme}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
