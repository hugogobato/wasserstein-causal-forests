#!/usr/bin/env python3
"""Generate self-contained Colab shards for sample-size sensitivity.

Run this after freezing the sample-size manifest::

    rtk python3 research/checks/wcf_sample_size_make_colab_notebooks.py \
        --shards 16

The generated notebooks embed the exact source tree and their manifest slice,
pin numerical libraries to one thread, checkpoint one parquet after every
cell, and download one zip bundle. Replications are never split between
notebooks, so methods remain paired and the oracle-truth cache is reused.

The forest baselines are included by default when selected. In that case each
notebook also contains the existing pinned R setup cell. The local runner is
the primary execution path; these notebooks are independent overflow shards.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.checks.wcf_sensitivity_make_colab_notebooks import (  # noqa: E402
    FOREST_ENVIRONMENT,
    FOREST_METHODS,
    FOREST_SETUP,
    RoughCostModel,
    build_source_archive,
    code,
    load_measurements,
    markdown,
    setup_cell,
)
from research.run_wcf_sample_size_sensitivity import (  # noqa: E402
    DEFAULT_METHODS,
    MANIFEST_PATH,
)

DEFAULT_OUTPUT_DIRECTORY = ROOT / "colab" / "wcf_sample_size_sensitivity_shards"
TARGET_SHARD_SECONDS = 1800.0
MAX_SHARDS = 64
REFERENCE_N_TRAIN = 1000
REFERENCE_N_GRID = 25
REFERENCE_N_PARTICLES = 10
REFERENCE_SECONDS = {
    "cwdb_dr": 200.0,
    "causal_drf": 40.0,
    "drf": 40.0,
}


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"manifest not found at {path}; run the sample-size freeze command first"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "manifest_contract_id",
        "manifest_checksum",
        "estimator_source_hash",
        "method_registry",
        "cells",
    ):
        if field not in document:
            raise SystemExit(f"manifest is missing {field!r}")
    return document


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


def _base_seconds(method: str, n_train: int, n_grid: int, n_particles: int) -> float:
    return REFERENCE_SECONDS.get(method, 200.0) * (
        n_train / REFERENCE_N_TRAIN
    ) * (n_grid / REFERENCE_N_GRID) * (n_particles / REFERENCE_N_PARTICLES) ** 2


def estimate_seconds(model: RoughCostModel, cell: dict) -> float:
    try:
        return float(
            model.seconds(
                cell["method"],
                cell["n_train"],
                cell["n_grid"],
                cell["n_particles"],
            )
        )
    except (KeyError, TypeError, ValueError):
        return max(
            30.0,
            _base_seconds(
                cell["method"],
                cell["n_train"],
                cell["n_grid"],
                cell["n_particles"],
            ),
        )


def allocate(
    cells: list[dict],
    model: RoughCostModel,
    shard_count: int,
    method_order: tuple[str, ...],
) -> tuple[list[list[dict]], list[float]]:
    groups = group_replications(cells)

    def group_cost(group: list[dict]) -> float:
        return sum(estimate_seconds(model, cell) for cell in group)

    groups.sort(key=group_cost, reverse=True)
    shards: list[list[dict]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    order = {name: index for index, name in enumerate(method_order)}
    for group in groups:
        index = loads.index(min(loads))
        shards[index].extend(group)
        loads[index] += group_cost(group)
    for shard in shards:
        shard.sort(key=lambda cell: (replication_key(cell), order[cell["method"]]))
    return shards, loads


def chunk_base64(payload: str, width: int = 96) -> str:
    return "\n".join(payload[index : index + width] for index in range(0, len(payload), width))


def source_setup(archive_base64: str, archive_sha256: str, checksum: str) -> str:
    return (
        "import base64\n"
        "import pathlib\n"
        "import sys\n"
        "import tempfile\n"
        "import zipfile\n\n"
        f"SOURCE_ARCHIVE_SHA256 = {archive_sha256!r}\n"
        "SOURCE_ARCHIVE_B64 = '''\\\n"
        + chunk_base64(archive_base64)
        + "\n'''\n"
        "workdir = pathlib.Path(tempfile.mkdtemp(prefix='wcf_sample_size_'))\n"
        "archive_path = workdir / 'wcf_source.zip'\n"
        "archive_path.write_bytes(base64.b64decode(SOURCE_ARCHIVE_B64))\n"
        "actual = __import__('hashlib').sha256(archive_path.read_bytes()).hexdigest()\n"
        "assert actual == SOURCE_ARCHIVE_SHA256, (actual, SOURCE_ARCHIVE_SHA256)\n"
        "with zipfile.ZipFile(archive_path) as archive:\n"
        "    archive.extractall(workdir)\n"
        "sys.path.insert(0, str(workdir / 'src'))\n"
        "print('embedded source:', (workdir / 'src').resolve())\n"
        f"print('manifest checksum: {checksum}')\n"
    )


THREAD_PIN_CELL = """\
import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'R_NUM_THREADS'):
    os.environ[_variable] = '1'
print('numerical libraries pinned to one thread')
"""


def registration_cell(index: int, total: int, slice_document: dict, cost: float) -> str:
    return (
        "import json\n"
        "from wasserstein_causal_forests.g3 import runner\n"
        "from wasserstein_causal_forests.g3.wcf_sensitivity import apply_method_registry\n\n"
        f"SHARD_INDEX = {index}\nSHARD_TOTAL = {total}\nESTIMATED_SECONDS = {cost!r}\n"
        "MANIFEST_SLICE = json.loads('''"
        + json.dumps(slice_document)
        + "''')\n"
        "apply_method_registry({'method_registry': MANIFEST_SLICE['method_registry']})\n"
        "print('registered methods:', sorted(MANIFEST_SLICE['method_registry']))\n"
        "print('cells:', len(MANIFEST_SLICE['cells']))\n"
    )


RUN_CELL = r"""
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
from wasserstein_causal_forests.g3 import runner
from wasserstein_causal_forests.g3.manifest import Cell

OUTPUT = Path('shard_output')
OUTPUT.mkdir(parents=True, exist_ok=True)
PARQUET = OUTPUT / 'wcf_sample_size_parquet.parquet'
LOG = OUTPUT / 'execution_log.jsonl'
FAILURES = OUTPUT / 'failure_rows.jsonl'
CACHE = OUTPUT / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)

def read_rows(path):
    return pq.read_table(path).to_pylist() if path.exists() else []

rows = read_rows(PARQUET)
by_cell = {}
for row in rows:
    by_cell.setdefault(row['cell_key'], []).append(row)
successful = {key for key, group in by_cell.items()
              if not any(row.get('metric') == 'cell_failure' for row in group)}
failed = {key for key, group in by_cell.items()
          if any(row.get('metric') == 'cell_failure' for row in group)}
if failed:
    rows = [row for row in rows if row['cell_key'] not in failed]
    print('retrying failed cells:', len(failed))

started = time.time()
n_ok = n_failed = n_skipped = 0
total_cells = len(MANIFEST_SLICE['cells'])
for position, item in enumerate(MANIFEST_SLICE['cells'], start=1):
    if item['cell_key'] in successful:
        n_skipped += 1
        continue
    cell = Cell(**{key: value for key, value in item.items()
                   if key not in ('cell_key', 'test_seed')})
    assert cell.key == item['cell_key']
    cell_rows = runner.run_cell(cell, cache_directory=CACHE,
                                manifest_contract_id=MANIFEST_SLICE['manifest_contract_id'])
    rows.extend(cell_rows)
    status = 'failed' if cell_rows[0].get('status') == 'failed' else 'ok'
    n_ok += status == 'ok'
    n_failed += status == 'failed'
    temporary = PARQUET.with_suffix('.tmp')
    runner.write_rows(rows, temporary)
    os.replace(temporary, PARQUET)
    with LOG.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'cell_key': cell.key, 'status': status,
                                 'n_rows': len(cell_rows),
                                 'wall_seconds': float(cell_rows[0].get('wall_seconds', 0.0)),
                                 'finished_at': time.time()}) + '\n')
    if status == 'failed':
        with FAILURES.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'cell_key': cell.key, 'rows': cell_rows}, default=str) + '\n')
    print(f'[{position}/{total_cells}] {cell.dgp} n={cell.n_train} '
          f'{cell.method} seed={cell.seed}: {status}', flush=True)
print(f'finished: {n_ok} ok, {n_failed} failed, {n_skipped} skipped; '
      f'{(time.time() - started) / 60.0:.1f} minutes')
"""


def diagnostics_cell(source_sha256: str, slice_document: dict, index: int, total: int) -> str:
    prefix = (
        "import json\n"
        "import platform\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n\n"
        "versions = {'python': sys.version.split()[0], 'platform': platform.platform()}\n"
        "for _name in ('numpy', 'scipy', 'sklearn', 'pandas', 'pyarrow'):\n"
        "    try:\n"
        "        _module = __import__(_name)\n"
        "        versions[_name] = getattr(_module, '__version__', 'unknown')\n"
        "    except ImportError:\n"
        "        versions[_name] = None\n"
    )
    config_line = (
        "config = {'shard_index': "
        + str(index)
        + ", 'shard_total': "
        + str(total)
        + ", 'manifest_contract_id': MANIFEST_SLICE['manifest_contract_id'], "
        + "'manifest_checksum': MANIFEST_SLICE['manifest_checksum'], "
        + "'source_archive_sha256': "
        + repr(source_sha256)
        + ", 'n_cells': len(MANIFEST_SLICE['cells']), 'versions': versions, "
        + "'estimated_seconds_reference': ESTIMATED_SECONDS}\n"
    )
    suffix = (
        "out = Path('shard_output')\n"
        "(out / 'shard_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')\n"
        "(out / 'manifest_slice.json').write_text(json.dumps(MANIFEST_SLICE, indent=2), encoding='utf-8')\n"
        "(out / 'wcf_sample_size_parquet.meta.json').write_text(json.dumps({"
        "'manifest_checksum': MANIFEST_SLICE['manifest_checksum'], "
        "'estimator_source_hash': MANIFEST_SLICE['estimator_source_hash'], "
        "'contract_id': MANIFEST_SLICE['manifest_contract_id'], "
        "'updated_at': time.time()}, indent=2), encoding='utf-8')\n"
        "print(json.dumps(config, indent=2))\n"
    )
    return prefix + config_line + suffix


DOWNLOAD_CELL = """\
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

output_file = 'wcf_sample_size_shard.zip'
with ZipFile(output_file, 'w', ZIP_DEFLATED) as archive:
    for path in Path('shard_output').rglob('*'):
        if path.is_file():
            archive.write(path, arcname=path.relative_to('shard_output'))
try:
    from google.colab import files
    files.download(output_file)
    print('Downloaded:', output_file)
except Exception as e:
    print('(Not on Colab / download skipped):', e)
"""


def build_notebook(
    index: int,
    total: int,
    shard: list[dict],
    cost: float,
    slice_document: dict,
    archive_base64: str,
    archive_sha256: str,
    include_forest: bool,
) -> dict:
    methods = sorted({cell["method"] for cell in shard})
    sizes = sorted({cell["n_train"] for cell in shard})
    dgps = sorted({cell["dgp"] for cell in shard})
    display = (
        f"# WCF sample-size sensitivity shard {index:02d}\n\n"
        f"Shard {index} of {total} contains {len(shard)} cells, methods "
        f"{', '.join('`' + value + '`' for value in methods)}, source DGPs "
        f"{', '.join(dgps)}, and sample sizes {sizes}.\n\n"
        f"The rough reference cost is {cost / 60.0:.0f} minutes. The estimate "
        "is only for balancing; Colab is usually slower. This notebook embeds "
        "its source and manifest slice, checkpoints after each cell, and ends "
        "by downloading one zip bundle.\n"
    )
    cells = [
        markdown(display),
        code(THREAD_PIN_CELL),
        markdown("## Embedded source"),
        code(source_setup(archive_base64, archive_sha256, slice_document["manifest_checksum"])),
    ]
    if include_forest:
        cells.extend([
            markdown("## R forest dependencies\n\nThis setup is needed for the Causal-DRF and DRF comparator cells."),
            code(FOREST_ENVIRONMENT),
            code(FOREST_SETUP),
        ])
    cells.extend([
        markdown("## Manifest registration"),
        code(registration_cell(index, total, slice_document, cost)),
        markdown("## Run\n\nRe-run this cell after interruption to resume from the checkpoint."),
        code(RUN_CELL),
        markdown("## Diagnostics and sidecar"),
        code(diagnostics_cell(archive_sha256, slice_document, index, total)),
        markdown("## Download"),
        code(DOWNLOAD_CELL),
    ])
    return {
        "cells": [
            {**cell, "source": cell["source"].splitlines(keepends=True)}
            for cell in cells
        ],
        "metadata": {
            "colab": {"provenance": [], "name": f"wcf_sample_size_shard_{index:02d}"},
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


def write_readme(
    path: Path,
    rows: list[tuple[int, int, float, tuple[str, ...]]],
    total_cells: int,
    total_hours: float,
    cost_description: str,
) -> None:
    lines = [
        "# WCF sample-size sensitivity Colab shards",
        "",
        f"These {len(rows)} self-contained notebooks cover {total_cells} cells "
        f"exactly once. The rough reference total is {total_hours:.1f} hours "
        f"({cost_description}).",
        "",
        "Upload one notebook per Colab session, choose Run all, and wait for "
        "the run cell to finish. Numerical libraries are pinned to one thread. "
        "The run cell checkpoints one parquet after every cell; rerunning it "
        "resumes successful cells and retries failed cells under the same seed. "
        "The final cell downloads exactly one `wcf_sample_size_shard.zip`.",
        "",
        "| Notebook | Cells | Estimated minutes | Methods |",
        "|---|---:|---:|---|",
    ]
    for index, count, seconds, methods in rows:
        lines.append(
            f"| `wcf_sample_size_shard_{index:02d}.ipynb` | {count} | "
            f"{seconds / 60.0:.0f} | {', '.join(methods)} |"
        )
    lines.extend([
        "",
        "Each bundle contains `wcf_sample_size_parquet.parquet`, its sidecar, "
        "the manifest slice, execution logs, failure rows when present, and "
        "dependency metadata. Rename the parquet and sidecar to "
        "`shard_colab_<index>.parquet` and "
        "`shard_colab_<index>.meta.json` in "
        "`results/wcf_sample_size_sensitivity/shards/`, then run:",
        "",
        "```bash",
        "rtk python3 research/run_wcf_sample_size_sensitivity.py merge",
        "```",
        "",
        "Do not edit the parquet, manifest slice, sidecar, or logs. The local "
        "merge checks the contract, source hash, manifest checksum, and cell "
        "keys. Failed cells remain explicit rather than being silently dropped.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--shards", type=int, default=0)
    parser.add_argument(
        "--methods",
        default="",
        help="comma-separated methods; empty uses every method in the manifest",
    )
    parser.add_argument("--cost-pilot", default="results/wcf_sensitivity/cost_pilot.json")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)

    manifest_path = Path(arguments.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = load_manifest(manifest_path)
    methods = tuple(
        token.strip() for token in arguments.methods.split(",") if token.strip()
    )
    if not methods:
        methods = tuple(manifest.get("methods", DEFAULT_METHODS))
    if not methods:
        raise SystemExit("--methods must contain at least one method")
    missing = sorted(set(methods) - set(manifest["method_registry"]))
    if missing:
        raise SystemExit(f"methods {missing} are absent from the frozen manifest")
    selected = [cell for cell in manifest["cells"] if cell["method"] in methods]
    if not selected:
        raise SystemExit("no manifest cells match --methods")

    pilot_path = Path(arguments.cost_pilot)
    if not pilot_path.is_absolute():
        pilot_path = ROOT / pilot_path
    measurements = load_measurements(pilot_path)
    model = RoughCostModel(measurements)
    cost_description = "cost pilot" if measurements else "base estimates"
    total_seconds = sum(estimate_seconds(model, cell) for cell in selected)
    if arguments.shards:
        shard_count = arguments.shards
    else:
        shard_count = max(1, math.ceil(total_seconds / TARGET_SHARD_SECONDS))
    if shard_count > MAX_SHARDS:
        print(f"warning: capping requested shard count at {MAX_SHARDS}")
        shard_count = MAX_SHARDS
    shard_count = min(shard_count, len(group_replications(selected)))
    if shard_count < 1:
        raise SystemExit("cannot allocate zero shards")
    shards, loads = allocate(selected, model, shard_count, methods)

    output_directory = Path(arguments.output_dir)
    if not output_directory.is_absolute():
        output_directory = ROOT / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    existing = list(output_directory.glob("*.ipynb"))
    if existing and not arguments.force:
        raise SystemExit(
            f"{output_directory} already contains notebooks; choose a new directory "
            "or pass --force after preserving the old shards"
        )
    if existing and arguments.force:
        for path in existing:
            path.unlink()

    include_forest = any(method in FOREST_METHODS for method in methods)
    archive_base64, archive_sha256 = build_source_archive(include_forest)
    rows: list[tuple[int, int, float, tuple[str, ...]]] = []
    for index, (shard, load) in enumerate(zip(shards, loads)):
        slice_document = {
            "manifest_contract_id": manifest["manifest_contract_id"],
            "manifest_checksum": manifest["manifest_checksum"],
            "estimator_source_hash": manifest["estimator_source_hash"],
            "method_registry": manifest["method_registry"],
            "cells": shard,
        }
        notebook = build_notebook(
            index,
            shard_count,
            shard,
            load,
            slice_document,
            archive_base64,
            archive_sha256,
            include_forest,
        )
        path = output_directory / f"wcf_sample_size_shard_{index:02d}.ipynb"
        write_notebook(path, notebook)
        rows.append((index, len(shard), load, tuple(sorted({cell["method"] for cell in shard}))))
    write_readme(
        output_directory / "README.md",
        rows,
        len(selected),
        total_seconds / 3600.0,
        cost_description,
    )
    print(f"wrote {len(rows)} notebooks to {output_directory}")
    print(f"selected cells: {len(selected)}; rough reference cost: {total_seconds / 3600.0:.1f} hours")
    print(f"embedded source archive: {len(archive_base64) / 1e6:.2f} MB base64 ({archive_sha256[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
