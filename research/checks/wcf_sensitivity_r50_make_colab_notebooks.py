#!/usr/bin/env python3
"""Generate self-contained Colab notebooks for the r50 sensitivity rerun.

The notebooks mirror the confirmatory shard set: pinned Python stack, embedded
source archive, verified authors' Causal-DRF repository, one manifest slice per
shard, per-cell checkpointing in a child process, coverage validation, and one
ZIP download per shard.  The runner embedded here is
``research/run_wcf_sensitivity_r50.py``, which evaluates every sensitivity cell
with the confirmatory protocol.

Usage::

    python3 research/checks/wcf_sensitivity_r50_make_colab_notebooks.py \
        --output-dir colab/wcf_sensitivity_r50_120_shards
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.run_wcf_sensitivity_r50 import (  # noqa: E402
    CONTRACT_ID,
    load_manifest,
)
from research.checks.wcf_confirmatory_make_colab_notebooks import (  # noqa: E402
    FOREST_SETUP_PINNED,
    PYTHON_SETUP,
    THREADS,
    causal_drf_paper_pin,
)
from research.checks.wcf_sensitivity_make_colab_notebooks import (  # noqa: E402
    FOREST_ENVIRONMENT,
    code,
    markdown,
)

TARGET_SHARD_SECONDS = 25_200.0
MINIMUM_SECONDS = 30.0
MAXIMUM_SHARDS = 160

#: Conservative single-threaded reference seconds at n_train=1000, K=25, M=10,
#: with the confirmatory evaluation included.  The WCF anchors are the measured
#: sensitivity costs scaled by the confirmatory-to-native evaluation ratio; the
#: forest anchors are the confirmatory constants.  The K and M factors are
#: conservative piecewise factors around the measured anchors, chosen to
#: over- rather than under-estimate: a shard that runs long loses its Colab
#: session.
BASE_SECONDS: dict[str, float] = {
    "cwdb_dr": 820.0,
    "cwdb_dr_flex": 900.0,
    "cwdb_dr_oracle": 700.0,
    "cwdb_dr_flex_rf": 1500.0,
    "causal_drf": 135.0,
    "drf": 125.0,
}
K_FACTOR = {5: 0.45, 25: 1.0, 49: 1.05}
M_FACTOR = {5: 0.60, 10: 1.0, 25: 2.90}
K_FACTOR_FOREST = {5: 0.75, 25: 1.0, 49: 1.35}


def cell_seconds(method: str, n_train: int, n_grid: int, n_particles: int) -> float:
    base = BASE_SECONDS.get(method)
    if base is None:
        raise SystemExit(f"no cost anchor for method {method!r}")
    scale = n_train / 1000.0
    if method in ("causal_drf", "drf"):
        return max(MINIMUM_SECONDS, base * scale * K_FACTOR_FOREST[n_grid])
    return max(
        MINIMUM_SECONDS,
        base * scale * K_FACTOR[n_grid] * M_FACTOR[n_particles],
    )


def group_seconds(group: list[dict]) -> float:
    return sum(
        cell_seconds(
            item["method"], item["n_train"], item["n_grid"], item["n_particles"]
        )
        for item in group
    )


def replication_key(item: dict) -> tuple:
    return (
        item["grid"],
        item["dgp"],
        item["n_train"],
        item["n_grid"],
        item["n_particles"],
        item["seed"],
    )


def group_replications(cells: list[dict]) -> list[list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for item in cells:
        groups[replication_key(item)].append(item)
    return list(groups.values())


def allocate(items: list[dict], n_shards: int) -> tuple[list[list[dict]], list[float]]:
    groups = sorted(group_replications(items), key=group_seconds, reverse=True)
    order = {
        name: position
        for position, name in enumerate(
            (
                "cwdb_dr",
                "cwdb_dr_flex",
                "cwdb_dr_oracle",
                "cwdb_dr_flex_rf",
                "causal_drf",
                "drf",
            )
        )
    }
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards
    for group in groups:
        index = loads.index(min(loads))
        shards[index].extend(group)
        loads[index] += group_seconds(group)
    for shard in shards:
        shard.sort(key=lambda item: (replication_key(item), order[item["method"]]))
    return shards, loads


def source_archive() -> tuple[str, str]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        roots = [
            ROOT / "src" / "wasserstein_causal_forests",
            ROOT / "research" / "baselines",
            ROOT / "code" / "drfinference-main",
        ]
        for directory in roots:
            for path in sorted(directory.rglob("*")) if directory.exists() else ():
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                if "results" in path.relative_to(directory).parts:
                    continue
                if path.name.endswith(":Zone.Identifier"):
                    continue
                archive.write(path, path.relative_to(ROOT).as_posix())
        runner = ROOT / "research" / "run_wcf_sensitivity_r50.py"
        archive.write(runner, runner.relative_to(ROOT).as_posix())
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), hashlib.sha256(payload).hexdigest()


def setup_source(encoded: str, source_sha: str, manifest_sha: str) -> str:
    chunks = "\n".join(encoded[i:i + 96] for i in range(0, len(encoded), 96))
    return f"""\
import base64, hashlib, os, pathlib, sys, tempfile, zipfile
SOURCE_ARCHIVE_SHA256 = {source_sha!r}
SOURCE_ARCHIVE_B64 = '''\\
{chunks}
'''
workdir = pathlib.Path(tempfile.mkdtemp(prefix='wcf_sensitivity_r50_'))
archive_path = workdir / 'source.zip'
archive_path.write_bytes(base64.b64decode(SOURCE_ARCHIVE_B64))
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == SOURCE_ARCHIVE_SHA256
with zipfile.ZipFile(archive_path) as archive:
    archive.extractall(workdir)
sys.path[:0] = [str(workdir), str(workdir / 'src')]
os.environ['WCF_SOURCE_ROOT'] = str(workdir)
print('source archive:', SOURCE_ARCHIVE_SHA256)
print('manifest:', {manifest_sha!r})
"""


def registration(index: int, total: int, shard: list[dict], manifest: dict, load: float) -> str:
    slice_doc = {
        "manifest_contract_id": manifest["manifest_contract_id"],
        "manifest_checksum": manifest["manifest_checksum"],
        "estimator_source_hash": manifest["estimator_source_hash"],
        "evaluation_protocol_id": manifest["evaluation_protocol_id"],
        "method_registry": manifest["method_registry"],
        "cells": shard,
    }
    return f"""\
import json
from pathlib import Path
SHARD_INDEX = {index}
SHARD_TOTAL = {total}
ESTIMATED_REFERENCE_SECONDS = {load!r}
MANIFEST_SLICE = json.loads('''{json.dumps(slice_doc)}''')
payload = {{
    'shard_index': SHARD_INDEX,
    'shard_total': SHARD_TOTAL,
    'manifest_slice': MANIFEST_SLICE,
    'source_archive_sha256': SOURCE_ARCHIVE_SHA256,
    'causal_drf_paper_repository': CAUSAL_DRF_PAPER_REPOSITORY,
    'causal_drf_paper_commit': CAUSAL_DRF_PAPER_COMMIT,
    'causal_drf_paper_files': dict(CAUSAL_DRF_PAPER_FILES),
    'causal_clean_drf_commit': CAUSAL_CLEAN_DRF_COMMIT,
}}
output = Path('shard_output')
output.mkdir(exist_ok=True)
Path('wcf_sensitivity_r50_payload.json').write_text(json.dumps(payload), encoding='utf-8')
print('shard', SHARD_INDEX, 'of', SHARD_TOTAL, '| cells', len(MANIFEST_SLICE['cells']))
"""


def _launcher(worker_name: str, worker_source: str) -> str:
    return f"""\
import os, subprocess, sys
from pathlib import Path
WORKER_SOURCE = {worker_source!r}
worker = Path(os.environ['WCF_SOURCE_ROOT']) / {worker_name!r}
worker.write_text(WORKER_SOURCE, encoding='utf-8')
environment = dict(os.environ)
environment['PYTHONUNBUFFERED'] = '1'
process = subprocess.Popen(
    [sys.executable, str(worker)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, env=environment,
)
try:
    for line in process.stdout:
        print(line, end='', flush=True)
    status = process.wait()
except KeyboardInterrupt:
    process.terminate()
    process.wait()
    raise
assert status == 0, f'{worker_name} exited with status ' + str(status)
print('{worker_name} finished cleanly')
"""


RUN_WORKER = r"""\
import gc, json, os, sys, time
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]

import pyarrow.parquet as pq
from research.run_wcf_sensitivity_r50 import run_cell
from wasserstein_causal_forests.g3.manifest import Cell
from wasserstein_causal_forests.g3.sensitivity_dgps import register_sensitivity_dgps
from wasserstein_causal_forests.g3.sensitivity_methods import register_sensitivity_methods
from wasserstein_causal_forests.g3.wcf_sensitivity import apply_method_registry

PAYLOAD = json.loads(Path('wcf_sensitivity_r50_payload.json').read_text(encoding='utf-8'))
MANIFEST_SLICE = PAYLOAD['manifest_slice']
SHARD_INDEX = PAYLOAD['shard_index']
register_sensitivity_dgps()
register_sensitivity_methods()
apply_method_registry({'method_registry': MANIFEST_SLICE['method_registry']})

OUTPUT = Path('shard_output')
OUTPUT.mkdir(exist_ok=True)
PARQUET = OUTPUT / 'sensitivity_r50_results.parquet'
LOG = OUTPUT / 'execution_log.jsonl'
CACHE = Path('/tmp/wcf_sensitivity_r50_cache') / f'{SHARD_INDEX:02d}'
CACHE.mkdir(parents=True, exist_ok=True)

rows = pq.read_table(PARQUET).to_pylist() if PARQUET.exists() else []
groups = {}
for row in rows:
    groups.setdefault(row['cell_key'], []).append(row)
successful = {key for key, values in groups.items()
              if not any(row['status'] == 'failed' for row in values)}
failed = set(groups) - successful
if failed:
    rows = [row for row in rows if row['cell_key'] not in failed]
started = time.time()
for position, item in enumerate(MANIFEST_SLICE['cells'], 1):
    if item['cell_key'] in successful:
        continue
    cell = Cell(**{key: value for key, value in item.items()
                   if key not in ('cell_key', 'test_seed')})
    cell_rows = run_cell(cell, cache_directory=CACHE)
    rows.extend(cell_rows)
    temporary = PARQUET.with_suffix('.tmp')
    from wasserstein_causal_forests.g3.runner import write_rows
    write_rows(rows, temporary)
    os.replace(temporary, PARQUET)
    with LOG.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'cell_key': cell.key,
                                 'status': cell_rows[0]['status'],
                                 'n_rows': len(cell_rows),
                                 'wall_seconds': cell_rows[0]['wall_seconds'],
                                 'finished_at': time.time()}) + '\n')
    print(f'[{position}/{len(MANIFEST_SLICE["cells"])}] {cell.grid} {cell.dgp} '
          f'K={cell.n_grid} M={cell.n_particles} n={cell.n_train} {cell.method} '
          f'seed={cell.seed}: {cell_rows[0]["status"]}', flush=True)
    del cell_rows
    gc.collect()
print('elapsed hours:', round((time.time() - started) / 3600, 3))
"""


FINALIZE_WORKER = r"""\
import hashlib, json, platform, subprocess, sys, time
from pathlib import Path

import pyarrow.parquet as pq

PAYLOAD = json.loads(Path('wcf_sensitivity_r50_payload.json').read_text(encoding='utf-8'))
MANIFEST_SLICE = PAYLOAD['manifest_slice']
SHARD_INDEX = PAYLOAD['shard_index']
SHARD_TOTAL = PAYLOAD['shard_total']
SOURCE_ARCHIVE_SHA256 = PAYLOAD['source_archive_sha256']

out = Path('shard_output')
parquet = out / 'sensitivity_r50_results.parquet'
frame = pq.read_table(parquet).to_pandas()
expected = {item['cell_key'] for item in MANIFEST_SLICE['cells']}
observed = set(frame.cell_key)
assert observed == expected, (len(expected - observed), len(observed - expected))
versions = {'python': sys.version, 'platform': platform.platform()}
for name in ('numpy','scipy','sklearn','pandas','pyarrow'):
    module = __import__(name)
    versions[name] = module.__version__
versions['R'] = subprocess.run(['Rscript','-e','cat(R.version.string)'], capture_output=True, text=True, check=True).stdout
versions['cran_drf'] = subprocess.run(['Rscript','-e','cat(as.character(packageVersion("drf")))'], capture_output=True, text=True, check=True).stdout
config = {'shard_index': SHARD_INDEX, 'shard_total': SHARD_TOTAL,
          'manifest_contract_id': MANIFEST_SLICE['manifest_contract_id'],
          'manifest_checksum': MANIFEST_SLICE['manifest_checksum'],
          'estimator_source_hash': MANIFEST_SLICE['estimator_source_hash'],
          'evaluation_protocol_id': MANIFEST_SLICE['evaluation_protocol_id'],
          'source_archive_sha256': SOURCE_ARCHIVE_SHA256, 'n_cells': len(expected),
          'causal_drf_paper_repository': PAYLOAD['causal_drf_paper_repository'],
          'causal_drf_paper_commit': PAYLOAD['causal_drf_paper_commit'],
          'causal_drf_paper_files': PAYLOAD['causal_drf_paper_files'],
          'causal_clean_drf_commit': PAYLOAD['causal_clean_drf_commit'],
          'n_failed': int(frame.loc[frame.status == 'failed', 'cell_key'].nunique()),
          'versions': versions, 'completed_at': time.time()}
(out / 'manifest_slice.json').write_text(json.dumps(MANIFEST_SLICE, indent=2), encoding='utf-8')
(out / 'completion.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
inventory = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in out.iterdir() if path.is_file()}
(out / 'sha256_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
print(json.dumps(config, indent=2))
"""


def run_cell_source() -> str:
    return _launcher('run_shard_worker.py', RUN_WORKER)


def finalize_cell_source() -> str:
    return _launcher('finalize_shard_worker.py', FINALIZE_WORKER)


def download(index: int, manifest_sha: str) -> str:
    short = manifest_sha[:12]
    return f"""\
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
output_file = 'wcf_sensitivity_r50_shard_{index:02d}_{short}.zip'
with ZipFile(output_file, 'w', ZIP_DEFLATED) as archive:
    for path in Path('shard_output').iterdir():
        if path.is_file():
            archive.write(path, path.name)
try:
    from google.colab import files
    files.download(output_file)
    print('Downloaded:', output_file)
except Exception as e:
    print('(Not on Colab / download skipped):', e)
"""


def notebook(index, total, shard, manifest, load, encoded, source_sha):
    title = (
        f"# WCF sensitivity r50 rerun, shard {index:02d} of {total}\n\n"
        f"This shard contains {len(shard)} estimator cells grouped into "
        f"{len(group_replications(shard))} paired replications of the frozen "
        f"sensitivity roster. Its conservative reference estimate is "
        f"{load / 3600:.2f} hours; Colab is slower than the reference machine, "
        "so run it in a fresh session and let the checkpoint cell resume it if "
        "the session ends. Run all cells."
    )
    cells = [
        markdown(title), code(THREADS), code(PYTHON_SETUP),
        code(setup_source(encoded, source_sha, manifest["manifest_checksum"])),
        code(FOREST_ENVIRONMENT), code(FOREST_SETUP_PINNED),
        code(causal_drf_paper_pin()),
        code(registration(index, total, shard, manifest, load)),
        code(run_cell_source()), code(finalize_cell_source()),
        code(download(index, manifest["manifest_checksum"])),
    ]
    return {
        "cells": [{**cell, "source": cell["source"].splitlines(keepends=True)} for cell in cells],
        "metadata": {"colab": {"name": f"wcf_sensitivity_r50_shard_{index:02d}"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "colab" / "wcf_sensitivity_r50_shards")
    )
    parser.add_argument("--shards", type=int, default=0)
    args = parser.parse_args()

    manifest = load_manifest()
    cells = manifest["cells"]
    if not cells:
        raise SystemExit("manifest has no cells")

    if args.shards:
        shard_count = args.shards
    else:
        total = sum(group_seconds(group) for group in group_replications(cells))
        shard_count = min(MAXIMUM_SHARDS, max(1, math.ceil(total / TARGET_SHARD_SECONDS)))
    groups = len(group_replications(cells))
    if shard_count < 1 or shard_count > groups:
        raise SystemExit("invalid shard count")

    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.ipynb")):
        raise SystemExit(f"refusing to overwrite notebooks in {output}")

    shards, loads = allocate(cells, shard_count)
    encoded, source_sha = source_archive()
    records = []
    for index, (shard, load) in enumerate(zip(shards, loads)):
        path = output / f"wcf_sensitivity_r50_shard_{index:02d}.ipynb"
        path.write_text(
            json.dumps(notebook(index, shard_count, shard, manifest, load, encoded, source_sha), indent=1),
            encoding="utf-8",
        )
        records.append({
            "index": index,
            "notebook": path.name,
            "cells": len(shard),
            "replications": len(group_replications(shard)),
            "estimated_seconds": load,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    generation = {
        "manifest_contract_id": CONTRACT_ID,
        "manifest_checksum": manifest["manifest_checksum"],
        "estimator_source_hash": manifest["estimator_source_hash"],
        "evaluation_protocol_id": manifest["evaluation_protocol_id"],
        "source_archive_sha256": source_sha,
        "n_shards": shard_count,
        "n_cells": sum(record["cells"] for record in records),
        "n_replications": len(group_replications(cells)),
        "target_shard_seconds": TARGET_SHARD_SECONDS,
        "notebooks": records,
    }
    (output / "notebook_generation.json").write_text(
        json.dumps(generation, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# WCF sensitivity r50 Colab shards",
        "",
        f"{shard_count} self-contained notebooks covering all "
        f"{generation['n_cells']} cells and {generation['n_replications']} paired "
        "replications of the frozen sensitivity roster exactly once, each cell "
        "evaluated with the confirmatory protocol.",
        "",
        "Run one notebook per Colab session with **Runtime -> Run all**. The "
        "shard index is fixed in each notebook. Each session installs the pinned "
        "stack, runs the numerical work in a fresh child process, verifies the "
        "embedded source and the authors' Causal-DRF repository, checkpoints after "
        "every cell, validates coverage, writes a hash inventory, and downloads "
        "one ZIP file.",
        "",
        "With 48 concurrent sessions, upload the first 48 notebooks, then upload "
        "the rest as sessions free up. Every shard is independent and stores its "
        "own `shard_output/` checkpoint, so an interrupted session resumes when "
        "the run cell is executed again.",
        "",
        "## Merging",
        "",
        "Unzip every bundle and copy each `sensitivity_r50_results.parquet` into "
        "`results/wcf_sensitivity_r50/shards/` as `shard_colab_<NN>.parquet` "
        "(the `colab_` infix keeps overflow shards from overwriting local numeric "
        "shards), then run:",
        "",
        "```bash",
        "python3 research/run_wcf_sensitivity_r50.py merge",
        "python3 research/run_wcf_sensitivity_r50.py summarize",
        "```",
        "",
        "| Notebook | Cells | Paired replications | Conservative hours |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{record['notebook']}` | {record['cells']} | "
        f"{record['replications']} | {record['estimated_seconds'] / 3600:.2f} |"
        for record in records
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {shard_count} notebooks to {output}")
    print(
        f"load range: {min(loads) / 3600:.2f} to {max(loads) / 3600:.2f} "
        "conservative reference hours"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
