#!/usr/bin/env python3
"""Generate self-contained Colab notebooks for the confirmatory rerun."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.run_wcf_confirmatory import MANIFEST_PATH, load_manifest
from research.checks.wcf_sensitivity_make_colab_notebooks import (
    CAUSAL_CLEAN_COMMIT,
    FOREST_ENVIRONMENT,
    code,
    markdown,
)

DEFAULT_OUTPUT = ROOT / "colab" / "wcf_confirmatory_r50_26_shards"
DEFAULT_SHARDS = 26

#: The decisive Causal-DRF baseline is the authors' `drf` fork at this commit
#: (the head of the `causal-clean` branch), driven through the simulation
#: settings of their paper repository at the commit below.  The notebooks
#: verify both artefacts at run time instead of trusting a prose description.
PAPER_REPOSITORY = "herbps10/causal_drf_paper"
PAPER_COMMIT = "06d156e1f2c17c676000f258ccdf15fc60544384"
PAPER_SETUP_PATH = "R/simulation_study_setup.R"
PAPER_SETUP_SHA256 = (
    "b4b6f94054c3814c7cce272cd66a10bf772a2efbcdeb98ef058e425a5f2607a2"
)
PAPER_STUDY_PATH = "R/simulation_study.R"
PAPER_STUDY_SHA256 = (
    "fbf7ec12a375b169737edaa07e4c3bb62bf5d10bcf56f80a66bbc307a2778a59"
)

# Conservative seconds per full confirmatory cell on the reference machine.
# WCF includes a full fit plus strict nested outer-fold nuisance fits.  Forest
# and ZIPT entries likewise include their three OOF refits.
CELL_SECONDS = {
    "cwdb_dr": 820.0,
    "cwdb_zipt": 180.0,
    "causal_drf": 135.0,
    "drf": 125.0,
}
DGP_MULTIPLIER = {"D6": 1.20, "D7": 1.10, "IC3": 1.15, "ZI3": 1.15}

THREADS = """\
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'R_NUM_THREADS'):
    os.environ[name] = '1'
print('numerical libraries pinned to one thread')
"""

PYTHON_SETUP = """\
%pip -q install numpy==2.4.3 scipy==1.17.1 scikit-learn==1.8.0 pandas==3.0.1 pyarrow==24.0.0
print('Python dependencies installed')
"""

FOREST_SETUP_PINNED = f"""\
%%bash
set -e
apt-get -qq update > /dev/null 2>&1
apt-get -qq install -y r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev curl > /dev/null 2>&1
Rscript -e 'options(Ncpus=2); install.packages(c("Rcpp","RcppEigen","jsonlite","remotes","transport","fastDummies","kernlab"), repos="https://cloud.r-project.org", quiet=TRUE)'
Rscript -e 'options(Ncpus=2); remotes::install_version("drf", version="1.3.1", repos="https://cloud.r-project.org", upgrade="never", quiet=TRUE); stopifnot(as.character(packageVersion("drf")) == "1.3.1")'
mkdir -p /content/Rlib/causal_drf
CAUSAL_SHA="{CAUSAL_CLEAN_COMMIT}"
TARBALL="/tmp/causal_clean_${{CAUSAL_SHA:0:12}}.tar.gz"
EXTRACT="/tmp/causal_clean_${{CAUSAL_SHA:0:12}}"
curl -fsSL "https://codeload.github.com/herbps10/drf/tar.gz/${{CAUSAL_SHA}}" -o "$TARBALL"
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
tar -xzf "$TARBALL" -C "$EXTRACT"
PKG_DIR=$(find "$EXTRACT" -maxdepth 4 -type d -path "*/r-package/drf" | head -1)
test -n "$PKG_DIR"
R CMD INSTALL --library=/content/Rlib/causal_drf "$PKG_DIR"
Rscript -e '.libPaths(c("/content/Rlib/causal_drf",.libPaths())); stopifnot(requireNamespace("drf", quietly=TRUE)); cat("causal-clean", as.character(packageVersion("drf")), "ready\\n")'
echo 'R dependencies installed and validated'
"""


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
        runner = ROOT / "research" / "run_wcf_confirmatory.py"
        archive.write(runner, runner.relative_to(ROOT).as_posix())
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), hashlib.sha256(payload).hexdigest()


def group_cells(items: list[dict]) -> list[list[dict]]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for item in items:
        groups[(item["dgp"], item["seed"])].append(item)
    return list(groups.values())


def cost(group: list[dict]) -> float:
    return DGP_MULTIPLIER.get(group[0]["dgp"], 1.0) * sum(
        CELL_SECONDS[item["method"]] for item in group
    )


def allocate(items: list[dict], n_shards: int) -> tuple[list[list[dict]], list[float]]:
    groups = sorted(group_cells(items), key=cost, reverse=True)
    shards = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards
    order = {name: i for i, name in enumerate(("cwdb_dr", "cwdb_zipt", "causal_drf", "drf"))}
    for group in groups:
        index = loads.index(min(loads))
        shards[index].extend(group)
        loads[index] += cost(group)
    for shard in shards:
        shard.sort(key=lambda x: (x["dgp"], x["seed"], order[x["method"]]))
    return shards, loads


def causal_drf_paper_pin() -> str:
    """Verify the authors' Causal-DRF repository at the pinned commit.

    The baseline itself is the authors' `drf` fork installed in the cell
    above.  This cell pins the simulation repository that documents the fit
    call and the 2500-tree/50-group budget the bridge reproduces, so the
    provenance chain is checked instead of asserted in prose.
    """

    return f"""\
# Pin the authors' Causal-DRF simulation repository. The model is implemented
# by the causal-clean `drf` fork installed above; this cell pins the reference
# repository whose `drf()` call and hyperparameters the driver reproduces.
import hashlib, re, urllib.request
CAUSAL_DRF_PAPER_REPOSITORY = {PAPER_REPOSITORY!r}
CAUSAL_DRF_PAPER_COMMIT = {PAPER_COMMIT!r}
CAUSAL_CLEAN_DRF_COMMIT = {CAUSAL_CLEAN_COMMIT!r}
CAUSAL_DRF_PAPER_FILES = {{
    {PAPER_SETUP_PATH!r}: {PAPER_SETUP_SHA256!r},
    {PAPER_STUDY_PATH!r}: {PAPER_STUDY_SHA256!r},
}}
CAUSAL_DRF_PAPER_BASE = (
    'https://raw.githubusercontent.com/' + CAUSAL_DRF_PAPER_REPOSITORY + '/'
    + CAUSAL_DRF_PAPER_COMMIT + '/'
)
CAUSAL_DRF_PAPER_SOURCES = {{}}
for path, expected in CAUSAL_DRF_PAPER_FILES.items():
    payload = urllib.request.urlopen(
        CAUSAL_DRF_PAPER_BASE + path, timeout=120
    ).read()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == expected, (path, digest, expected)
    CAUSAL_DRF_PAPER_SOURCES[path] = payload.decode('utf-8')
    print('verified', CAUSAL_DRF_PAPER_REPOSITORY + '@' + CAUSAL_DRF_PAPER_COMMIT,
          path, digest)

setup_source = re.sub(r'\\s+', ' ', CAUSAL_DRF_PAPER_SOURCES[{PAPER_SETUP_PATH!r}])
study_source = re.sub(r'\\s+', ' ', CAUSAL_DRF_PAPER_SOURCES[{PAPER_STUDY_PATH!r}])
for fragment in (
    'drf(X, Y, W, num.trees = num_trees, ci.group.size = ci_group_size',
    'response.scaling = FALSE',
):
    assert fragment in setup_source, fragment
for fragment in (
    'dgp = c("nothing", "confounding", "effect", "both")',
    'num_trees = c(50 * 50)',
    'ci_group_size = round(num_trees / 50)',
):
    assert fragment in study_source, fragment
print('Causal-DRF baseline pin verified: causal-clean drf fork at',
      CAUSAL_CLEAN_DRF_COMMIT)
print('the WCF driver reproduces the pinned call with num_trees=2500, '
      'ci.group.size=50, response.scaling=FALSE (seed explicit, one thread)')
"""


def setup_source(encoded: str, source_sha: str, manifest_sha: str) -> str:
    chunks = "\n".join(encoded[i:i + 96] for i in range(0, len(encoded), 96))
    return f"""\
import base64, hashlib, pathlib, sys, tempfile, zipfile
SOURCE_ARCHIVE_SHA256 = {source_sha!r}
SOURCE_ARCHIVE_B64 = '''\\
{chunks}
'''
workdir = pathlib.Path(tempfile.mkdtemp(prefix='wcf_confirmatory_'))
archive_path = workdir / 'source.zip'
archive_path.write_bytes(base64.b64decode(SOURCE_ARCHIVE_B64))
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == SOURCE_ARCHIVE_SHA256
with zipfile.ZipFile(archive_path) as archive:
    archive.extractall(workdir)
sys.path[:0] = [str(workdir), str(workdir / 'src')]
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
from research.run_wcf_confirmatory import run_cell
from wasserstein_causal_forests.g3.wcf_sensitivity import apply_method_registry
SHARD_INDEX = {index}
SHARD_TOTAL = {total}
ESTIMATED_REFERENCE_SECONDS = {load!r}
MANIFEST_SLICE = json.loads('''{json.dumps(slice_doc)}''')
apply_method_registry({{'method_registry': MANIFEST_SLICE['method_registry']}})
print('shard', SHARD_INDEX, 'of', SHARD_TOTAL, '| cells', len(MANIFEST_SLICE['cells']))
"""


RUN = r"""
import gc, json, os, time
from pathlib import Path
import pyarrow.parquet as pq
from wasserstein_causal_forests.g3.manifest import Cell

OUTPUT = Path('shard_output')
OUTPUT.mkdir(exist_ok=True)
PARQUET = OUTPUT / 'confirmatory_results.parquet'
LOG = OUTPUT / 'execution_log.jsonl'
CACHE = Path('/tmp/wcf_confirmatory_cache') / f'{SHARD_INDEX:02d}'
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
    print(f'[{position}/{len(MANIFEST_SLICE["cells"])}] {cell.dgp} '
          f'{cell.method} seed={cell.seed}: {cell_rows[0]["status"]}', flush=True)
    del cell_rows
    gc.collect()
print('elapsed hours:', round((time.time() - started) / 3600, 3))
"""


def finalize(source_sha: str) -> str:
    return f"""\
import hashlib, json, platform, subprocess, sys, time
from pathlib import Path
import pyarrow.parquet as pq
out = Path('shard_output')
parquet = out / 'confirmatory_results.parquet'
frame = pq.read_table(parquet).to_pandas()
expected = {{item['cell_key'] for item in MANIFEST_SLICE['cells']}}
observed = set(frame.cell_key)
assert observed == expected, (len(expected - observed), len(observed - expected))
versions = {{'python': sys.version, 'platform': platform.platform()}}
for name in ('numpy','scipy','sklearn','pandas','pyarrow'):
    module = __import__(name)
    versions[name] = module.__version__
versions['R'] = subprocess.run(['Rscript','-e','cat(R.version.string)'], capture_output=True, text=True, check=True).stdout
versions['cran_drf'] = subprocess.run(['Rscript','-e','cat(as.character(packageVersion("drf")))'], capture_output=True, text=True, check=True).stdout
config = {{'shard_index': SHARD_INDEX, 'shard_total': SHARD_TOTAL,
           'manifest_contract_id': MANIFEST_SLICE['manifest_contract_id'],
           'manifest_checksum': MANIFEST_SLICE['manifest_checksum'],
           'estimator_source_hash': MANIFEST_SLICE['estimator_source_hash'],
           'evaluation_protocol_id': MANIFEST_SLICE['evaluation_protocol_id'],
           'source_archive_sha256': {source_sha!r}, 'n_cells': len(expected),
           'causal_drf_paper_repository': CAUSAL_DRF_PAPER_REPOSITORY,
           'causal_drf_paper_commit': CAUSAL_DRF_PAPER_COMMIT,
           'causal_drf_paper_files': dict(CAUSAL_DRF_PAPER_FILES),
           'causal_clean_drf_commit': CAUSAL_CLEAN_DRF_COMMIT,
           'n_failed': int(frame.loc[frame.status == 'failed', 'cell_key'].nunique()),
           'versions': versions, 'completed_at': time.time()}}
(out / 'manifest_slice.json').write_text(json.dumps(MANIFEST_SLICE, indent=2), encoding='utf-8')
(out / 'completion.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
inventory = {{path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in out.iterdir() if path.is_file()}}
(out / 'sha256_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
print(json.dumps(config, indent=2))
"""


def download(index: int, manifest_sha: str) -> str:
    short = manifest_sha[:12]
    return f"""\
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
output_file = 'wcf_confirmatory_shard_{index:02d}_{short}.zip'
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
        f"# WCF confirmatory rerun, shard {index:02d} of {total}\n\n"
        f"This shard contains {len(shard)} estimator cells grouped into "
        f"{len(group_cells(shard))} paired DGP-seed replications. Its conservative "
        f"reference estimate is {load / 3600:.2f} hours. Run all cells. The run "
        "cell checkpoints after every estimator and can be rerun after interruption."
    )
    cells = [
        markdown(title), code(THREADS), code(PYTHON_SETUP),
        code(setup_source(encoded, source_sha, manifest["manifest_checksum"])),
        code(FOREST_ENVIRONMENT), code(FOREST_SETUP_PINNED),
        code(causal_drf_paper_pin()),
        code(registration(index, total, shard, manifest, load)),
        code(RUN), code(finalize(source_sha)),
        code(download(index, manifest["manifest_checksum"])),
    ]
    return {
        "cells": [{**cell, "source": cell["source"].splitlines(keepends=True)} for cell in cells],
        "metadata": {"colab": {"name": f"wcf_confirmatory_shard_{index:02d}"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    args = parser.parse_args()
    manifest = load_manifest()
    if args.shards < 1 or args.shards > len(group_cells(manifest["cells"])):
        raise SystemExit("invalid shard count")
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.ipynb")):
        raise SystemExit(f"refusing to overwrite notebooks in {output}")
    shards, loads = allocate(manifest["cells"], args.shards)
    encoded, source_sha = source_archive()
    records = []
    for index, (shard, load) in enumerate(zip(shards, loads)):
        path = output / f"wcf_confirmatory_shard_{index:02d}.ipynb"
        path.write_text(json.dumps(notebook(index, args.shards, shard, manifest, load,
                                            encoded, source_sha), indent=1), encoding="utf-8")
        records.append({"index": index, "notebook": path.name, "cells": len(shard),
                        "replications": len(group_cells(shard)),
                        "estimated_seconds": load,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    generation = {"manifest_checksum": manifest["manifest_checksum"],
                  "estimator_source_hash": manifest["estimator_source_hash"],
                  "source_archive_sha256": source_sha, "n_shards": args.shards,
                  "n_cells": sum(x["cells"] for x in records), "notebooks": records}
    (output / "notebook_generation.json").write_text(json.dumps(generation, indent=2) + "\n", encoding="utf-8")
    lines = ["# WCF confirmatory Colab shards", "",
             f"Run all {args.shards} notebooks with at most 51 concurrent sessions. Together they cover all 2,100 cells and 700 paired replications exactly once.", "",
             "Each notebook installs pinned dependencies, verifies the embedded source, verifies the authors' Causal-DRF repository at its pinned commit, checkpoints every estimator, validates coverage, writes a hash inventory, and downloads one ZIP file.", "",
             "| Notebook | Cells | Paired replications | Conservative hours |", "|---|---:|---:|---:|"]
    lines.extend(f"| `{x['notebook']}` | {x['cells']} | {x['replications']} | {x['estimated_seconds']/3600:.2f} |" for x in records)
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.shards} notebooks to {output}")
    print(f"load range: {min(loads)/3600:.2f} to {max(loads)/3600:.2f} reference hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
