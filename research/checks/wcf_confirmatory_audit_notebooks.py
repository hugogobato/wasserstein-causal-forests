#!/usr/bin/env python3
"""Audit the generated confirmatory Colab notebooks before they are uploaded.

Run from the repository root::

    python3 research/checks/wcf_confirmatory_audit_notebooks.py \
        --shards-dir colab/wcf_confirmatory_r50_26_shards

The audit is content-addressed against the frozen manifest and the generation
record.  It checks notebook validity, the exact union of every manifest slice,
paired-group integrity, the embedded source archive, pinned dependencies, the
authors' Causal-DRF provenance pin, and the download fallback.  A PASS means
the notebooks are a faithful, complete transport of the frozen contract.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.checks.wcf_confirmatory_make_colab_notebooks import (  # noqa: E402
    FOREST_ENVIRONMENT,
    FOREST_SETUP_PINNED,
    PAPER_COMMIT,
    PAPER_REPOSITORY,
    PAPER_SETUP_PATH,
    PAPER_SETUP_SHA256,
    PAPER_STUDY_PATH,
    PAPER_STUDY_SHA256,
    PYTHON_SETUP,
    RUN,
    THREADS,
    causal_drf_paper_pin,
    source_archive,
)
from research.run_wcf_confirmatory import (  # noqa: E402
    ORDINARY_METHODS,
    PAPER_DGPS_ALL,
    ZERO_METHODS,
    load_manifest,
    source_hash,
)

#: Layout of the generated notebooks.  Cells 1, 2, 4, 5, 6 and 8 are
#: shard-independent and audited by exact equality.
CELL_THREADS = 1
CELL_PYTHON = 2
CELL_ARCHIVE = 3
CELL_FOREST_ENV = 4
CELL_FOREST_SETUP = 5
CELL_PAPER_PIN = 6
CELL_REGISTRATION = 7
CELL_RUN = 8
CELL_FINALIZE = 9
CELL_DOWNLOAD = 10
FIXED_CELL_COUNT = 11
NOTEBOOK_PATTERN = re.compile(r"^wcf_confirmatory_shard_(\d{2})\.ipynb$")
REQUIRED_ARCHIVE_PATHS = (
    "research/run_wcf_confirmatory.py",
    "research/baselines/g3_causal_drf_original_driver.R",
    "research/baselines/g3_driver.R",
    "research/baselines/g3_drf_original_driver.R",
    "code/drfinference-main/drf-foo.R",
    "src/wasserstein_causal_forests/g3/confirmatory_evaluation.py",
    "src/wasserstein_causal_forests/g3/phase65_dgps.py",
)
PYTHON_PIN_PATTERN = re.compile(r"%pip -q install ((?:\S+==\S+\s*)+)")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def cell_source(nb: dict, index: int) -> str:
    return "".join(nb["cells"][index]["source"])


def extract(pattern: str, text: str, name: str) -> str:
    match = re.search(pattern, text, re.S)
    if match is None:
        raise AssertionError(f"missing {name}")
    return match.group(1)


class Audit:
    """Collect every finding instead of failing on the first one."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.notes: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def finish(self) -> int:
        for note in self.notes:
            print("note:", note)
        if self.errors:
            print(f"\nFAIL: {len(self.errors)} audit finding(s)")
            for index, error in enumerate(self.errors[:40], 1):
                print(f"{index:3d}. {error}")
            if len(self.errors) > 40:
                print(f"     ... and {len(self.errors) - 40} more")
            return 1
        print("\nPASS: notebooks are a complete, consistent transport of the contract")
        return 0


def audit_notebook(
    audit: Audit,
    path: Path,
    record: dict,
    manifest: dict,
    archive_sha: str,
    n_shards: int,
    seen_keys: set[str],
    group_shards: dict[tuple[str, int], set[int]],
    method_seeds: dict[tuple[str, str], set[int]],
) -> None:
    label = path.name
    raw = path.read_bytes()
    audit.check(sha256(raw) == record["sha256"], f"{label}: notebook sha256 mismatch")
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as error:
        audit.errors.append(f"{label}: invalid notebook JSON: {error}")
        return
    audit.check(nb.get("nbformat") == 4, f"{label}: nbformat is not 4")
    if len(nb.get("cells", [])) != FIXED_CELL_COUNT:
        audit.errors.append(
            f"{label}: has {len(nb.get('cells', []))} cells, expected "
            f"{FIXED_CELL_COUNT}; skipping cell checks"
        )
        return

    audit.check(cell_source(nb, CELL_THREADS) == THREADS,
                f"{label}: thread-pinning cell changed")
    audit.check(cell_source(nb, CELL_PYTHON) == PYTHON_SETUP,
                f"{label}: Python setup cell changed")
    audit.check(cell_source(nb, CELL_FOREST_ENV) == FOREST_ENVIRONMENT,
                f"{label}: Causal-DRF library environment cell changed")
    audit.check(cell_source(nb, CELL_FOREST_SETUP) == FOREST_SETUP_PINNED,
                f"{label}: R setup cell changed")
    audit.check(cell_source(nb, CELL_PAPER_PIN) == causal_drf_paper_pin(),
                f"{label}: authors' Causal-DRF provenance cell changed")
    audit.check(cell_source(nb, CELL_RUN) == RUN, f"{label}: run cell changed")

    pins = PYTHON_PIN_PATTERN.findall(cell_source(nb, CELL_PYTHON))
    audit.check(len(pins) == 1, f"{label}: expected one Python pin line")
    for pin in pins:
        audit.check(
            all("==" in item for item in pin.split()),
            f"{label}: unpinned Python dependency in {pin!r}",
        )

    archive_cell = cell_source(nb, CELL_ARCHIVE)
    declared_archive = extract(
        r"SOURCE_ARCHIVE_SHA256 = '([0-9a-f]{64})'", archive_cell,
        "SOURCE_ARCHIVE_SHA256",
    )
    audit.check(
        declared_archive == archive_sha,
        f"{label}: embedded source hash {declared_archive} != {archive_sha}",
    )
    encoded = re.sub(
        r"\s+", "",
        extract(r"SOURCE_ARCHIVE_B64 = '''\\\n(.*?)\n'''", archive_cell,
                "SOURCE_ARCHIVE_B64"),
    )
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        audit.errors.append(f"{label}: embedded source is not valid base64: {error}")
        return
    audit.check(
        sha256(payload) == declared_archive,
        f"{label}: embedded archive does not hash to its declared digest",
    )
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        audit.errors.append(f"{label}: embedded source is not a readable ZIP")
        return
    for required in REQUIRED_ARCHIVE_PATHS:
        audit.check(required in names, f"{label}: archive lacks {required}")

    registration = cell_source(nb, CELL_REGISTRATION)
    index = int(extract(r"SHARD_INDEX = (\d+)", registration, "SHARD_INDEX"))
    total = int(extract(r"SHARD_TOTAL = (\d+)", registration, "SHARD_TOTAL"))
    estimate = float(extract(
        r"ESTIMATED_REFERENCE_SECONDS = ([0-9.]+)", registration,
        "ESTIMATED_REFERENCE_SECONDS",
    ))
    audit.check(index == record["index"], f"{label}: shard index mismatch")
    audit.check(total == n_shards, f"{label}: shard total {total} != {n_shards}")
    audit.check(
        estimate == record["estimated_seconds"],
        f"{label}: reference estimate disagrees with the generation record",
    )
    slice_doc = json.loads(extract(
        r"MANIFEST_SLICE = json.loads\('''(.*?)'''\)", registration, "MANIFEST_SLICE"
    ))
    for field in (
        "manifest_contract_id",
        "manifest_checksum",
        "estimator_source_hash",
        "evaluation_protocol_id",
    ):
        audit.check(
            slice_doc.get(field) == manifest.get(field),
            f"{label}: manifest slice field {field} mismatch",
        )
    audit.check(
        slice_doc.get("method_registry") == manifest["method_registry"],
        f"{label}: method registry differs from the frozen manifest",
    )

    declared = {item["cell_key"]: item for item in manifest["cells"]}
    keys = [item["cell_key"] for item in slice_doc["cells"]]
    audit.check(len(keys) == len(set(keys)), f"{label}: duplicate cells in slice")
    audit.check(
        len(keys) == record["cells"],
        f"{label}: slice has {len(keys)} cells, record says {record['cells']}",
    )
    groups: dict[tuple[str, int], list[str]] = {}
    for item in slice_doc["cells"]:
        key = item["cell_key"]
        if key in declared:
            if item != declared[key]:
                audit.errors.append(
                    f"{label}: cell {key} payload differs from the frozen manifest"
                )
        else:
            audit.errors.append(f"{label}: undeclared cell {key}")
        if key in seen_keys:
            audit.errors.append(f"{label}: cell {key} already claimed by another shard")
        seen_keys.add(key)
        groups.setdefault((item["dgp"], item["seed"]), []).append(item["method"])
        method_seeds.setdefault((item["dgp"], item["method"]), set()).add(item["seed"])
    audit.check(
        len(groups) == record["replications"],
        f"{label}: {len(groups)} paired groups, record says {record['replications']}",
    )
    for (dgp, seed), methods in groups.items():
        expected_methods = ZERO_METHODS if dgp.startswith("ZI") else ORDINARY_METHODS
        if sorted(methods) != sorted(expected_methods):
            audit.errors.append(
                f"{label}: group {dgp} seed={seed} has methods {sorted(methods)}"
            )
        group_shards.setdefault((dgp, seed), set()).add(index)

    finalize = cell_source(nb, CELL_FINALIZE)
    for fragment in (
        "manifest_slice.json",
        "completion.json",
        "sha256_inventory.json",
        "CAUSAL_DRF_PAPER_COMMIT",
        "CAUSAL_DRF_PAPER_FILES",
        "CAUSAL_CLEAN_DRF_COMMIT",
    ):
        audit.check(fragment in finalize, f"{label}: finalize cell lacks {fragment}")

    download = cell_source(nb, CELL_DOWNLOAD)
    audit.check(
        "from google.colab import files" in download
        and "files.download(output_file)" in download
        and "except Exception" in download,
        f"{label}: download fallback missing",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", default="colab/wcf_confirmatory_r50_26_shards")
    parser.add_argument("--generation-record")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    directory = Path(args.shards_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    record_path = Path(args.generation_record) if args.generation_record else (
        directory / "notebook_generation.json"
    )
    generation = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = load_manifest()
    audit = Audit()

    audit.check(
        generation["manifest_checksum"] == manifest["manifest_checksum"],
        "generation record and frozen manifest disagree",
    )
    audit.check(
        generation["estimator_source_hash"] == manifest["estimator_source_hash"],
        "generation record and frozen manifest estimator hashes disagree",
    )
    notebooks = generation["notebooks"]
    audit.check(
        generation["n_shards"] == len(notebooks),
        "generation record shard count mismatch",
    )
    audit.check(
        generation["n_cells"] == manifest["n_cells"],
        "generation record cell total mismatch",
    )
    audit.check(
        [record["index"] for record in notebooks] == list(range(len(notebooks))),
        "generation record indices are not contiguous",
    )
    _, archive_sha = source_archive()
    audit.check(
        generation["source_archive_sha256"] == archive_sha,
        "embedded source archive does not match the current frozen tree",
    )
    audit.check(
        source_hash() == manifest["estimator_source_hash"],
        "current source tree does not match the frozen estimator source hash",
    )

    on_disk = sorted(path.name for path in directory.glob("*.ipynb"))
    audit.check(
        on_disk == sorted(record["notebook"] for record in notebooks),
        "notebook directory does not match the generation record exactly",
    )
    for record in notebooks:
        match = NOTEBOOK_PATTERN.match(record["notebook"])
        audit.check(
            match is not None and int(match.group(1)) == record["index"],
            f"generation record has an invalid notebook name: {record['notebook']}",
        )

    seen_keys: set[str] = set()
    group_shards: dict[tuple[str, int], set[int]] = {}
    method_seeds: dict[tuple[str, str], set[int]] = {}
    for record in notebooks:
        path = directory / record["notebook"]
        if not path.exists():
            audit.errors.append(f"missing notebook {record['notebook']}")
            continue
        audit_notebook(
            audit, path, record, manifest, archive_sha, len(notebooks),
            seen_keys, group_shards, method_seeds,
        )

    declared = {item["cell_key"] for item in manifest["cells"]}
    audit.check(
        seen_keys == declared,
        f"cell union mismatch: missing={len(declared - seen_keys)} "
        f"undeclared={len(seen_keys - declared)}",
    )
    splits = {
        key: sorted(shards)
        for key, shards in group_shards.items()
        if len(shards) != 1
    }
    audit.check(not splits, f"paired groups split across shards: {splits}")
    for paper_label, dgp in PAPER_DGPS_ALL.items():
        for method in (ZERO_METHODS if dgp.startswith("ZI") else ORDINARY_METHODS):
            seeds = method_seeds.get((dgp, method), set())
            audit.check(
                len(seeds) == len(manifest["replication_seeds"]),
                f"{paper_label} ({dgp}) {method}: {len(seeds)} replication seeds",
            )
    audit.notes.append(
        f"cells {len(seen_keys)}/{manifest['n_cells']}; paired groups "
        f"{len(group_shards)}/{len(manifest['replication_seeds']) * len(PAPER_DGPS_ALL)}; "
        f"shards {len(notebooks)}; archive {archive_sha[:12]}"
    )
    result = {
        "status": "FAIL" if audit.errors else "PASS",
        "shards_directory": str(directory.relative_to(ROOT)),
        "n_shards": len(notebooks),
        "n_cells": len(seen_keys),
        "manifest_checksum": manifest["manifest_checksum"],
        "source_archive_sha256": archive_sha,
        "causal_drf_paper": {
            "repository": PAPER_REPOSITORY,
            "commit": PAPER_COMMIT,
            "files": {
                PAPER_SETUP_PATH: PAPER_SETUP_SHA256,
                PAPER_STUDY_PATH: PAPER_STUDY_SHA256,
            },
        },
        "findings": audit.errors,
    }
    if not args.no_write:
        output = directory / "notebook_audit.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output.relative_to(ROOT)}")
    return audit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
