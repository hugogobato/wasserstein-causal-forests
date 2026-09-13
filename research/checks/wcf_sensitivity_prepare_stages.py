#!/usr/bin/env python3
"""Prepare the fresh-seed confirmation and random-forest propensity stages.

Both stages are addenda to the frozen primary manifest and never edit it:

* ``confirm`` re-tests the proposed primary (K=49, M=10) against the incumbent
  (K=25, M=10) on fresh seeds 20--39 for IC0--IC3, as the report requires when
  the pre-fixed sensitivity rule selects a new primary setting.
* ``flex_rf`` runs the same flexible-propensity comparison as ``propensity`` but
  with the random-forest factory, which the calibration diagnostic showed to be
  a competent flexible propensity estimator, unlike the overconfident default
  gradient-boosting factory.

Usage:

    python3 research/checks/wcf_sensitivity_prepare_stages.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.manifest import Cell  # noqa: E402
from wasserstein_causal_forests.g3.wcf_sensitivity import (  # noqa: E402
    WCF_SENSITIVITY_CONTRACT_ID,
    build_wcf_method_registry,
    build_wcf_sensitivity_manifest,
)

RESULTS = ROOT / "results" / "wcf_sensitivity"

CONFIRM_DGPS = ("IC0", "IC1", "IC2", "IC3")
CONFIRM_PAIRS = ((25, 10), (49, 10))
CONFIRM_SEEDS = tuple(range(20, 40))

FLEX_RF_DGPS = ("SYM-NL", "SYM-MU")
FLEX_RF_SIZES = (500, 1000)
FLEX_RF_SEEDS = tuple(range(10))

FLEX_RF_ENTRY = {
    "role": "variant",
    "adapter": "cwdb_dr",
    "produces_law": True,
    "cross_fitted": True,
    "parameters": {
        "contrast_candidates": [0.0, 50.0, 500.0],
        "n_folds": 3,
        "common_grid_levels": "COMMON199+INTERIOR",
        "propensity_factory": "random_forest",
    },
}


def _document(template: dict, cells: list[Cell], block: str, registry: dict) -> dict:
    document = json.loads(json.dumps(template))
    document["blocks"] = [block]
    document["block_counts"] = {block: len(cells)}
    document["n_cells"] = len(cells)
    document["method_registry"] = registry
    document["cells"] = [cell.to_dict() for cell in cells]
    checksum = hashlib.sha256(
        json.dumps(document["cells"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    document["manifest_checksum"] = checksum
    return document


def main() -> int:
    if not (RESULTS / "manifest.json").exists():
        raise SystemExit("the primary manifest is missing; freeze the primary study first")
    template = build_wcf_sensitivity_manifest(["income_onefactor"])
    registry = build_wcf_method_registry()

    confirm_cells = [
        Cell("wcf_sensitivity_v1_confirm", dgp, 1000, k, m, "cwdb_dr", seed)
        for dgp in CONFIRM_DGPS
        for k, m in CONFIRM_PAIRS
        for seed in CONFIRM_SEEDS
    ]
    confirm = _document(template, confirm_cells, "confirm", registry)
    confirm_path = RESULTS / "confirm" / "manifest.json"
    confirm_path.parent.mkdir(parents=True, exist_ok=True)
    confirm_path.write_text(json.dumps(confirm, indent=2), encoding="utf-8")
    print(f"confirm: {len(confirm_cells)} cells -> {confirm_path}")

    flex_registry = dict(registry)
    flex_registry["cwdb_dr_flex_rf"] = FLEX_RF_ENTRY
    flex_cells = [
        Cell("wcf_sensitivity_v1_flex_rf", dgp, n, 25, 10, "cwdb_dr_flex_rf", seed)
        for dgp in FLEX_RF_DGPS
        for n in FLEX_RF_SIZES
        for seed in FLEX_RF_SEEDS
    ]
    flex = _document(template, flex_cells, "flex_rf", flex_registry)
    flex_path = RESULTS / "flex_rf" / "manifest.json"
    flex_path.parent.mkdir(parents=True, exist_ok=True)
    flex_path.write_text(json.dumps(flex, indent=2), encoding="utf-8")
    print(f"flex_rf: {len(flex_cells)} cells -> {flex_path}")

    for path, cells, block in (
        (confirm_path, confirm_cells, "confirm"),
        (flex_path, flex_cells, "flex_rf"),
    ):
        document = json.loads(path.read_text(encoding="utf-8"))
        keys = [item["cell_key"] for item in document["cells"]]
        assert len(keys) == len(set(keys)), f"duplicate keys in {block}"
        for item in document["cells"]:
            assert item["method"] in document["method_registry"], item["method"]
        print(
            f"  {block}: checksum {document['manifest_checksum'][:16]} "
            f"blocks={document['blocks']} registry={sorted(document['method_registry'])}"
        )
    assert len({cell.key for cell in confirm_cells}) == 160
    assert len({cell.key for cell in flex_cells}) == 40
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
