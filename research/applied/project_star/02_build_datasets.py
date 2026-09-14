"""Build and persist every Project STAR WCF dataset variant.

Run from the repository root:

    PYTHONPATH=src python research/applied/project_star/02_build_datasets.py

Outputs under results/applied_study_exploration/project_star/data/:
    dataset.npz + manifest.json          primary math, control q_star
    math_min15/dataset.npz + manifest.json
    math_reg2/dataset.npz + manifest.json
    math_qstarall/dataset.npz + manifest.json
    reading/dataset.npz + manifest.json
    units_*.csv                          unit-level tables (raw covariates)
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_common import (  # noqa: E402
    EXTRA_FUNCTIONALS,
    RESULTS_DIR,
    build_dataset,
    midpoint_levels,
)

DATA = RESULTS_DIR / "data"


def save_variant(ds, name: str, units=None, out_dir: str | None = None) -> dict:
    if out_dir is not None:
        folder = DATA / out_dir
    else:
        folder = DATA / name
    ds.save(folder)
    stem = name or "math_primary"
    if units is not None:
        units.to_csv(DATA / f"units_{stem}.csv", index=False)
    return {
        "name": stem,
        "data_dir": str(folder),
        "n": int(ds.X.shape[0]),
        "n_treated": int(ds.A.sum()),
        "n_control": int((ds.A == 0).sum()),
        "manifest_meta": ds.meta,
        "units_csv": str(DATA / f"units_{stem}.csv") if units is not None else None,
    }


def main() -> None:
    summaries = []

    ds_math, units_math, _ = build_dataset()
    summaries.append(save_variant(ds_math, "math_primary", units_math, out_dir=""))

    keep15 = (units_math["tested_n"] >= 15).to_numpy()
    ds15 = replace(
        ds_math,
        A=ds_math.A[keep15],
        X=ds_math.X[keep15],
        Q=ds_math.Q[keep15],
        moderator_raw=ds_math.moderator_raw[keep15],
        meta={
            **ds_math.meta,
            "variant": "min15",
            "note": "subsample of primary dataset; q_star and ecdf values frozen at primary fit",
        },
    )
    summaries.append(
        save_variant(ds15, "math_min15", units_math[keep15], out_dir="math_min15")
    )

    keep2 = units_math["classtype"].isin([1, 2]).to_numpy()
    ds2 = replace(
        ds_math,
        A=ds_math.A[keep2],
        X=ds_math.X[keep2],
        Q=ds_math.Q[keep2],
        moderator_raw=ds_math.moderator_raw[keep2],
        meta={
            **ds_math.meta,
            "variant": "regular_only_aide_dropped",
            "note": "A=1 small vs A=0 classtype 2 only; aide classes removed",
        },
    )
    summaries.append(save_variant(ds2, "math_reg2", units_math[keep2], out_dir="math_reg2"))

    ds_all, units_all, _ = build_dataset(qstar="all", study="project_star_math_qstarall")
    summaries.append(
        save_variant(ds_all, "math_qstarall", units_all, out_dir="math_qstarall")
    )

    ds_read, units_read, _ = build_dataset(
        score_name="reading", study="project_star_reading"
    )
    summaries.append(save_variant(ds_read, "reading", units_read, out_dir="reading"))

    out = {
        "variants": summaries,
        "extra_functionals": list(EXTRA_FUNCTIONALS),
        "midpoint_grid": midpoint_levels().tolist(),
    }
    with open(DATA / "build_summary.json", "w") as handle:
        json.dump(out, handle, indent=2, default=str)
    for s in summaries:
        print(s["name"], s["n"], s["n_treated"], s["n_control"])
    print("wrote", DATA / "build_summary.json")


if __name__ == "__main__":
    main()
