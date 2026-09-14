"""Inspect the raw Project STAR files and verify pre-registration claims.

Run from the repository root:

    PYTHONPATH=src python research/applied/project_star/01_inspect_data.py

Writes results/applied_study_exploration/project_star/misc/inspection.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_common import GRADES, SCHOOL_FL_VARS, load_raw, school_features  # noqa: E402

OUT = Path("results/applied_study_exploration/project_star/misc/inspection.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    students, schools = load_raw()
    sf = school_features(schools)
    report: dict = {
        "students_shape": list(students.shape),
        "schools_shape": list(schools.shape),
        "student_sha256": sha256(Path("/tmp/opencode/wcf_scout/STAR_Students.tab")),
        "school_sha256": sha256(Path("/tmp/opencode/wcf_scout/STAR_K3_Schools.tab")),
        "grades": {},
        "school_overlap": {},
        "school_file_checks": {},
    }

    # 1. School-file free-lunch columns vs student-record aggregates.
    for grade in GRADES:
        assigned = students[students[f"{grade}classtype"].notna()]
        arm_mask = {
            "small": assigned[f"{grade}classtype"] == 1,
            "regular": assigned[f"{grade}classtype"].isin([2, 3]),
        }
        sub = students[students[f"{grade}schid"].notna() & students[f"{grade}freelunch"].notna()]
        share = (
            sub.assign(fl=(sub[f"{grade}freelunch"] == 1).astype(float))
            .groupby(f"{grade}schid")["fl"]
            .mean()
            * 100
        )
        share.index = share.index.astype(int)
        official = schools.set_index("schid")[SCHOOL_FL_VARS[grade]].astype(float)
        common = share.index.intersection(official.index)
        diff = (share.loc[common] - official.loc[common]).abs()
        ctype = pd.to_numeric(
            students.loc[students[f"{grade}tchid"].notna(), f"{grade}classtype"]
        )
        report["grades"][grade] = {
            "n_students_with_class": int(students[f"{grade}classtype"].notna().sum()),
            "n_units": int(
                students.loc[students[f"{grade}tchid"].notna()]
                .groupby([f"{grade}schid", f"{grade}tchid"])
                .ngroups
            ),
            "classtype_student_counts": {
                str(k): int(v) for k, v in ctype.value_counts().sort_index().items()
            },
            "freelunch_missing_share_assigned": float(
                assigned[f"{grade}freelunch"].isna().mean()
            ),
            "math_missing_share_by_arm": {
                label: float(
                    assigned.loc[arm_mask[label], f"{grade}tmathss"].isna().mean()
                )
                for label in ("small", "regular")
            },
            "reading_missing_share_by_arm": {
                label: float(
                    assigned.loc[arm_mask[label], f"{grade}treadss"].isna().mean()
                )
                for label in ("small", "regular")
            },
        }
        report["school_file_checks"][grade] = {
            "official_column": SCHOOL_FL_VARS[grade],
            "n_schools_compared": int(len(common)),
            "pearson_r_official_vs_student_share": float(
                share.loc[common].corr(official.loc[common])
            ),
            "mean_abs_diff_pct_points": float(diff.mean()),
        }

        # 2. Unit-level constancy checks.
        key = [f"{grade}schid", f"{grade}tchid"]
        kept = students[students[f"{grade}tchid"].notna()].copy()
        kept["unit_id"] = (
            f"{grade}_" + kept[key[0]].astype(int).astype(str)
            + "_" + kept[key[1]].astype(int).astype(str)
        )
        ct = kept.groupby("unit_id")[f"{grade}classtype"].nunique()
        report["school_overlap"][grade] = {
            "n_units": int(kept["unit_id"].nunique()),
            "units_with_two_classtypes": int((ct > 1).sum()),
            "n_schools": int(kept[key[0]].nunique()),
            "schools_with_both_arms": int(
                (
                    kept.assign(small=kept[f"{grade}classtype"] == 1)
                    .groupby(key[0])["small"]
                    .agg(lambda s: s.any() and (~s).any())
                ).sum()
            ),
        }

    # 3. Urbanicity agreement between official school file and student records.
    agree = []
    for grade in GRADES:
        sub = students[students[f"{grade}schid"].notna()]
        mode = sub.groupby(f"{grade}schid")[f"{grade}surban"].agg(
            lambda s: s.mode().iloc[0]
        )
        mode.index = mode.index.astype(int)
        official = schools.set_index("schid")["var1"].astype(float)
        common = mode.index.intersection(official.index)
        agree.append((mode.loc[common] == official.loc[common]).mean())
    report["school_file_checks"]["urbanicity_agreement"] = float(np.mean(agree))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
