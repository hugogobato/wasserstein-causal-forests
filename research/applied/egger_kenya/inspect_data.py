"""Inspect the Egger et al. (2022) replication package .dta files.

Prints file-level metadata plus columns matching study-relevant patterns.
Read-only; run from the repo root:

    /usr/bin/python3 research/applied/egger_kenya/inspect_data.py
"""

from __future__ import annotations

import sys

import pyreadstat

BASE = "/tmp/opencode/wcf_scout/egger/replication_package"

FILES = [
    "code/data/GE_HHLevel_ECMA.dta",
    "code/data/GE_VillageLevel_ECMA.dta",
    "code/data/GE_HH-Analysis_AllHHs.dta",
    "code/data/GE_HH-Census_Analysis_HHLevel.dta",
    "code/rawdata/GE_Treat_Status_Master_PUBLIC.dta",
    "code/rawdata/GE_HH-Census-BL_PUBLIC.dta",
    "code/rawdata/GE_HH-SampleMaster_PUBLIC.dta",
    "code/rawdata/GE_HH-Survey-BL_PUBLIC.dta",
    "code/rawdata/GE_HH-Survey-EL1_PUBLIC.dta",
]

PATTERNS = (
    "village",
    "treat",
    "sat",
    "hhweight",
    "weight",
    "nondurab",
    "durab",
    "exp",
    "cons",
    "asset",
    "educ",
    "numhh",
    "hhsize",
    "size",
    "pop",
    "subloc",
    "district",
    "county",
    "region",
    "endline",
    "el",
    "survey",
    "id",
)


def main() -> None:
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for rel in FILES:
        if only and not any(token in rel for token in only):
            continue
        path = f"{BASE}/{rel}"
        header, meta = pyreadstat.read_dta(path, metadataonly=True)
        print("=" * 100)
        print(f"FILE {rel}: n={header.shape[0]} cols={header.shape[1]}")
        matches = [
            c
            for c in header.columns
            if any(p in c.lower() for p in PATTERNS)
        ]
        print(f"  matched {len(matches)} of {len(header.columns)} columns")
        for col in matches:
            label = dict(zip(meta.column_names, meta.column_labels)).get(col, "")
            print(f"    {col!r:42s} {str(label)[:70]}")
        print("  variable labels for key ones:")
        for col in (
            "village_code",
            "treat",
            "hi_sat",
            "sat",
            "hhweight_EL",
            "nondurables_exp_pc",
            "durables_exp_pc",
        ):
            if col in header.columns:
                label = dict(zip(meta.column_names, meta.column_labels)).get(col, "")
                print(f"    {col!r:42s} {str(label)[:70]}")


if __name__ == "__main__":
    main()
