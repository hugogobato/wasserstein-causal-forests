"""Inspect candidate covariates/outcomes and design variables in the Egger package.

Read-only; run from the repo root.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyreadstat

BASE = "/tmp/opencode/wcf_scout/egger/replication_package"

def cols_of(rel: str) -> list[str]:
    _, meta = pyreadstat.read_dta(f"{BASE}/{rel}", metadataonly=True)
    return list(meta.column_names), dict(zip(meta.column_names, meta.column_labels))


def show(rel: str, patterns: str, limit: int = 200) -> None:
    names, labels = cols_of(rel)
    rx = re.compile(patterns, re.IGNORECASE)
    print("=" * 100)
    print(f"{rel}: {len(names)} columns; pattern {patterns!r}")
    hits = [c for c in names if rx.search(c)]
    print(f"  {len(hits)} matches")
    for c in hits[:limit]:
        print(f"    {c!r:55s} {str(labels.get(c))[:70]}")


def crosstab(rel: str, cols: list[str]) -> None:
    df, meta = pyreadstat.read_dta(f"{BASE}/{rel}", usecols=cols)
    print("=" * 100)
    print(f"{rel} crosstabs (rows={meta.number_rows})")
    print(df.groupby(cols, dropna=False).size().to_string())


def main() -> None:
    show("code/rawdata/GE_Treat_Status_Master_PUBLIC.dta", r".")
    crosstab(
        "code/rawdata/GE_Treat_Status_Master_PUBLIC.dta",
        ["treat", "hi_sat", "sat_grp", "ge_treat_status", "flag_dropvill"],
    )
    show("code/data/GE_VillageLevel_ECMA.dta", r"(pop|hh|asset|cons|exp|educ|n_|num|roof|wall|floor|elig|poverty|inc)")
    show(
        "code/data/GE_HHLevel_ECMA.dta",
        r"(nondurables_exp_pc.*BL|durables_exp_pc.*BL|consumption.*BL|p1_assets.*BL|asset.*BL|hhsize|num_hh|educ|child|water|electr|rent|foodcons)",
    )
    show("code/rawdata/GE_HH-Survey-BL_PUBLIC.dta", r"(nondurables_exp_pc|durables_exp_pc|p1_assets|hhsize|num_hh|hh_size|education|p7_1)")
    show("code/data/GE_HH-Analysis_AllHHs.dta", r"(^treat$|^village_code$|^hhweight|nondurables_exp_pc$|_BL$|hhsize)")


if __name__ == "__main__":
    main()
