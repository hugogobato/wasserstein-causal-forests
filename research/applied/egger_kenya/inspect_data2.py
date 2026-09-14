"""Targeted inspection of the Egger et al. (2022) replication package.

For the small village-level files reads the full data; for the large
household-level files reads only the columns needed for the applied study.
Read-only; run from the repo root:

    /usr/bin/python3 research/applied/egger_kenya/inspect_data2.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadstat

BASE = "/tmp/opencode/wcf_scout/egger/replication_package"


def describe(rel: str, usecols: list[str] | None = None, show_cols: bool = False) -> None:
    path = f"{BASE}/{rel}"
    df, meta = pyreadstat.read_dta(path, usecols=usecols)
    print("=" * 100)
    print(f"FILE {rel}: rows={meta.number_rows} cols={meta.number_columns}")
    labels = dict(zip(meta.column_names, meta.column_labels))
    if show_cols:
        for col in meta.column_names:
            print(f"    {col!r:42s} {str(labels.get(col))[:80]}")
    if usecols:
        print("  selected columns:")
        for col in usecols:
            if col not in df.columns:
                print(f"    MISSING: {col}")
                continue
            s = df[col]
            nun = s.nunique(dropna=True)
            print(f"    {col!r:42s} dtype={str(s.dtype):8s} nuniq={nun:7d} nan={s.isna().mean():.3f}")
            if nun <= 12:
                vc = s.value_counts(dropna=False).sort_index()
                print(f"      values: {dict(vc)}")
            else:
                q = s.quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
                print(f"      quantiles(0,.25,.5,.75,1): {[round(float(v), 3) for v in q]}")
    return df, meta


def main() -> None:
    print("### GE_Treat_Status_Master_PUBLIC.dta (village treatment assignment)")
    df, meta = describe("code/rawdata/GE_Treat_Status_Master_PUBLIC.dta", show_cols=True)
    print(df.head(10).to_string())

    print("\n### GE_VillageLevel_ECMA.dta columns (all)")
    _, _ = describe("code/data/GE_VillageLevel_ECMA.dta", show_cols=True)

    print("\n### GE_HH-Census-BL_PUBLIC.dta columns (all)")
    _, _ = describe("code/rawdata/GE_HH-Census-BL_PUBLIC.dta", show_cols=True)

    print("\n### GE_HH-SampleMaster_PUBLIC.dta columns (all)")
    _, _ = describe("code/rawdata/GE_HH-SampleMaster_PUBLIC.dta", show_cols=True)

    print("\n### GE_HH-Census_Analysis_HHLevel.dta columns (all)")
    _, _ = describe("code/data/GE_HH-Census_Analysis_HHLevel.dta", show_cols=True)

    print("\n### GE_HHLevel_ECMA.dta key columns")
    describe(
        "code/data/GE_HHLevel_ECMA.dta",
        usecols=[
            "hhid",
            "village_code",
            "treat",
            "hi_sat",
            "hhweight_EL",
            "nondurables_exp_pc",
            "durables_exp_pc",
            "nondurables_exp",
            "durables_exp",
            "eligible",
            "sublocation_code",
            "survey_mth",
        ],
    )


if __name__ == "__main__":
    main()
