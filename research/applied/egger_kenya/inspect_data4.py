"""Design/merge diagnostics for the Egger Kenya applied study.

Checks arm counts, treat/hi_sat consistency across files, per-village sample
sizes, outcome coverage, and baseline covariate availability.
Read-only; run from the repo root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadstat

BASE = "/tmp/opencode/wcf_scout/egger/replication_package"


def rd(rel: str, cols: list[str]) -> pd.DataFrame:
    df, meta = pyreadstat.read_dta(f"{BASE}/{rel}", usecols=cols)
    return df


def main() -> None:
    pd.set_option("display.width", 200)

    master = rd(
        "code/rawdata/GE_Treat_Status_Master_PUBLIC.dta",
        [
            "village_code",
            "treat",
            "hi_sat",
            "sat_grp",
            "ge_treat_status",
            "flag_dropvill",
            "subcounty",
            "location_code",
            "sublocation_code",
            "satcluster",
            "low_exposure_zone",
        ],
    )
    print("MASTER: rows", len(master), "unique villages", master.village_code.nunique())
    keep = master[master.flag_dropvill == 0].copy()
    print("MASTER after flag_dropvill==0:", len(keep))
    print("treat counts:", keep.treat.value_counts(dropna=False).to_dict())
    print("hi_sat counts:", keep.hi_sat.value_counts(dropna=False).to_dict())
    print("sat_grp counts:", keep.sat_grp.value_counts(dropna=False).to_dict())
    print("subcounty:", keep.subcounty.value_counts(dropna=False).to_dict())
    print("satcluster n:", keep.satcluster.nunique())
    print("treat x hi_sat:\n", pd.crosstab(keep.treat, keep.hi_sat, dropna=False))
    print("treat x sat_grp:\n", pd.crosstab(keep.treat, keep.sat_grp, dropna=False))
    print("treat x low_exposure_zone:\n", pd.crosstab(keep.treat, keep.low_exposure_zone, dropna=False))

    vl = rd(
        "code/data/GE_VillageLevel_ECMA.dta",
        ["village_code", "treat", "hi_sat", "n_allents", "n_allents_BL", "vill_profit1_wins_s_PPP", "vill_profit1_wins_s_PPP_BL"],
    )
    print("\nVILLAGELEVEL: rows", len(vl), "unique", vl.village_code.nunique())
    print("treat counts:", vl.treat.value_counts(dropna=False).to_dict())
    print("hi_sat counts:", vl.hi_sat.value_counts(dropna=False).to_dict())
    print("treat x hi_sat:\n", pd.crosstab(vl.treat, vl.hi_sat, dropna=False))

    m = keep.merge(vl, on="village_code", suffixes=("_master", "_vl"))
    print("merged master+vl:", len(m))
    print("treat disagreement:", int((m.treat_master != m.treat_vl).sum()))
    print("hi_sat disagreement:", int((m.hi_sat_master != m.hi_sat_vl).sum()))
    print("missing vl n_allents_BL:", m.n_allents_BL.isna().sum())

    hh = rd(
        "code/data/GE_HHLevel_ECMA.dta",
        [
            "hhid",
            "village_code",
            "treat",
            "hi_sat",
            "hhweight_EL",
            "nondurables_exp_pc",
            "durables_exp_pc",
            "eligible",
            "sublocation_code",
            "survey_mth",
            "hhsize1_BL",
            "hhsize2_BL",
            "numchildren_BL",
            "p1_assets_BL",
            "p1_assets_wins_PPP_BL",
            "h1_8_electricity_BL",
            "a_assets_std_BL",
            "nondurables_exp_pc_wins",
            "nondurables_exp_pc_PPP",
            "durables_exp_pc_PPP",
        ],
    )
    print("\nHHLEVEL: rows", len(hh), "villages", hh.village_code.nunique())
    print("hh treat counts:", hh.treat.value_counts(dropna=False).to_dict())
    print("hh hi_sat counts:", hh.hi_sat.value_counts(dropna=False).to_dict())
    print("hh treat x hi_sat:\n", pd.crosstab(hh.treat, hh.hi_sat, dropna=False))

    hvv = hh.groupby("village_code").agg(
        n_hh=("hhid", "size"),
        treat=("treat", "first"),
        hi_sat=("hi_sat", "first"),
        n_out=("nondurables_exp_pc", "count"),
    )
    print("\nper-village hh count distribution:")
    print(hvv.n_hh.describe().to_string())
    print("n_hh histogram:", np.bincount(hvv.n_hh.values.astype(int)).tolist())
    print("villages with >=20 hh:", int((hvv.n_hh >= 20).sum()), "of", len(hvv))
    print("villages with >=10 hh:", int((hvv.n_hh >= 10).sum()))

    print("\nmerged hh-level treat vs master treat disagreements:")
    mm = hh.merge(keep[["village_code", "treat", "hi_sat"]], on="village_code", suffixes=("_hh", "_master"))
    print("rows", len(mm), "missing master:", mm.treat_master.isna().sum())
    print("treat disagreements:", int((mm.treat_hh != mm.treat_master).sum()))
    print("hi_sat disagreements:", int((mm.hi_sat_hh != mm.hi_sat_master).sum()))

    print("\noutcome coverage:")
    print("nondurables_exp_pc NaN:", hh.nondurables_exp_pc.isna().sum())
    print("nondurables_exp_pc ==0:", int((hh.nondurables_exp_pc == 0).sum()))
    print("hhweight_EL NaN/<=0:", int((hh.hhweight_EL.isna() | (hh.hhweight_EL <= 0)).sum()))
    print("hhweight_EL describe:", hh.hhweight_EL.describe().to_dict())

    print("\nbaseline covariates missingness and quantiles:")
    for c in ["hhsize1_BL", "hhsize2_BL", "numchildren_BL", "p1_assets_BL", "p1_assets_wins_PPP_BL", "h1_8_electricity_BL", "a_assets_std_BL"]:
        s = hh[c]
        print(f"  {c:28s} nan={s.isna().mean():.4f} q=[{s.quantile(0):.3f},{s.quantile(.25):.3f},{s.quantile(.5):.3f},{s.quantile(.75):.3f},{s.quantile(1):.3f}]")

    census = rd(
        "code/rawdata/GE_HH-Census-BL_PUBLIC.dta",
        ["hhid_key", "village_code", "sublocation_code", "eligible", "has_biz", "biz_profit1", "biz_revenue1", "roof", "homeless", "marital_status"],
    )
    print("\nCENSUS: rows", len(census), "villages", census.village_code.nunique())
    print("eligible:", census.eligible.value_counts(dropna=False).to_dict())
    print("has_biz:", census.has_biz.value_counts(dropna=False).to_dict())
    print("homeless:", census.homeless.value_counts(dropna=False).to_dict())
    print("biz_profit1 nan:", census.biz_profit1.isna().mean(), "quantiles", census.biz_profit1.quantile([0, .5, .9, 1]).tolist())
    print("roof values:", census.roof.value_counts(dropna=False).head(10).to_dict())
    cv = census.groupby("village_code").size()
    print("census hh per village:", cv.describe().to_dict())
    missing_villages = set(keep.village_code) - set(census.village_code)
    print("villages in master without census rows:", len(missing_villages))


if __name__ == "__main__":
    main()
