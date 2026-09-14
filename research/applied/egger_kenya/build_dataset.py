"""Build the analysis-ready Egger et al. (2022) Kenya dataset for WCF.

Unit: village (n=653). Treatment: village-level `treat`. Outcome: K=25 weighted
quantiles of household per-capita nondurable consumption at endline
(`nondurables_exp_pc`, weights `hhweight_EL`), one vector per village.

Sources (replication package, CC BY 4.0):
  Zenodo DOI 10.5281/zenodo.16548593 / extracted at
  /tmp/opencode/wcf_scout/egger/replication_package
  - code/rawdata/GE_Treat_Status_Master_PUBLIC.dta
  - code/data/GE_HHLevel_ECMA.dta
  - code/rawdata/GE_HH-Census-BL_PUBLIC.dta
  - code/data/GE_VillageLevel_ECMA.dta

Run from the repo root:
    PYTHONPATH=src /usr/bin/python3 research/applied/egger_kenya/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from wasserstein_causal_forests.applied.adapter import (  # noqa: E402
    AppliedDataset,
    ecdf_moderator,
    midpoint_levels,
    quantile_matrix,
    weighted_quantiles,
)

BASE = Path("/tmp/opencode/wcf_scout/egger/replication_package")
OUT = Path("results/applied_study_exploration/egger_kenya")
DATA_OUT = OUT / "data"
LEVELS = midpoint_levels()
K = LEVELS.size

PRIMARY_OUTCOME = "nondurables_exp_pc"
WEIGHT = "hhweight_EL"


def read(rel: str, cols: list[str]) -> pd.DataFrame:
    df, _ = pyreadstat.read_dta(str(BASE / rel), usecols=cols)
    return df


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=0)
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def build() -> AppliedDataset:
    # ------------------------------------------------------------------ design
    master = read(
        "code/rawdata/GE_Treat_Status_Master_PUBLIC.dta",
        [
            "village_code",
            "treat",
            "hi_sat",
            "sat_grp",
            "flag_dropvill",
            "subcounty",
            "location_code",
            "sublocation_code",
            "satcluster",
        ],
    )
    n_raw = len(master)
    master = master[master.flag_dropvill == 0].copy()
    assert len(master) == 653, len(master)
    master["treat"] = master["treat"].astype(int)
    master["hi_sat"] = master["hi_sat"].astype(int)
    assert master.treat.sum() == 328 and (master.treat == 0).sum() == 325
    assert master.village_code.is_unique

    # ------------------------------------------------------- household outcomes
    hh = read(
        "code/data/GE_HHLevel_ECMA.dta",
        [
            "hhid",
            "village_code",
            "treat",
            "hhweight_EL",
            "nondurables_exp_pc",
            "durables_exp_pc",
            "hhsize1_BL",
            "h1_8_electricity_BL",
            "p1_assets_wins_PPP_BL",
        ],
    )
    n_hh_raw = len(hh)
    hh = hh.merge(
        master[["village_code", "treat"]],
        on="village_code",
        suffixes=("_hh", "_master"),
        how="left",
        validate="many_to_one",
    )
    assert hh.treat_master.notna().all()
    assert int((hh.treat_hh != hh.treat_master).sum()) == 0
    hh = hh.drop(columns=["treat_hh"]).rename(columns={"treat_master": "treat"})

    usable = (
        hh[PRIMARY_OUTCOME].notna()
        & hh[WEIGHT].notna()
        & (hh[WEIGHT] > 0)
        & hh["village_code"].notna()
    )
    dropped = hh[~usable]
    hh = hh[usable].copy()

    units, q = quantile_matrix(
        hh[PRIMARY_OUTCOME].to_numpy(),
        hh[WEIGHT].to_numpy(),
        hh["village_code"].to_numpy(),
        levels=LEVELS,
    )
    assert q.shape[1] == K
    assert np.all(np.isfinite(q)) and np.all(np.diff(q, axis=1) >= -1e-12)

    qdf = pd.DataFrame(
        {f"q{k + 1:02d}": q[:, k] for k in range(K)}, index=units
    )
    qdf.index.name = "village_code"

    # ----------------------------------------------------------------- village X
    cens = read(
        "code/rawdata/GE_HH-Census-BL_PUBLIC.dta",
        ["village_code", "eligible", "has_biz"],
    )
    cens["has_biz_yes"] = np.where(
        cens.has_biz.astype(str).eq("yes"), 1.0,
        np.where(cens.has_biz.astype(str).eq("no"), 0.0, np.nan),
    )
    cens["eligible_ind"] = cens.eligible.astype(float)
    census = (
        cens.groupby("village_code")
        .agg(
            census_hh=("has_biz", "size"),
            share_biz=("has_biz_yes", "mean"),
            share_elig=("eligible_ind", "mean"),
        )
        .reset_index()
    )
    assert census.share_biz.notna().all() and census.share_elig.notna().all()

    vlev = read(
        "code/data/GE_VillageLevel_ECMA.dta",
        ["village_code", "treat", "hi_sat", "n_allents_BL"],
    )
    vlev = vlev.rename(columns={"treat": "treat_vl", "hi_sat": "hi_sat_vl"})

    vill = (
        hh.groupby("village_code")
        .agg(
            assets_mod=("p1_assets_wins_PPP_BL", "mean"),
            hhsize_bl=("hhsize1_BL", "mean"),
            elec_bl=("h1_8_electricity_BL", "mean"),
            n_hh_sample=("hhid", "size"),
        )
        .reset_index()
    )

    x = (
        vill.merge(master[["village_code", "treat", "hi_sat", "subcounty"]], on="village_code", how="left")
        .merge(vlev, on="village_code", how="left", validate="one_to_one")
        .merge(census, on="village_code", how="left", validate="one_to_one")
    )
    assert len(x) == len(units)
    assert x[["treat", "treat_vl", "hi_sat", "hi_sat_vl"]].notna().all().all()
    assert int((x.treat != x.treat_vl).sum()) == 0
    assert int((x.hi_sat != x.hi_sat_vl).sum()) == 0

    x["log_census_hh"] = np.log(x["census_hh"].astype(float))
    x["ents_pc_bl"] = x["n_allents_BL"].astype(float) / x["census_hh"].astype(float)
    x["subcounty_ugunja"] = (x["subcounty"] == "UGUNJA").astype(float)
    x["subcounty_ukwala"] = (x["subcounty"] == "UKWALA").astype(float)

    feature_names = [
        "assets_bl_village_mean",
        "hhsize_bl_village_mean",
        "electricity_bl_village_share",
        "log_census_households",
        "census_share_nonag_business",
        "baseline_enterprises_per_capita",
        "subcounty_ugunja",
        "subcounty_ukwala",
    ]
    X = np.column_stack(
        [
            x["assets_mod"].to_numpy(float),                       # moderator raw, col 0
            zscore(x["hhsize_bl"].to_numpy(float)),
            zscore(x["elec_bl"].to_numpy(float)),
            zscore(x["log_census_hh"].to_numpy(float)),
            zscore(x["share_biz"].to_numpy(float)),
            zscore(x["ents_pc_bl"].to_numpy(float)),
            x["subcounty_ugunja"].to_numpy(float),
            x["subcounty_ukwala"].to_numpy(float),
        ]
    )
    assert np.all(np.isfinite(X))
    moderator_raw = x["assets_mod"].to_numpy(float)
    X[:, 0] = ecdf_moderator(moderator_raw)

    A = x["treat"].to_numpy(int)
    assert A.sum() == 328 and (A == 0).sum() == 325

    # ----------------------------------------------------------------- q_star
    pooled = hh[hh.treat == 0]
    q_star_control = weighted_quantiles(
        pooled[PRIMARY_OUTCOME].to_numpy(), pooled[WEIGHT].to_numpy(), levels=LEVELS
    )
    pooled_all = weighted_quantiles(
        hh[PRIMARY_OUTCOME].to_numpy(), hh[WEIGHT].to_numpy(), levels=LEVELS
    )
    cap = float(np.interp(0.75, LEVELS, q_star_control))  # control p75 (KSH pc)
    q_star_capped = np.minimum(q_star_control, cap)

    meta = {
        "source": "Egger, Haushofer, Miguel, Niehaus, Walker (2022), replication package",
        "zenodo_doi": "10.5281/zenodo.16548593",
        "license": "CC BY 4.0",
        "extracted_at": str(BASE),
        "files": [
            "code/rawdata/GE_Treat_Status_Master_PUBLIC.dta",
            "code/data/GE_HHLevel_ECMA.dta",
            "code/rawdata/GE_HH-Census-BL_PUBLIC.dta",
            "code/data/GE_VillageLevel_ECMA.dta",
        ],
        "unit": "village",
        "outcome": "weighted quantiles of nondurables_exp_pc (hhweight_EL) at endline, K=25 midpoint grid",
        "treatment": "village treat (binary, 0/1)",
        "design_counts": {
            "master_rows_raw": int(n_raw),
            "master_rows_kept_flag_dropvill0": int(len(master)),
            "hh_rows_raw": int(n_hh_raw),
            "hh_rows_usable": int(len(hh)),
            "hh_rows_dropped": int(len(dropped)),
            "villages_with_quantiles": int(len(units)),
            "treated": int(A.sum()),
            "control": int((A == 0).sum()),
            "hi_sat1_among_treated": int((x.loc[x.treat == 1, "hi_sat"] == 1).sum()),
            "hi_sat1_among_control": int((x.loc[x.treat == 0, "hi_sat"] == 1).sum()),
        },
        "hh_per_village": {
            "min": int(x.n_hh_sample.min()),
            "median": float(x.n_hh_sample.median()),
            "mean": float(x.n_hh_sample.mean()),
            "max": int(x.n_hh_sample.max()),
            "n_ge15": int((x.n_hh_sample >= 15).sum()),
            "n_ge20": int((x.n_hh_sample >= 20).sum()),
        },
        "moderator": {
            "variable": "village mean of p1_assets_wins_PPP_BL (baseline household assets, winsorized, PPP)",
            "note": "no constructed baseline consumption per capita exists in the package; baseline assets are the poverty proxy",
        },
        "q_star_primary": "weighted quantiles of pooled CONTROL-village household nondurables_exp_pc (status-quo benchmark)",
        "q_star_secondary": f"pooled control quantiles capped at the control p75 = {cap:.1f} KSH pc (compressed benchmark)",
        "feature_names": feature_names,
        "caveats": [
            "inner sample is ~12.6 households/village; Q is estimated with sampling error",
            "SUTVA is questionable: saturation design creates cross-village spillovers",
            "q_star is a within-data status-quo benchmark; positive reference contrast means moving away from it",
        ],
    }

    for name, vec in [
        ("q_star_control", q_star_control),
        ("q_star_capped", q_star_capped),
        ("q_star_pooled_all", pooled_all),
    ]:
        meta[name] = [float(v) for v in vec]

    ds = AppliedDataset(
        study="egger_kenya",
        X=X,
        A=A,
        Q=q,
        moderator_raw=moderator_raw,
        q_star=q_star_control,
        feature_names=feature_names,
        meta=meta,
    ).validate()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    ds.save(DATA_OUT)

    # extra transparency artifacts (not inputs of the estimator)
    qdf.to_csv(OUT / "village_quantiles.csv")
    x[
        [
            "village_code",
            "treat",
            "hi_sat",
            "subcounty",
            "n_hh_sample",
            "assets_mod",
            "hhsize_bl",
            "elec_bl",
            "census_hh",
            "share_biz",
            "share_elig",
            "n_allents_BL",
        ]
    ].to_csv(OUT / "village_covariates.csv", index=False)
    pd.DataFrame(
        {
            "u": LEVELS,
            "q_star_control": q_star_control,
            "q_star_capped": q_star_capped,
            "q_star_pooled_all": pooled_all,
        }
    ).to_csv(OUT / "q_star.csv", index=False)
    with open(OUT / "build_summary.json", "w") as fh:
        json.dump(
            {
                "n": int(ds.X.shape[0]),
                "p": int(ds.X.shape[1]),
                "K": int(ds.Q.shape[1]),
                "n_treated": int(A.sum()),
                "n_control": int((A == 0).sum()),
                "moderator_bin_counts": np.bincount(
                    np.searchsorted([-0.5, 0.0, 0.5], ds.X[:, 0], side="right"), minlength=4
                ).tolist(),
                "q_star_capped_at": cap,
                "hh_dropped_rows": int(len(dropped)),
            },
            fh,
            indent=2,
        )
    return ds


if __name__ == "__main__":
    ds = build()
    print("built", ds.study, ds.X.shape, ds.Q.shape, "T=", int(ds.A.sum()), "C=", int((ds.A == 0).sum()))
    print("moderator bins:", np.bincount(np.searchsorted([-0.5, 0.0, 0.5], ds.X[:, 0], side="right"), minlength=4).tolist())
    print("q_star_control:", np.round(ds.q_star, 1).tolist())
    print("q_star_capped:", np.round(np.asarray(ds.meta["q_star_capped"]), 1).tolist())
