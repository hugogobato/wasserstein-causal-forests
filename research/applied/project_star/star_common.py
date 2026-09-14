"""Shared Project STAR ingestion and WCF run helpers.

Pre-registered study: see PREREGISTRATION.md in this directory. Nothing here
modifies the estimator; the module only builds the classroom-level dataset the
frozen adapter expects and post-processes returned model objects.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wasserstein_causal_forests.applied.adapter import (
    AppliedDataset,
    ecdf_moderator,
    make_functionals,
    midpoint_levels,
    quantile_matrix,
    run_wcf,
    weighted_quantiles,
)

STUDENT_PATH = Path("/tmp/opencode/wcf_scout/STAR_Students.tab")
SCHOOL_PATH = Path("/tmp/opencode/wcf_scout/STAR_K3_Schools.tab")
CANONICAL_STUDENT = Path("/tmp/opencode/wcf_scout/canonical/STAR_Students.tab")
CANONICAL_SCHOOL = Path("/tmp/opencode/wcf_scout/canonical/STAR_K-3_Schools.tab")
RESULTS_DIR = Path("results/applied_study_exploration/project_star")

SOURCES = {
    "student_file": {
        "name": "STAR_Students.tab",
        "dataverse_doi": "10.7910/DVN/SIWH9F",
        "datafile_id": 666716,
        "url": "https://dataverse.harvard.edu/api/access/datafile/666716",
        "license": "CC0",
        "sha256": "769be163ed54515858efa60b1a069c49ca0c475f0b0f9f5bdf90413be9d3ba97",
    },
    "school_file": {
        "name": "STAR_K-3_Schools.tab",
        "dataverse_doi": "10.7910/DVN/SIWH9F",
        "datafile_id": 666717,
        "url": "https://dataverse.harvard.edu/api/access/datafile/666717",
        "license": "CC0",
        "sha256": "776f2d3c4c09f047bfdfa5ce9406ed745ded425c4f5b4721f2c48aa81389cba9",
    },
}

GRADES = ("gk", "g1", "g2", "g3")
GRADE_LABEL = {"gk": "K", "g1": "1", "g2": "2", "g3": "3"}
SCHOOL_FL_VARS = {"gk": "var9", "g1": "var21", "g2": "var33", "g3": "var45"}
EXTRA_FUNCTIONALS = {
    "p10": lambda q: q[:, 2],
    "p90": lambda q: q[:, 22],
}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    students = pd.read_csv(STUDENT_PATH, sep="\t", low_memory=False)
    schools = pd.read_csv(SCHOOL_PATH, sep="\t", low_memory=False)
    return students, schools


def school_features(schools: pd.DataFrame) -> pd.DataFrame:
    """Official school-level urbanicity and free-lunch share (school file)."""

    out = pd.DataFrame({"schid": schools["schid"].astype(int)})
    out["school_urban"] = schools["var1"].astype(float)
    official = schools[[SCHOOL_FL_VARS[g] for g in GRADES]].mean(axis=1, skipna=True)
    out["school_fl_official"] = official / 100.0
    out["school_fl_n_grades"] = schools[[SCHOOL_FL_VARS[g] for g in GRADES]].notna().sum(axis=1)
    return out


def _first_nonnull(values: pd.Series):
    valid = values.dropna()
    return float(valid.iloc[0]) if len(valid) else np.nan


def prepare_grade(
    students: pd.DataFrame,
    grade: str,
    score_col: str,
    school_feat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Assigned-student table, tested-student table with z-scores, units, info."""

    classtype = students[f"{grade}classtype"]
    schid = students[f"{grade}schid"]
    tchid = students[f"{grade}tchid"]
    keep = classtype.notna() & schid.notna() & tchid.notna()
    sub = students.loc[keep].copy()
    sub["unit_id"] = (
        f"{grade}_" + sub[f"{grade}schid"].astype(int).astype(str)
        + "_" + sub[f"{grade}tchid"].astype(int).astype(str)
    )
    tested = sub[sub[score_col].notna()].copy()
    mean = float(tested[score_col].mean())
    sd = float(tested[score_col].std(ddof=1))
    tested["z"] = (tested[score_col].astype(float) - mean) / sd

    info = {
        "grade": grade,
        "score_col": score_col,
        "n_assigned_with_class": int(len(sub)),
        "n_tested": int(len(tested)),
        "score_mean": mean,
        "score_sd": sd,
        "n_units_assigned": int(sub["unit_id"].nunique()),
        "n_units_with_test": int(tested["unit_id"].nunique()),
    }

    # Student-computed school-grade free-lunch share (fallback only).
    fl = sub[sub[f"{grade}freelunch"].notna()]
    school_grade_fl = (
        fl.assign(is_fl=(fl[f"{grade}freelunch"] == 1).astype(float))
        .groupby(f"{grade}schid")["is_fl"]
        .mean()
    )
    school_grade_fl.index = school_grade_fl.index.astype(int)

    rows = []
    for unit, grp in sub.groupby("unit_id", sort=True):
        tgrp_rows = int((tested["unit_id"] == unit).sum())
        row = {
            "unit_id": unit,
            "grade": grade,
            "schid": int(grp[f"{grade}schid"].iloc[0]),
            "tchid": int(grp[f"{grade}tchid"].iloc[0]),
            "classtype": int(_first_nonnull(grp[f"{grade}classtype"])),
            "classsize": _first_nonnull(grp[f"{grade}classsize"]),
            "tested_n": tgrp_rows,
            "assigned_n": int(len(grp)),
        }
        for attr in ("tgen", "trace", "thighdegree", "tyears"):
            row[f"teacher_{attr}"] = _first_nonnull(grp[f"{grade}{attr}"])
            row[f"teacher_{attr}_imputed"] = bool(grp[f"{grade}{attr}"].isna().all())
        gender = grp["gender"]
        race = grp["race"]
        lunch = grp[f"{grade}freelunch"]
        row["share_female_n"] = int(gender.notna().sum())
        row["share_black_n"] = int(race.notna().sum())
        row["share_freelunch_n"] = int(lunch.notna().sum())
        row["share_female"] = (
            float((gender == 2).sum()) / row["share_female_n"]
            if row["share_female_n"] else np.nan
        )
        row["share_black"] = (
            float((race == 2).sum()) / row["share_black_n"]
            if row["share_black_n"] else np.nan
        )
        row["share_freelunch"] = (
            float((lunch == 1).sum()) / row["share_freelunch_n"]
            if row["share_freelunch_n"] else np.nan
        )
        rows.append(row)
    units = pd.DataFrame(rows).set_index("unit_id", drop=False)
    units = units.merge(school_feat, on="schid", how="left", validate="many_to_one")
    units["school_fl_student"] = units["schid"].map(school_grade_fl)

    for attr in ("tgen", "trace", "thighdegree", "tyears"):
        col = f"teacher_{attr}"
        if units[col].isna().any():
            units.loc[units[col].isna(), col] = units[col].median()
    for col in ("share_female", "share_black", "share_freelunch"):
        if units[col].isna().any():
            units.loc[units[col].isna(), col] = units[col].median()
    return sub, tested, units, info


def build_dataset(
    *,
    score_name: str = "math",
    min_tested: int = 5,
    include_aide: bool = True,
    qstar: str = "control",
    study: str | None = None,
) -> tuple[AppliedDataset, pd.DataFrame, dict]:
    """Build the classroom-grade dataset exactly as pre-registered."""

    students, schools = load_raw()
    school_feat = school_features(schools)
    suffix = {"math": "tmathss", "reading": "treadss"}[score_name]
    study = study or f"project_star_{score_name}"

    unit_tables, tested_tables, grade_infos, missingness = [], [], [], []
    benchmark_parts = []
    for grade in GRADES:
        sub, tested, units, info = prepare_grade(
            students, grade, f"{grade}{suffix}", school_feat
        )
        unit_tables.append(units)
        tested_tables.append(tested[["unit_id", "z"]].copy())
        grade_infos.append(info)
        for arm, label in ((1, "small"), (0, "regular")):
            grp = sub[(sub[f"{grade}classtype"] == 1) == (arm == 1)]
            missingness.append(
                {
                    "grade": grade,
                    "arm": label,
                    "n_assigned": int(len(grp)),
                    "math_missing_share": float(grp[f"{grade}tmathss"].isna().mean()),
                    "reading_missing_share": float(grp[f"{grade}treadss"].isna().mean()),
                }
            )
        if qstar == "control":
            keep_units = set(sub[sub[f"{grade}classtype"].isin([2, 3])]["unit_id"])
        elif qstar == "all":
            keep_units = set(sub["unit_id"])
        else:
            raise ValueError("qstar must be 'control' or 'all'")
        bench_rows = tested["unit_id"].isin(keep_units)
        benchmark_parts.append(
            weighted_quantiles(tested.loc[bench_rows, "z"], None, midpoint_levels())
        )

    units_all = pd.concat(unit_tables, axis=0)
    tested_all = pd.concat(tested_tables, axis=0)

    units = units_all[units_all["tested_n"] >= min_tested].copy()
    dropped_below = int((units_all["tested_n"] < min_tested).sum())
    dropped_aide = 0
    if not include_aide:
        dropped_aide = int((units["classtype"] == 3).sum())
        units = units[units["classtype"].isin([1, 2])].copy()
    units = units.sort_values("unit_id").reset_index(drop=True)

    tested_keep = tested_all[tested_all["unit_id"].isin(set(units["unit_id"]))]
    unit_ids, q_matrix = quantile_matrix(
        tested_keep["z"].to_numpy(),
        np.ones(len(tested_keep)),
        tested_keep["unit_id"].to_numpy(),
        levels=midpoint_levels(),
    )
    units = units.set_index("unit_id").loc[pd.Index(unit_ids)]
    units.index.name = "unit_id"
    units = units.reset_index()

    q_star = np.mean(np.vstack(benchmark_parts), axis=0)

    raw_mod = units["school_fl_official"].to_numpy(dtype=float)
    n_mod_fallback = int((~np.isfinite(raw_mod)).sum())
    if n_mod_fallback:
        raw_mod = np.where(
            np.isfinite(raw_mod), raw_mod, units["school_fl_student"].to_numpy(dtype=float)
        )
        if not np.all(np.isfinite(raw_mod)):
            raw_mod = np.where(
                np.isfinite(raw_mod), raw_mod, np.nanmedian(raw_mod)
            )

    X = np.column_stack(
        [
            ecdf_moderator(raw_mod),
            (units["teacher_tgen"].to_numpy() == 2).astype(float),
            (units["teacher_trace"].to_numpy() != 1).astype(float),
            (units["teacher_thighdegree"].to_numpy() >= 3).astype(float),
            units["teacher_tyears"].to_numpy(dtype=float),
            units["school_urban"].to_numpy(dtype=float),
            units["share_female"].to_numpy(dtype=float),
            units["share_black"].to_numpy(dtype=float),
            units["share_freelunch"].to_numpy(dtype=float),
        ]
    )
    A = (units["classtype"].to_numpy() == 1).astype(np.int64)

    feature_names = [
        "moderator_school_fl_ecdf",
        "teacher_female",
        "teacher_nonwhite",
        "teacher_graduate_degree",
        "teacher_years",
        "school_urbanicity",
        "share_female",
        "share_black",
        "share_freelunch",
    ]
    meta = {
        "unit_definition": "(grade, g?schid, g?tchid); one classroom-grade",
        "score": score_name,
        "standardization": "within-grade student z-score over all tested students",
        "min_tested": int(min_tested),
        "include_aide": bool(include_aide),
        "qstar": qstar,
        "sources": SOURCES,
        "grade_info": grade_infos,
        "missingness_by_arm_grade": missingness,
        "dropped_units_below_min_tested": dropped_below,
        "dropped_units_aide": dropped_aide,
        "moderator_fallback_units": n_mod_fallback,
        "moderator_raw_name": (
            "school free/reduced-lunch share (official school file, mean across grades)"
        ),
        "q_star_definition": (
            "equal-weight average over grades of pooled control-student quantile vectors"
            if qstar == "control"
            else "equal-weight average over grades of pooled all-student quantile vectors"
        ),
    }
    ds = AppliedDataset(
        study=study,
        X=X,
        A=A,
        Q=q_matrix,
        moderator_raw=raw_mod,
        q_star=q_star,
        feature_names=feature_names,
        meta=meta,
    )
    return ds, units, meta


def aipw_details(model, ds: AppliedDataset, treatment=None) -> dict:
    """Per-functional AIPW marginal and 4-bin contrast with influence-function SEs.

    ``treatment`` must be the vector the model was actually fitted on (the
    permuted one for placebo runs); it defaults to the dataset's own A.
    """

    from wasserstein_causal_forests.cwdb.dr_calibration import aipw_scores, hajek_bin_means
    from wasserstein_causal_forests.g3.dgps import moderator_bins

    a = ds.A if treatment is None else np.asarray(treatment, dtype=float)
    functionals = make_functionals(ds.q_star)
    functionals.update(EXTRA_FUNCTIONALS)
    bins = moderator_bins(ds.X)
    out = {}
    for name, h in functionals.items():
        observed = np.asarray(h(ds.Q), dtype=float)
        mu = {}
        for arm in (0, 1):
            parts = model.oof_particles_[arm]
            flat = parts.reshape(-1, parts.shape[-1])
            vals = np.asarray(h(flat), dtype=float).reshape(parts.shape[0], parts.shape[1])
            mu[arm] = vals.mean(axis=1)
        scores = aipw_scores(observed, mu[0], mu[1], model.ehat_train_, a)
        bin_means = hajek_bin_means(scores, bins, 4)
        bin_se = []
        for b in range(4):
            rows = scores[bins == b]
            bin_se.append(
                float(np.std(rows) / np.sqrt(rows.size)) if rows.size else None
            )
        out[name] = {
            "marginal": float(np.mean(scores)),
            "marginal_if_se": float(np.std(scores) / np.sqrt(scores.size)),
            "bin_contrasts": [None if np.isnan(v) else float(v) for v in bin_means],
            "bin_if_se": bin_se,
            "bin_n": [int((bins == b).sum()) for b in range(4)],
        }
    return out


def run_and_save(
    ds: AppliedDataset,
    out_path: Path,
    *,
    random_state: int = 0,
    treatment_override=None,
    tag: str = "run",
    **kwargs,
) -> dict:
    """Fit the frozen WCF spec, add recomputed SEs, and persist a JSON record."""

    t0 = time.time()
    model, results = run_wcf(
        ds,
        random_state=random_state,
        extra_functionals=dict(EXTRA_FUNCTIONALS),
        treatment_override=treatment_override,
        **kwargs,
    )
    results["wall_seconds"] = round(time.time() - t0, 1)
    results["tag"] = tag
    results["aipw_details"] = aipw_details(model, ds, treatment=treatment_override)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    return results
