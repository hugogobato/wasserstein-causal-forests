#!/usr/bin/env python3
"""Build two robustness dataset variants for the primary study.

Both are post-hoc checks, clearly separated from the pre-registered primary run:

1. `data_yearfe_2000_2016` adds the (z-scored) calendar year as a ninth X
   column so that the cross-fitted propensity and the tree can separate
   minimum-wage events from national wage-growth years. This deliberately
   departs from the "all covariates lagged" rule and is reported as a
   robustness check only.

2. `data_fd_2000_2016` replaces the outcome with the within-state year-over-year
   change of every quantile (isotonic-projected back to nondecreasing rows),
   treating the zero vector as the primary reference and a linear compression
   ramp as a second benchmark.

Run from the repository root:

    PYTHONPATH=src python research/applied/primary_state_minwage/build_variants.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "results" / "applied_study_exploration" / "primary_state_minwage"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(STUDY_DIR))
from wasserstein_causal_forests.applied.adapter import (  # noqa: E402
    AppliedDataset,
    ecdf_moderator,
    grid_weights,
    midpoint_levels,
)

from build_dataset import policy_panel, annual_cpi, ensure_policy, STATE_ABB  # noqa: E402

K = 25
GRID = midpoint_levels(K)
PRIMARY = DATA_ROOT / "data"
PANEL = DATA_ROOT / "data" / "state_year_panel.csv"

FEATURE_NAMES = [
    "moderator_median_wage_lag_ecdf",
    "log_real_min_wage_lag",
    "median_wage_lag",
    "p10_wage_lag",
    "below_share_lag",
    "log_earner_pop_lag",
    "hsl_share_lag",
    "female_share_lag",
]


def base_frame(year_lo: int = 2000, year_hi: int = 2016) -> pd.DataFrame:
    panel = pd.read_csv(PANEL).sort_values(["statefips", "year"]).reset_index(drop=True)
    cpi, _ = annual_cpi(sorted(panel["year"].unique()))
    policy = policy_panel(
        Path("/tmp/opencode/wcf_primary_cache/mw_state/mw_state_annual.dta")
        if Path("/tmp/opencode/wcf_primary_cache/mw_state/mw_state_annual.dta").exists()
        else ensure_policy(Path("/tmp/opencode/wcf_primary_cache")),
        cpi,
    )
    keep = ["statefips", "year", "A1", "A2", "A3", "rmw", "min_mw", "min_fed_mw", "fed_only"]
    df = panel.merge(policy[keep], on=["statefips", "year"], how="left")
    grouped = df.groupby("statefips", group_keys=False)
    df["rmw_lag"] = grouped["rmw"].shift(1)
    df["log_mw_lag"] = np.log(df["rmw_lag"])
    df["med_wage_lag"] = grouped["q13"].shift(1)
    df["p10_wage_lag"] = grouped["q3"].shift(1)
    df["w_sum_lag"] = grouped["w_sum"].shift(1)
    df["below_share_lag"] = grouped["below_share"].shift(1)
    df["hsl_share_lag"] = grouped["hsl_share"].shift(1)
    df["female_share_lag"] = grouped["female_share"].shift(1)
    df = df[(df["year"] >= year_lo) & (df["year"] <= year_hi)].copy()
    df = df.dropna(subset=[
        "med_wage_lag", "p10_wage_lag", "w_sum_lag", "below_share_lag",
        "hsl_share_lag", "female_share_lag", "rmw_lag", "A1",
    ])
    return df.reset_index(drop=True)


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    z_cols = [
        "log_mw_lag", "med_wage_lag", "p10_wage_lag", "below_share_lag",
        "w_sum_lag", "hsl_share_lag", "female_share_lag",
    ]
    z = df[z_cols].to_numpy(dtype=float).copy()
    z[:, 4] = np.log(z[:, 4])
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    moderator = ecdf_moderator(df["med_wage_lag"].to_numpy(dtype=float))
    return np.column_stack([moderator, z])


def build_yearfe(df: pd.DataFrame) -> None:
    X = design_matrix(df)
    year_z = df["year"].to_numpy(dtype=float)
    year_z = (year_z - year_z.mean()) / year_z.std()
    X = np.column_stack([X, year_z])
    Q = df[[f"q{k + 1}" for k in range(K)]].to_numpy(dtype=float)
    out = DATA_ROOT / "data_yearfe_2000_2016"
    AppliedDataset(
        study="primary_state_minwage_yearfe_2000_2016",
        X=X,
        A=df["A1"].to_numpy(dtype=np.int64),
        Q=Q,
        moderator_raw=df["med_wage_lag"].to_numpy(dtype=float),
        q_star=np.load(PRIMARY / "benchmarks.npz")["nordic_scaled"],
        feature_names=FEATURE_NAMES + ["year_z"],
        meta={
            "label": "post-hoc robustness: calendar year added to X",
            "departure": "year is not lagged; reported as robustness only",
            "base_window": [2000, 2016],
        },
    ).save(out)
    df.to_csv(out / "panel_with_lags.csv", index=False)
    np.savez(
        out / "benchmarks.npz",
        **{k: np.load(PRIMARY / "benchmarks.npz")[k] for k in
           ["nordic_raw", "nordic_scaled", "us2016", "scalar"]},
    )
    print(f"wrote {out} n={len(df)} A1={int(df['A1'].sum())}")


def build_fd(df: pd.DataFrame, delta: float) -> None:
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    idx = np.arange(K, dtype=float)
    diffs = []
    for _, row in df.iterrows():
        q_now = row[[f"q{k + 1}" for k in range(K)]].to_numpy(dtype=float)
        q_prev = row[[f"q{k + 1}_lag" for k in range(K)]].to_numpy(dtype=float)
        d = q_now - q_prev
        diffs.append(iso.fit_transform(idx, d))
    Q = np.vstack(diffs)
    out = DATA_ROOT / "data_fd_2000_2016"
    ramp = delta * (1.0 - GRID)
    AppliedDataset(
        study="primary_state_minwage_fd_2000_2016",
        X=design_matrix(df),
        A=df["A1"].to_numpy(dtype=np.int64),
        Q=Q,
        moderator_raw=df["med_wage_lag"].to_numpy(dtype=float),
        q_star=np.zeros(K),
        feature_names=FEATURE_NAMES,
        meta={
            "label": "post-hoc robustness: within-state year-over-year quantile change",
            "q_star": "zero vector; ramp benchmark in benchmarks.npz",
            "ramp_delta": float(delta),
            "base_window": [2000, 2016],
        },
    ).save(out)
    df.to_csv(out / "panel_with_lags.csv", index=False)
    np.savez(
        out / "benchmarks.npz",
        qstar_zero=np.zeros(K),
        qstar_ramp=ramp,
        ramp_delta=np.array([delta]),
    )
    print(f"wrote {out} n={len(df)} A1={int(df['A1'].sum())} ramp_delta={delta:.3f}")


def main() -> None:
    df = base_frame()
    # add lagged quantiles for the first-difference variant
    raw = pd.read_csv(PANEL).sort_values(["statefips", "year"])
    raw["key"] = raw["statefips"].astype(str) + "_" + raw["year"].astype(str)
    df["key"] = df["statefips"].astype(str) + "_" + df["year"].astype(str)
    qcols = [f"q{k + 1}" for k in range(K)]
    lagged = raw.groupby("statefips")[qcols].shift(1)
    raw = pd.concat([raw[["key"]], lagged], axis=1).rename(
        columns={c: f"{c}_lag" for c in qcols}
    )
    df = df.merge(raw, on="key", how="left")
    assert df[[f"{c}_lag" for c in qcols]].notna().all().all()

    build_yearfe(df.drop(columns=[f"{c}_lag" for c in qcols]))
    treated = df[df["A1"] == 1]
    delta = (treated["rmw"] - treated["rmw_lag"]).mean()
    build_fd(df, float(delta))

    summary = {
        "fd_ramp_delta": float(delta),
        "n_treated": int(df["A1"].sum()),
        "n_rows": int(len(df)),
        "years": [int(df.year.min()), int(df.year.max())],
    }
    (DATA_ROOT / "variants_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
