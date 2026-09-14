#!/usr/bin/env python3
"""Build the primary applied dataset for the WCF paper.

Study: US state-year hourly-wage distributions 1979-2022 and state minimum-wage
increases. See `preregistration.md` in this directory for the frozen choices.

The script is self-contained: it downloads the raw CPS MORG files and the
Vaghul-Zipperer minimum-wage panel when they are missing, builds the state-year
quantile panel, applies the frozen sample filters and treatment definitions, and
writes one `AppliedDataset` per year window through the shared adapter.

Run from the repository root:

    PYTHONPATH=src python research/applied/primary_state_minwage/build_dataset.py

Caches (large raw files) go to `--cache`, default /tmp/opencode/wcf_primary_cache;
the script reuses the earlier agent downloads in /tmp/opencode/morg_annual and
/tmp/opencode/mw_state when present.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get(
        "WCF_STUDY_OUT",
        REPO_ROOT / "results" / "applied_study_exploration" / "primary_state_minwage",
    )
)

sys.path.insert(0, str(REPO_ROOT / "src"))
from wasserstein_causal_forests.applied.adapter import (  # noqa: E402
    AppliedDataset,
    ecdf_moderator,
    midpoint_levels,
    weighted_quantiles,
)

MORG_BASE = "https://data.nber.org/morg/annual/morg{yy:02d}.dta"
VZ_URL = (
    "https://github.com/benzipperer/historicalminwage/releases/download/"
    "v1.4.0/mw_state_stata.zip"
)
FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
NORDIC_GRID = (
    REPO_ROOT
    / "results"
    / "applied_study_exploration"
    / "nordic_benchmark"
    / "nordic_qstar_k25_grid.csv"
)
CPI_QUARTERLY = STUDY_DIR / "cpi_urs_quarterly_2016base.csv"

K = 25
GRID = midpoint_levels(K)

# CPS state code -> FIPS, derived empirically from morg89/morg90 (both variables
# present; mapping is bijective over the 51 units). See preregistration.md.
CPS_STATE_TO_FIPS = {
    63: 1, 94: 2, 86: 4, 71: 5, 93: 6, 84: 8, 16: 9, 51: 10, 53: 11, 59: 12,
    58: 13, 95: 15, 82: 16, 33: 17, 32: 18, 42: 19, 47: 20, 61: 21, 72: 22,
    11: 23, 52: 24, 14: 25, 34: 26, 41: 27, 64: 28, 43: 29, 81: 30, 46: 31,
    88: 32, 12: 33, 22: 34, 85: 35, 21: 36, 56: 37, 44: 38, 31: 39, 73: 40,
    92: 41, 23: 42, 15: 44, 57: 45, 45: 46, 62: 47, 74: 48, 87: 49, 13: 50,
    54: 51, 91: 53, 55: 54, 35: 55, 83: 56,
}

STATE_ABB = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}

INDIVIDUAL_VARS = [
    "year", "state", "stfips", "intmonth", "age", "earnwt", "paidhre",
    "earnhre", "earnwke", "uhourse", "class", "class94",
    "I25a", "I25b", "I25c", "I25d",
    "earnhr", "uhours", "uearnwk",
    "gradeat", "gradecp", "grade92", "ihigrdc", "sex",
]

GRADE92_TO_YEARS = {
    31: 0.0, 32: 2.5, 33: 5.5, 34: 7.5, 35: 9.0, 36: 10.0, 37: 11.0,
    38: 12.0, 39: 12.0, 40: 13.0, 41: 14.0, 42: 14.0, 43: 16.0, 44: 18.0,
    45: 18.0, 46: 18.0,
}


def log(msg: str) -> None:
    print(f"[build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                with open(tmp, "wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
            tmp.replace(dest)
            return
        except Exception as error:  # noqa: BLE001
            last_error = error
            log(f"download retry {attempt + 1}/{retries} for {url}: {error}")
            time.sleep(3 + 3 * attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def ensure_morg(cache: Path, years: list[int]) -> Path:
    """Locate or fetch the annual MORG files for `years`."""

    target = cache / "morg_annual"
    target.mkdir(parents=True, exist_ok=True)
    legacy = Path("/tmp/opencode/morg_annual")
    todo = []
    for year in years:
        name = f"morg{year % 100:02d}.dta"
        dest = target / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            continue
        if (legacy / name).exists() and (legacy / name).stat().st_size > 1_000_000:
            os.link(legacy / name, dest)
            continue
        todo.append(year)
    if todo:
        log(f"downloading {len(todo)} MORG files: {todo}")
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(
                    _download, MORG_BASE.format(yy=year % 100), target / f"morg{year % 100:02d}.dta"
                ): year
                for year in todo
            }
            for future in futures:
                future.result()
    return target


def ensure_policy(cache: Path) -> Path:
    """Locate or fetch the Vaghul-Zipperer annual minimum-wage panel."""

    target = cache / "mw_state" / "mw_state_annual.dta"
    if target.exists():
        return target
    legacy = Path("/tmp/opencode/mw_state/mw_state_annual.dta")
    if legacy.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(legacy, target)
        return target
    log("downloading Vaghul-Zipperer minimum-wage panel")
    archive = cache / "mw_state_stata.zip"
    _download(VZ_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        with zf.open("mw_state_annual.dta") as handle:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(handle.read())
    return target


# ---------------------------------------------------------------------------
# deflator
# ---------------------------------------------------------------------------

def annual_cpi(extra_years: list[int] | None = None) -> tuple[pd.Series, str]:
    """CPI-U-RS annual means, base 2016 = 100, spliced with CPI-U after 2016."""

    q = pd.read_csv(CPI_QUARTERLY)
    ann = q.groupby("year")["cpi"].mean().sort_index()
    note = "CPI-U-RS quarterly series from Cengiz et al. (2019), annual means"
    need = [y for y in (extra_years or []) if y > 2016]
    if need:
        dest = Path("/tmp/opencode/cpi_cpiaucsl.csv")
        if not dest.exists():
            try:
                _download(FRED_CPI_URL, dest)
            except Exception as error:  # noqa: BLE001
                log(f"WARNING: FRED CPI-U download failed ({error}); "
                    f"cannot extend deflator beyond 2016")
                return ann, note
        raw = pd.read_csv(dest)
        raw.columns = [c.strip() for c in raw.columns]
        raw["date"] = pd.to_datetime(raw["DATE"] if "DATE" in raw else raw["observation_date"])
        raw["year"] = raw["date"].dt.year
        value_col = [c for c in raw.columns if c not in ("DATE", "observation_date", "year")][0]
        cpiu = raw.groupby("year")[value_col].mean()
        base = float(cpiu.loc[2016])
        for year in sorted(cpiu.index):
            if year > 2016:
                ann.loc[year] = 100.0 * float(cpiu.loc[year]) / base
        note += "; 2017+ spliced with FRED CPIAUCSL growth anchored at 2016"
    return ann.sort_index(), note


# ---------------------------------------------------------------------------
# policy panel
# ---------------------------------------------------------------------------

def policy_panel(policy_path: Path, cpi: pd.Series) -> pd.DataFrame:
    df, _ = pyreadstat.read_dta(str(policy_path))
    df = df[["statefips", "stateabb", "year", "min_fed_mw", "min_mw"]].copy()
    for col in ("min_fed_mw", "min_mw"):
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    df = df.sort_values(["statefips", "year"]).reset_index(drop=True)
    df["cpi"] = df["year"].map(cpi)
    df["eff_mw"] = df[["min_mw", "min_fed_mw"]].max(axis=1)
    df["rmw"] = df["eff_mw"] / (df["cpi"] / 100.0)
    grouped = df.groupby("statefips", group_keys=False)
    df["rmw_lag"] = grouped["rmw"].shift(1)
    df["min_mw_lag"] = grouped["min_mw"].shift(1)
    df["min_fed_lag"] = grouped["min_fed_mw"].shift(1)
    df["d_rmw"] = df["rmw"] - df["rmw_lag"]
    df["d_min"] = df["min_mw"] - df["min_mw_lag"]
    df["d_fed"] = df["min_fed_mw"] - df["min_fed_lag"]
    fed_only = (
        (df["d_fed"] > 0)
        & (df["min_mw_lag"] == df["min_fed_lag"])
        & (df["min_mw"] == df["min_fed_mw"])
    )
    df["fed_only"] = fed_only
    df["A1"] = ((df["d_rmw"] > 0.25) & (df["d_min"] > 0) & ~fed_only).astype(int)
    df["A2"] = (df["d_rmw"] > 0.25).astype(int)
    df["A3"] = (df["min_mw"] > df["min_fed_mw"]).astype(int)
    return df


# ---------------------------------------------------------------------------
# individual-year build
# ---------------------------------------------------------------------------

def process_year(path: Path, cpi: pd.Series, rmw_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One MORG file -> (state-year rows, national rows)."""

    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    available = [c for c in INDIVIDUAL_VARS if c in meta.column_names]
    df, _ = pyreadstat.read_dta(str(path), usecols=available)
    year = int(np.nanmedian(pd.to_numeric(df["year"], errors="coerce")))

    for col in available:
        if col not in ("state", "stfips"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "stfips" in df.columns and df["stfips"].notna().any():
        state = df["stfips"]
        if "state" in df.columns:
            mapped = df["state"].map(CPS_STATE_TO_FIPS)
            state = state.fillna(mapped)
    else:
        state = df["state"].map(CPS_STATE_TO_FIPS)
    df["statefips"] = pd.to_numeric(state, errors="coerce")

    n_raw = len(df)
    mask = (df["age"] >= 16) & (df["age"] <= 64) & (df["earnwt"] > 0) & df["statefips"].notna()
    df = df[mask].copy()

    if year >= 1994 and "class94" in df.columns and df["class94"].notna().any():
        df = df[~df["class94"].isin([6, 7])]
    elif "class" in df.columns and df["class"].notna().any():
        df = df[~df["class"].isin([5, 6])]
    df = df[df["paidhre"].isin([1, 2])].copy()

    wage_paid = df["earnhre"] / 100.0
    hours = df["uhourse"].where(df["uhourse"] > 0)
    wage_derived = (df["earnwke"] / hours).where(df["earnwke"] > 0)
    wage = wage_paid.where(df["paidhre"] == 1, wage_derived)

    cols = set(df.columns)
    hours_imp = (df["I25a"] > 0) if "I25a" in cols else pd.Series(False, index=df.index)
    earn_imp = (df["I25d"] > 0) if "I25d" in cols else pd.Series(False, index=df.index)
    wage_imp = (df["I25c"] > 0) if "I25c" in cols else pd.Series(False, index=df.index)

    month = df["intmonth"] if "intmonth" in cols else pd.Series(np.nan, index=df.index)
    if year == 1994 or year == 1995:
        zero_block = (year == 1994) | (month <= 8) | month.isna()
        hours_imp = hours_imp.where(~zero_block, False)
        earn_imp = earn_imp.where(~zero_block, False)
        wage_imp = wage_imp.where(~zero_block, False)
    if 1989 <= year <= 1993:
        hours_imp = pd.Series(False, index=df.index)
        earn_imp = pd.Series(False, index=df.index)
        wage_imp = pd.Series(False, index=df.index)
        if "earnhr" in cols:
            wage_imp = ((df["earnhr"].isna() | (df["earnhr"] == 0)) & (df["earnhre"] > 0))
        if "uhours" in cols:
            hours_imp = ((df["uhours"].isna() | (df["uhours"] == 0)) & (df["uhourse"] > 0))
        if "uearnwk" in cols:
            earn_imp = ((df["uearnwk"].isna() | (df["uearnwk"] == 0)) & (df["earnwke"] > 0))

    imputed = ((df["paidhre"] == 2) & (hours_imp | earn_imp)) | (
        (df["paidhre"] == 1) & wage_imp
    )
    wage = wage.where(~imputed)

    hgradecp = pd.Series(np.nan, index=df.index)
    if "gradeat" in cols and "gradecp" in cols:
        hgradecp = df["gradeat"].where(df["gradecp"] == 1)
        hgradecp = hgradecp.where(df["gradecp"] != 2, df["gradeat"] - 1)
    if "ihigrdc" in cols:
        hgradecp = hgradecp.where(hgradecp.notna(), df["ihigrdc"])
    if "grade92" in cols:
        hgradecp = hgradecp.where(df["grade92"].isna(), df["grade92"].map(GRADE92_TO_YEARS))
    hsl = (hgradecp <= 12).astype(float)
    female = (df["sex"] == 2).astype(float)

    df = df.assign(
        wage=wage, earnwt=df["earnwt"], hsl=hsl, female=female
    )
    n_age = len(df)
    before = len(df)
    df = df[np.isfinite(df["wage"]) & (df["wage"] > 0)]
    n_wage = len(df)

    df["wage_2016"] = df["wage"] / (cpi.loc[year] / 100.0)
    top_share = float(np.mean(df["wage_2016"] >= 99.99)) if len(df) else float("nan")

    rows = []
    for fips, group in df.groupby("statefips"):
        if len(group) < 20:
            continue
        rmw = rmw_map.get((int(fips), year), np.nan)
        w = group["earnwt"].to_numpy()
        q = weighted_quantiles(group["wage_2016"].to_numpy(), w, GRID)
        row = {"statefips": int(fips), "year": year, "n_obs": len(group), "w_sum": float(w.sum())}
        for k in range(K):
            row[f"q{k + 1}"] = float(q[k])
        row["below_share"] = (
            float(np.average(group["wage_2016"] < rmw, weights=w)) if np.isfinite(rmw) else np.nan
        )
        row["hsl_share"] = float(np.average(group["hsl"], weights=w))
        row["female_share"] = float(np.average(group["female"], weights=w))
        rows.append(row)

    nat_q = weighted_quantiles(df["wage_2016"].to_numpy(), df["earnwt"].to_numpy(), GRID)
    nat = {"year": year, "n_obs": len(df), "w_sum": float(df["earnwt"].sum()),
           "topcode_share": top_share}
    for k in range(K):
        nat[f"q{k + 1}"] = float(nat_q[k])

    log(
        f"{year}: raw {n_raw} -> age {n_age} -> wage {n_wage} | "
        f"states {df['statefips'].nunique()} | smallest state-year n "
        f"{int(df.groupby('statefips').size().min())}"
    )
    return pd.DataFrame(rows), pd.DataFrame([nat])


def build_raw_panel(cache: Path, years: list[int], force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_file = DATA_ROOT / "data" / "state_year_panel.csv"
    nat_file = DATA_ROOT / "data" / "national_quantiles.csv"
    if state_file.exists() and nat_file.exists() and not force:
        log(f"reusing cached panel {state_file}")
        return pd.read_csv(state_file), pd.read_csv(nat_file)

    morg_dir = ensure_morg(cache, years)
    policy_path = ensure_policy(cache)
    cpi, cpi_note = annual_cpi(sorted(set(years) | {2016}))
    log(f"deflator: {cpi_note}")
    policy = policy_panel(policy_path, cpi)
    rmw_map = {
        (int(row.statefips), int(row.year)): float(row.rmw)
        for row in policy.itertuples()
        if np.isfinite(row.rmw)
    }

    state_rows, nat_rows = [], []
    for year in years:
        path = morg_dir / f"morg{year % 100:02d}.dta"
        if not path.exists():
            log(f"WARNING: missing {path}; skipping {year}")
            continue
        s, n = process_year(path, cpi, rmw_map)
        state_rows.append(s)
        nat_rows.append(n)
    panel = pd.concat(state_rows, ignore_index=True)
    national = pd.concat(nat_rows, ignore_index=True)
    (DATA_ROOT / "data").mkdir(parents=True, exist_ok=True)
    panel.to_csv(state_file, index=False)
    national.to_csv(nat_file, index=False)
    log(f"wrote {state_file} ({panel.shape}) and {nat_file} ({national.shape})")
    return panel, national


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------

def load_benchmarks(national: pd.DataFrame, panel: pd.DataFrame) -> dict:
    nordic = pd.read_csv(NORDIC_GRID)
    nordic_raw = nordic["B_incY2024_4c_popw"].to_numpy(dtype=float)
    nat2016 = national[national["year"] == 2016].iloc[0]
    us2016 = np.array([nat2016[f"q{k + 1}"] for k in range(K)], dtype=float)
    pooled_median_2016 = float(np.interp(0.5, GRID, us2016))
    nordic_median = float(np.interp(0.5, GRID, nordic_raw))
    scalar = pooled_median_2016 / nordic_median
    return {
        "nordic_raw": nordic_raw,
        "nordic_scaled": nordic_raw * scalar,
        "scalar": float(scalar),
        "pooled_median_2016": pooled_median_2016,
        "nordic_median": nordic_median,
        "us2016": us2016,
    }


def assemble(
    panel: pd.DataFrame,
    benchmarks: dict,
    year_lo: int,
    year_hi: int,
    treatment: str,
) -> AppliedDataset:
    df = panel.sort_values(["statefips", "year"]).reset_index(drop=True)

    policy_path = ensure_policy(Path(os.environ.get("WCF_PRIMARY_CACHE", "/tmp/opencode/wcf_primary_cache")))
    cpi, _ = annual_cpi(sorted(df["year"].unique()))
    policy = policy_panel(policy_path, cpi)
    keep = ["statefips", "year", "A1", "A2", "A3", "rmw", "min_mw", "min_fed_mw", "fed_only"]
    df = df.merge(policy[keep], on=["statefips", "year"], how="left")

    grouped = df.groupby("statefips", group_keys=False)
    df["rmw_lag"] = grouped["rmw"].shift(1)
    df["log_mw_lag"] = np.log(df["rmw_lag"])
    lag_cols = {
        "med_wage_lag": "q13",
        "p10_wage_lag": "q3",
        "w_sum_lag": "w_sum",
        "below_share_lag": "below_share",
        "hsl_share_lag": "hsl_share",
        "female_share_lag": "female_share",
    }
    for name, source in lag_cols.items():
        df[name] = grouped[source].shift(1)

    df = df[(df["year"] >= year_lo) & (df["year"] <= year_hi)].copy()
    before = len(df)
    df = df.dropna(subset=[
        "med_wage_lag", "p10_wage_lag", "w_sum_lag", "below_share_lag",
        "hsl_share_lag", "female_share_lag", "rmw", "log_mw_lag", treatment,
    ])
    log(f"window {year_lo}-{year_hi}: {before} -> {len(df)} complete state-years")

    moderator_raw = df["med_wage_lag"].to_numpy(dtype=float)
    moderator = ecdf_moderator(moderator_raw)
    z_cols = [
        "log_mw_lag", "med_wage_lag", "p10_wage_lag", "below_share_lag",
        "w_sum_lag", "hsl_share_lag", "female_share_lag",
    ]
    z = df[z_cols].to_numpy(dtype=float).copy()
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    z[:, 4] = np.log(df["w_sum_lag"].to_numpy(dtype=float))
    z[:, 4] = (z[:, 4] - z[:, 4].mean()) / z[:, 4].std()
    X = np.column_stack([moderator, z])
    feature_names = [
        "moderator_median_wage_lag_ecdf",
        "log_real_min_wage_lag",
        "median_wage_lag",
        "p10_wage_lag",
        "below_share_lag",
        "log_earner_pop_lag",
        "hsl_share_lag",
        "female_share_lag",
    ]
    Q = df[[f"q{k + 1}" for k in range(K)]].to_numpy(dtype=float)
    A = df[treatment].to_numpy(dtype=np.int64)

    meta = {
        "study": "primary_state_minwage",
        "year_window": [int(year_lo), int(year_hi)],
        "treatment_primary": treatment,
        "n_states": int(df["statefips"].nunique()),
        "sources": {
            "microdata": "NBER CPS MORG annual, https://data.nber.org/morg/annual/morgYY.dta",
            "policy": "Vaghul-Zipperer v1.4.0, " + VZ_URL,
            "deflator": "CPI-U-RS from Cengiz et al. (2019) replica (doi:10.7910/DVN/TJCTC7), "
                        "2017+ spliced with " + FRED_CPI_URL,
            "nordic": "results/applied_study_exploration/nordic_benchmark/"
                      "nordic_qstar_k25_grid.csv column B_incY2024_4c_popw "
                      "(Eurostat ilc_di01 TC, PPS, income year 2024, DK/FI/NO/SE pop-weighted)",
        },
        "sample": (
            "age 16-64; earnwt>0; not self-employed; paidhre in {1,2}; "
            "non-imputed (Cengiz-Hirsch-Schumacher rules); wage=earnhre/100 if paid-hourly "
            "else earnwke/uhourse; positive finite wage; 2016 dollars via CPI-U-RS"
        ),
        "treatment_definitions": {
            "A1": "real effective MW rose > $0.25, own nominal rate rose, not federal-only",
            "A2": "real effective MW rose > $0.25 (federal-induced included)",
            "A3": "state rate strictly above the federal floor",
        },
        "q_star": {
            "primary": "nordic_scaled",
            "scalar": benchmarks["scalar"],
            "scalar_definition": "pooled US 2016 sample median / Nordic u=0.5 value",
            "pooled_median_2016": benchmarks["pooled_median_2016"],
            "nordic_median": benchmarks["nordic_median"],
        },
        "moderator": "lagged state median real wage, ecdf transform in X[:,0]",
        "feature_names": feature_names,
        "preregistration": "research/applied/primary_state_minwage/preregistration.md",
    }
    ds = AppliedDataset(
        study=f"primary_state_minwage_{year_lo}_{year_hi}",
        X=X,
        A=A,
        Q=Q,
        moderator_raw=moderator_raw,
        q_star=benchmarks["nordic_scaled"],
        feature_names=feature_names,
        meta=meta,
    ).validate()

    out_dir = DATA_ROOT / "data" if (year_lo, year_hi) == (2000, 2016) else (
        DATA_ROOT / f"data_{year_lo}_{year_hi}"
    )
    ds.save(out_dir)
    np.savez(
        out_dir / "benchmarks.npz",
        nordic_raw=benchmarks["nordic_raw"],
        nordic_scaled=benchmarks["nordic_scaled"],
        us2016=benchmarks["us2016"],
        scalar=benchmarks["scalar"],
    )
    treatments = df[["statefips", "year", "A1", "A2", "A3", "rmw", "fed_only"]]
    treatments.to_csv(out_dir / "treatments.csv", index=False)
    df.to_csv(out_dir / "panel_with_lags.csv", index=False)
    log(f"saved {out_dir} | n={len(df)} A1={int(A.sum())} A0={int((A == 0).sum())}")
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=Path(os.environ.get("WCF_PRIMARY_CACHE", "/tmp/opencode/wcf_primary_cache")))
    parser.add_argument("--years", type=str, default="1979-2024")
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--variants", type=str,
                        default="primary:2000-2016,ext:1980-2016,ext:2017-2022")
    args = parser.parse_args()

    lo, hi = (int(x) for x in args.years.split("-"))
    years = list(range(lo, hi + 1))
    panel, national = build_raw_panel(args.cache, years, force=args.rebuild_panel)
    benchmarks = load_benchmarks(national, panel)
    log(f"nordic scalar = {benchmarks['scalar']:.6f} "
        f"(pooled-2016 median {benchmarks['pooled_median_2016']:.4f} / "
        f"nordic median {benchmarks['nordic_median']:.2f})")

    for spec in args.variants.split(","):
        name, window = spec.split(":")
        y_lo, y_hi = (int(x) for x in window.split("-"))
        log(f"assembling variant {name}: {y_lo}-{y_hi}")
        assemble(panel, benchmarks, y_lo, y_hi, "A1")


if __name__ == "__main__":
    main()
