"""Household-level reference checks for the Egger Kenya study (independent of WCF).

Reproduces the trusted scalar directly from the household endline data:
  - weighted difference in mean per-capita nondurable consumption by village arm
  - weighted difference in mean log(1+consumption)
  - same for total consumption
with village-clustered standard errors. This validates the treatment merge and
the sign/magnitude expectation before trusting the village-quantile WCF output.

Run from the repo root:
    /usr/bin/python3 research/applied/egger_kenya/hh_reference_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

BASE = Path("/tmp/opencode/wcf_scout/egger/replication_package")
OUT = Path("results/applied_study_exploration/egger_kenya")


def cluster_robust_se(y: np.ndarray, a: np.ndarray, w: np.ndarray, cluster: np.ndarray) -> dict:
    """Weighted difference in means with cluster-robust (village) SE."""

    def wmean(mask: np.ndarray) -> float:
        ww = w[mask]
        return float(np.sum(ww * y[mask]) / np.sum(ww))

    est = wmean(a == 1) - wmean(a == 0)
    # influence-function representation of the Hajek difference
    psi = np.where(a == 1, w / w[a == 1].sum(), 0.0) * (y - wmean(a == 1)) - np.where(
        a == 0, w / w[a == 0].sum(), 0.0
    ) * (y - wmean(a == 0))
    clusters, inv = np.unique(cluster, return_inverse=True)
    sums = np.bincount(inv, weights=psi, minlength=len(clusters))
    var = float(np.sum(sums**2))
    return {
        "estimate": est,
        "se_clustered": float(np.sqrt(var)),
        "t": float(est / np.sqrt(var)),
        "n": int(y.size),
        "n_clusters": int(len(clusters)),
        "mean_treated": wmean(a == 1),
        "mean_control": wmean(a == 0),
    }


def main() -> None:
    df, _ = pyreadstat.read_dta(
        str(BASE / "code/data/GE_HHLevel_ECMA.dta"),
        usecols=[
            "village_code",
            "treat",
            "eligible",
            "hhweight_EL",
            "nondurables_exp_pc",
            "durables_exp_pc",
        ],
    )
    df = df.dropna(subset=["nondurables_exp_pc", "hhweight_EL", "treat"])
    w = df.hhweight_EL.to_numpy(float)
    a = df.treat.to_numpy(int)
    cl = df.village_code.to_numpy()

    out = {}
    for subset_name, mask in [
        ("all", np.ones(len(df), dtype=bool)),
        ("eligible", df.eligible.to_numpy() == 1),
        ("ineligible", df.eligible.to_numpy() == 0),
        ("hi_sat1", df.merge(
            pd.read_csv(OUT / "village_covariates.csv")[["village_code", "hi_sat"]],
            on="village_code", how="left",
        ).hi_sat.to_numpy() == 1),
        ("hi_sat0", df.merge(
            pd.read_csv(OUT / "village_covariates.csv")[["village_code", "hi_sat"]],
            on="village_code", how="left",
        ).hi_sat.to_numpy() == 0),
    ]:
        sub = {}
        for name, y in {
            "nondurables_exp_pc": df.nondurables_exp_pc.to_numpy(float),
            "log1p_nondurables_exp_pc": np.log1p(df.nondurables_exp_pc.to_numpy(float)),
            "total_exp_pc": (df.nondurables_exp_pc + df.durables_exp_pc.fillna(0)).to_numpy(float),
        }.items():
            sub[name] = cluster_robust_se(y[mask], a[mask], w[mask], cl[mask])
            print(subset_name, name, json.dumps(sub[name], indent=2))
        out[subset_name] = sub

    with open(OUT / "hh_reference_check.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
