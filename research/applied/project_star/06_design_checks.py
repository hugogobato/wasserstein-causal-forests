"""Design diagnostics: inner-n distributions, threshold feasibility, balance.

    PYTHONPATH=src python research/applied/project_star/06_design_checks.py

Writes results/applied_study_exploration/project_star/misc/design_checks.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_common import RESULTS_DIR  # noqa: E402

UNITS = RESULTS_DIR / "data" / "units_math_primary.csv"
OUT = RESULTS_DIR / "misc" / "design_checks.json"


def main() -> None:
    units = pd.read_csv(UNITS)
    units["arm"] = np.where(units["classtype"] == 1, "small", "regular")
    out: dict = {"n_units": int(len(units))}

    inner = {}
    for arm, grp in units.groupby("arm"):
        q = grp["tested_n"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        inner[arm] = {str(k): int(v) for k, v in q.items()}
        inner[f"{arm}_mean"] = float(grp["tested_n"].mean())
    out["tested_n_quantiles"] = inner

    thresholds = {}
    for thr in (13, 15, 17, 18, 20):
        small = int(((units["arm"] == "small") & (units["tested_n"] >= thr)).sum())
        regular = int(((units["arm"] == "regular") & (units["tested_n"] >= thr)).sum())
        thresholds[str(thr)] = {
            "small_units": small,
            "regular_units": regular,
            "adapter_min_arm_5_satisfied": small >= 5 and regular >= 5,
        }
    out["threshold_feasibility"] = thresholds

    small = units[units["arm"] == "small"]
    control = units[units["arm"] == "regular"]
    balance = {}
    for col in (
        "classsize",
        "assigned_n",
        "teacher_tyears",
        "share_female",
        "share_black",
        "share_freelunch",
        "school_fl_official",
        "school_urban",
    ):
        a = small[col].astype(float)
        b = control[col].astype(float)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        balance[col] = {
            "small_mean": float(a.mean()),
            "regular_mean": float(b.mean()),
            "std_diff": float((a.mean() - b.mean()) / pooled) if pooled > 0 else None,
        }
    out["balance"] = balance

    within = {}
    for col in (
        "teacher_tyears",
        "share_female",
        "share_black",
        "share_freelunch",
        "teacher_nonwhite",
        "teacher_graduate_degree",
    ):
        data = units.copy()
        if col == "teacher_nonwhite":
            data[col] = (data["teacher_trace"] != 1).astype(float)
        elif col == "teacher_graduate_degree":
            data[col] = (data["teacher_thighdegree"] >= 3).astype(float)
        school_dummies = pd.get_dummies(data["schid"], drop_first=True).to_numpy(dtype=float)
        X = np.column_stack(
            [np.ones(len(data)), data["arm"].eq("small").to_numpy(dtype=float), school_dummies]
        )
        y = data[col].astype(float).to_numpy()
        xtx_inv = np.linalg.inv(X.T @ X)
        beta = xtx_inv @ (X.T @ y)
        resid = y - X @ beta
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in np.unique(data["schid"]):
            rows = (data["schid"] == c).to_numpy()
            score = X[rows].T @ resid[rows]
            meat += np.outer(score, score)
        g = data["schid"].nunique()
        corr = g / (g - 1) * (len(data) - 1) / (len(data) - X.shape[1])
        cov = corr * xtx_inv @ meat @ xtx_inv
        within[col] = {
            "within_school_diff_small_minus_control": float(beta[1]),
            "cluster_se": float(np.sqrt(cov[1, 1])),
        }
    out["within_school_balance"] = within

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
