"""Naive/trusted-scalar checks for the Project STAR applied study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_common import GRADES, load_raw, school_features, prepare_grade  # noqa: E402


def student_panel(score_name: str = "math") -> pd.DataFrame:
    """One row per tested student with within-grade z-score, unit id, arm, grade."""

    students, schools = load_raw()
    sf = school_features(schools)
    suffix = {"math": "tmathss", "reading": "treadss"}[score_name]
    frames = []
    for grade in GRADES:
        sub, tested, _, _ = prepare_grade(students, grade, f"{grade}{suffix}", sf)
        keep = tested[["unit_id", "z"]].copy()
        arm = (
            sub[["unit_id", f"{grade}classtype"]]
            .drop_duplicates("unit_id")
            .set_index("unit_id")[f"{grade}classtype"]
        )
        keep["A"] = keep["unit_id"].map((arm == 1).astype(int))
        keep["grade"] = grade
        frames.append(keep)
    return pd.concat(frames, ignore_index=True)


def _cluster_ols(
    y: np.ndarray, X: np.ndarray, cluster: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and cluster-robust (CR1) standard errors."""

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cluster):
        rows = cluster == c
        xg = X[rows]
        ug = resid[rows]
        score = xg.T @ ug
        meat += np.outer(score, score)
    n, k = X.shape
    g = len(np.unique(cluster))
    correction = g / (g - 1) * (n - 1) / max(n - k, 1)
    cov = correction * xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.diag(cov))


def trusted_scalar(score_name: str = "math", min_tested: int = 5) -> dict:
    """Student- and unit-level naive small-class effects in within-grade SD units."""

    panel = student_panel(score_name)
    out: dict = {"score": score_name, "n_students": int(len(panel))}

    by_grade = {}
    for grade, grp in panel.groupby("grade", sort=False):
        X = np.column_stack([np.ones(len(grp)), grp["A"].to_numpy(dtype=float)])
        beta, se = _cluster_ols(grp["z"].to_numpy(dtype=float), X, grp["unit_id"].to_numpy())
        by_grade[grade] = {
            "n_treated": int(grp["A"].sum()),
            "n_control": int((grp["A"] == 0).sum()),
            "diff": float(beta[1]),
            "cluster_se": float(se[1]),
        }
    out["student_by_grade"] = by_grade

    grade_dummies = pd.get_dummies(panel["grade"], drop_first=True).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(panel)), panel["A"].to_numpy(dtype=float), grade_dummies])
    beta, se = _cluster_ols(panel["z"].to_numpy(dtype=float), X, panel["unit_id"].to_numpy())
    out["student_grade_fe"] = {
        "diff": float(beta[1]),
        "cluster_se": float(se[1]),
        "n_clusters": int(panel["unit_id"].nunique()),
    }

    # Unit-level mean of the class quantile vectors (mean functional plug-in).
    tested_per_unit = panel.groupby("unit_id").size()
    keep_units = tested_per_unit[tested_per_unit >= min_tested].index
    unit_means = (
        panel[panel["unit_id"].isin(set(keep_units))]
        .groupby(["unit_id", "grade", "A"])["z"]
        .mean()
        .reset_index()
    )
    X = np.column_stack(
        [
            np.ones(len(unit_means)),
            unit_means["A"].to_numpy(dtype=float),
            pd.get_dummies(unit_means["grade"], drop_first=True).to_numpy(dtype=float),
        ]
    )
    beta, se = _cluster_ols(
        unit_means["z"].to_numpy(dtype=float), X, unit_means["unit_id"].to_numpy()
    )
    out["unit_mean_grade_fe"] = {
        "diff": float(beta[1]),
        "hc_se": float(se[1]),
        "n_units": int(len(unit_means)),
        "treated_mean": float(unit_means.loc[unit_means.A == 1, "z"].mean()),
        "control_mean": float(unit_means.loc[unit_means.A == 0, "z"].mean()),
    }
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(trusted_scalar(), indent=2))
