"""Post-hoc robustness dataset: school-saturated covariate matrix.

The primary X contains only coarse school aggregates (urbanicity, free-lunch
share), so its logistic propensity is not the exact within-school assignment
probability. Adding school fixed effects makes X saturate the randomisation
unit, so the cross-fitted propensity can recover each school's small-class
share exactly. This is an unregistered post-hoc check, labelled as such.

    PYTHONPATH=src python research/applied/project_star/07_build_school_saturated.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_common import RESULTS_DIR  # noqa: E402

from wasserstein_causal_forests.applied.adapter import AppliedDataset  # noqa: E402

DATA = RESULTS_DIR / "data"


def main() -> None:
    ds = AppliedDataset.load(DATA)
    units = pd.read_csv(DATA / "units_math_primary.csv")
    assert len(units) == ds.X.shape[0]

    dummies = pd.get_dummies(units["schid"], prefix="school", drop_first=True)
    dummy_names = dummies.columns.tolist()
    X = np.column_stack([ds.X[:, :1], dummies.to_numpy(dtype=float), ds.X[:, 1:]])
    feature_names = [ds.feature_names[0]] + dummy_names + ds.feature_names[1:]
    assert X.shape[1] == len(feature_names)

    out = replace(
        ds,
        X=X,
        feature_names=feature_names,
        meta={
            **ds.meta,
            "variant": "school_saturated_X",
            "status": "post-hoc robustness (not in the pre-registration)",
            "note": (
                "adds 79 school fixed effects so the cross-fitted propensity can "
                "recover the within-school assignment fraction"
            ),
        },
    )
    out.save(DATA / "math_schoolsat")
    summary = {"n": int(X.shape[0]), "p": int(X.shape[1]), "n_school_dummies": len(dummy_names)}
    with open(DATA / "math_schoolsat" / "build.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
