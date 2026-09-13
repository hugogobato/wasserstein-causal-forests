#!/usr/bin/env python3
"""Persist the propensity-factory calibration diagnostic cited in the paper.

The sensitivity study stores only the mean fitted propensity per cell, so the
diagnostic behind the gradient-boosting negative finding is recomputed here
from the frozen DGP definitions and written to
`results/wcf_sensitivity/analysis/propensity_calibration_diagnostic.json`.

Configuration matches the AIPW layer exactly: five stratified folds generated
with `stratified_folds(a, 5, random_state)`, each fold model seeded with
`random_state + fold`, and `random_state = cell seed + 31`. The run below uses
cell seed 0, n = 1000, and the three effectful symmetric assignment regimes.

Usage:

    python3 research/checks/wcf_sensitivity_propensity_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.cwdb.cross_fitted import (  # noqa: E402
    stratified_folds,
)
from wasserstein_causal_forests.cwdb.dr_calibration import (  # noqa: E402
    hist_gradient_boosting_propensity_factory,
    logistic_propensity_factory,
    random_forest_propensity_factory,
)
from wasserstein_causal_forests.g3 import runner  # noqa: E402,F401
from wasserstein_causal_forests.g3.dgps import build_dgp  # noqa: E402
from wasserstein_causal_forests.g3.sensitivity_dgps import (  # noqa: E402
    register_sensitivity_dgps,
)

AIPW_N_FOLDS = 5
CLIP = (0.02, 0.98)
SEED = 0
N_ROWS = 1000
DGPS = ("SYM-NL", "SYM-MU", "SYM-LIN")
OUTPUT = (
    ROOT
    / "results"
    / "wcf_sensitivity"
    / "analysis"
    / "propensity_calibration_diagnostic.json"
)


def _oof_predictions(X, treatment, factory):
    random_state = SEED + 31
    folds = stratified_folds(treatment, AIPW_N_FOLDS, random_state)
    predictions = np.empty(X.shape[0])
    for fold in range(AIPW_N_FOLDS):
        held_out = folds == fold
        if not np.any(held_out):
            continue
        model = factory(random_state + fold)
        model.fit(X[~held_out], treatment[~held_out])
        predictions[held_out] = model.predict_proba(X[held_out])[:, 1]
    return predictions


def main() -> int:
    register_sensitivity_dgps()
    factories = {
        "logistic": logistic_propensity_factory,
        "gradient_boosting": hist_gradient_boosting_propensity_factory,
        "random_forest": random_forest_propensity_factory,
    }
    report: dict[str, object] = {
        "configuration": {
            "n_rows": N_ROWS,
            "cell_seed": SEED,
            "aipw_random_state": SEED + 31,
            "n_folds": AIPW_N_FOLDS,
            "clip": list(CLIP),
            "regimes": list(DGPS),
            "note": (
                "out-of-fold predictions before clipping; the clip fractions "
                "count predictions below 0.02 or above 0.98"
            ),
        },
        "regimes": {},
    }
    for dgp_id in DGPS:
        sample = build_dgp(dgp_id, 25).sample(N_ROWS, seed=SEED)
        entry: dict[str, object] = {}
        for name, factory in factories.items():
            raw = _oof_predictions(sample.X, sample.treatment, factory)
            truth = np.asarray(sample.propensity, dtype=float)
            entry[name] = {
                "auc_against_treatment": round(
                    float(roc_auc_score(sample.treatment, raw)), 6
                ),
                "correlation_with_true_propensity": round(
                    float(np.corrcoef(raw, truth)[0, 1]), 6
                ),
                "rmse_against_true_propensity": round(
                    float(np.sqrt(np.mean((raw - truth) ** 2))), 6
                ),
                "min_raw": round(float(raw.min()), 6),
                "max_raw": round(float(raw.max()), 6),
                "clip_low_fraction": round(float(np.mean(raw < CLIP[0])), 6),
                "clip_high_fraction": round(float(np.mean(raw > CLIP[1])), 6),
                "clip_total_fraction": round(
                    float(np.mean((raw < CLIP[0]) | (raw > CLIP[1]))), 6
                ),
            }
        report["regimes"][dgp_id] = entry
        for name in factories:
            values = entry[name]
            print(
                f"{dgp_id:8s} {name:18s} raw=[{values['min_raw']:.3f},"
                f"{values['max_raw']:.3f}] clip={values['clip_total_fraction']:.3f} "
                f"rmse={values['rmse_against_true_propensity']:.3f}"
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
