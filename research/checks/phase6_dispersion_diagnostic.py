#!/usr/bin/env python3
"""Phase 6 opening diagnostic: measure the under-dispersion mechanism.

For a few (regime, n, seed) combinations this script fits the frozen C-WDB-v1
booster, predicts particle clouds on a large test design, and compares the
implied reference-distance expectation against the quadrature truth per arm:

    bias_a = mean_x [ (1/M) sum_m d_W(p_m, nu*) - E{d_W(q(Y^a), nu*) | X} ].

A negative bias means the fitted cloud under-disperses the conditional law,
which biases every convex spread-sensitive functional low. The script writes
one JSON artefact; it fits no competitor and computes no ranking.
"""

from __future__ import annotations

import json
import os
import sys

for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.cwdb.model import CWDBRegressor  # noqa: E402
from wasserstein_causal_forests.g3.dgps import build_dgp  # noqa: E402
from wasserstein_causal_forests.g3.manifest import BOOSTING_BUDGET  # noqa: E402

OUTPUT = ROOT / "results" / "phase6" / "dispersion_diagnostic.json"

#: Pilot coordinates: outside every manifest, so nothing decisive depends on
#: this file. Five seeds keep the per-arm bias estimate stable to ~0.01.
REGIMES = ("D5", "D6", "D1")
N_TRAIN = 500
SEEDS = (0, 1, 2, 3, 4)
N_TEST = 4000


def main() -> int:
    records = []
    for dgp_id in REGIMES:
        dgp = build_dgp(dgp_id, 25)
        weights = dgp.grid.weights
        reference = dgp.grid.reference_quantiles()

        def distance(block: np.ndarray) -> np.ndarray:
            diff = block - reference
            return np.sqrt(np.sum(weights * diff * diff, axis=-1))

        for seed in SEEDS:
            train = dgp.sample(N_TRAIN, seed=seed)
            model = CWDBRegressor(
                random_state=seed, architecture="v1", sharing="partial",
                arm_shrinkage=5.0, init_sharing="per_arm", **BOOSTING_BUDGET,
            )
            model.fit(train.X, train.treatment, train.quantiles, weights)
            rng = np.random.default_rng(900_000 + seed)
            X_test = rng.uniform(-1.0, 1.0, size=(N_TEST, train.X.shape[1]))
            entry = {"dgp": dgp_id, "n_train": N_TRAIN, "seed": seed, "arms": {}}
            for arm in (0, 1):
                particles = model.predict_particles(X_test, arm)
                r_hat = distance(particles).mean(axis=1)
                r_true = dgp.reference_distance(X_test, arm)
                barycenter = particles.mean(axis=1)
                cloud_sd = float(np.sqrt(np.mean(
                    np.sum(weights * (particles - barycenter[:, None, :]) ** 2, axis=-1)
                )))
                entry["arms"][str(arm)] = {
                    "r_hat_mean": float(np.mean(r_hat)),
                    "r_true_mean": float(np.mean(r_true)),
                    "bias": float(np.mean(r_hat - r_true)),
                    "cloud_sd": cloud_sd,
                }
            records.append(entry)
            print(f"done {dgp_id} seed {seed}", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "under-dispersion diagnostic for the Phase 6 report",
        "coordinates": {"regimes": list(REGIMES), "n_train": N_TRAIN,
                         "seeds": list(SEEDS), "n_test": N_TEST},
        "records": records,
        "summary": {
            dgp_id: {
                arm: float(np.mean([
                    r["arms"][arm]["bias"] for r in records if r["dgp"] == dgp_id
                ]))
                for arm in ("0", "1")
            }
            for dgp_id in REGIMES
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
