#!/usr/bin/env python3
"""Run the preregistered Phase 6.5 bandwidth-selection pilot locally.

    python research/checks/phase65_bandwidth_pilot.py [--workers 4]

For every (IC regime, n_train) cell that contains `causal_drf_retn` cells, run
the frozen selection rule of `research/simulation_preregistration_phase65.md`:
pilot seeds 100 and 101 (outside every decisive range), candidates
{0.25, 0.5, 1, 2, 4} entered as explicit kernel-bandwidth multipliers
through `g3_causal_drf_retn_driver.R`, scored by held-out energy risk against the realised outcome
distributions of the scoring half. The argmin per cell is frozen into
`results/manifests/phase65_bandwidth_selection.json`; the full score table is
kept alongside for the record.

The pilot is idempotent per combo and writes the document only after every
combo has finished.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "R_NUM_THREADS",
):
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# The Causal-DRF driver requires the authors' causal-clean package from the
# project-local library; without this variable the R subprocess loads CRAN
# drf, which rejects the treatment argument. Same contract as Track B.
_CAUSAL_LIB = ROOT / "results" / "Rlib" / "causal_drf"
if not (_CAUSAL_LIB / "drf").is_dir():
    raise SystemExit(
        f"{_CAUSAL_LIB} does not contain a built causal-clean drf; build it "
        "once (see research/baselines/PROVENANCE.md) or run the forest "
        "shards on Colab."
    )
os.environ["WCF_CAUSAL_DRF_R_LIB"] = str(_CAUSAL_LIB)

# Registration side effects: the IC regimes and the Phase 6.5 variants only
# exist once these modules have run, exactly as in the runner.
from wasserstein_causal_forests.g3 import phase65 as _phase65  # noqa: E402,F401
from wasserstein_causal_forests.g3 import phase6_dgps as _p6d  # noqa: E402,F401
from wasserstein_causal_forests.g3 import phase65_dgps as _p65d  # noqa: E402,F401

_p6d.register_phase6_dgps()
_p65d.register_phase65_dgps()

from wasserstein_causal_forests.g3.dgps import build_dgp  # noqa: E402
from wasserstein_causal_forests.g3.phase65_methods import (  # noqa: E402
    BANDWIDTH_CANDIDATES,
    SELECTION_SEEDS,
    select_bandwidth_multiplier,
)

OUTPUT_PATH = ROOT / "results" / "manifests" / "phase65_bandwidth_selection.json"
CACHE_DIRECTORY = ROOT / "results" / "rcpp_cache"

#: The decisive retune cells live on the income track at both sample sizes.
PILOT_COMBOS: tuple[tuple[str, int], ...] = tuple(
    (dgp_name, n_train)
    for dgp_name in ("IC0", "IC1", "IC2", "IC3")
    for n_train in (500, 1000)
)


def _run_combo(payload: tuple[str, int]) -> tuple[str, float, dict[str, float]]:
    dgp_name, n_train = payload
    pin = os.environ.copy()
    started = time.perf_counter()
    best, means = select_bandwidth_multiplier(
        build_dgp(dgp_name, 25),
        n_train,
        seeds=SELECTION_SEEDS,
        candidates=BANDWIDTH_CANDIDATES,
        cache_directory=CACHE_DIRECTORY,
    )
    wall = time.perf_counter() - started
    key = f"{dgp_name}|{n_train}"
    print(
        f"{key}: multiplier {best}  ({wall / 60:.1f} min)  "
        + " ".join(f"{c}:{v:.5f}" for c, v in sorted(means.items())),
        flush=True,
    )
    _ = pin
    return key, best, {str(c): v for c, v in means.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()

    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    multipliers: dict[str, float] = {}
    scores: dict[str, dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for key, best, means in pool.map(_run_combo, PILOT_COMBOS):
            multipliers[key] = best
            scores[key] = means

    document = {
        "rule": (
            "held-out energy score against realised outcomes; pilot seeds "
            f"{list(SELECTION_SEEDS)}; candidate grid "
            f"{list(BANDWIDTH_CANDIDATES)} entered as explicit kernel-bandwidth "
            f"multipliers"
        ),
        "multipliers": multipliers,
        "scores": scores,
    }
    OUTPUT_PATH.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"froze {OUTPUT_PATH}")
    print(f"  multipliers: {multipliers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
