#!/usr/bin/env python3
"""Refit the Kenya primary specification with the external KIHBS benchmark.

The estimator's reference functional uses the dataset's q_star. This script
replaces the pooled-control benchmark with the World Bank PIP reconstruction of
the KIHBS 2015/16 national per-capita consumption distribution (see
build_external_benchmark.py) and fits one seed per process.

Usage:
    PYTHONPATH=src python research/applied/egger_kenya/run_external_benchmark.py --seed 0
    PYTHONPATH=src python research/applied/egger_kenya/run_external_benchmark.py --aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.applied.adapter import (  # noqa: E402
    AppliedDataset,
    aggregate_replications,
    run_wcf,
)

OUT = ROOT / "results" / "applied_study_exploration" / "egger_kenya"
BENCH = OUT / "benchmark_external"
RUNS = OUT / "runs" / "external"
SEEDS = (0, 1, 2)


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        per_seed = []
        for seed in SEEDS:
            with open(RUNS / f"seed{seed}.json") as handle:
                per_seed.append(json.load(handle))
        aggregate = aggregate_replications(per_seed)
        payload = {
            "stage": "external",
            "benchmark": "KIHBS 2015/16 national per-capita consumption distribution",
            "benchmark_file": "benchmark_external/kihbs_2015_national_benchmark.npz",
            "seeds": list(SEEDS),
            "n": per_seed[0]["n"],
            "n_treated": per_seed[0]["n_treated"],
            "n_control": per_seed[0]["n_control"],
            "per_seed": per_seed,
            "aggregate": aggregate,
        }
        RUNS.mkdir(parents=True, exist_ok=True)
        with open(RUNS / "results.json", "w") as handle:
            json.dump(payload, handle, indent=2)
        print(json.dumps(aggregate["marginal_dr"]["reference"], indent=2))
        return

    if args.seed is None:
        parser.error("--seed is required unless --aggregate is used")

    ds = AppliedDataset.load(OUT / "data")
    bench = np.load(BENCH / "kihbs_2015_national_benchmark.npz")
    q_star = np.asarray(bench["q_star_ksh_per_year"], dtype=float)
    ds_external = replace(ds, q_star=q_star).validate()

    started = time.time()
    _, results = run_wcf(ds_external, random_state=args.seed)
    results["elapsed_seconds"] = time.time() - started
    results["stage"] = "external"
    results["external_benchmark_file"] = "benchmark_external/kihbs_2015_national_benchmark.npz"
    results["q_star_sha256"] = sha256(q_star)
    results["primary_q_star_sha256"] = sha256(np.asarray(ds.q_star, dtype=float))
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / f"seed{args.seed}.json", "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(
        f"seed {args.seed}: n={results['n']} elapsed={results['elapsed_seconds']:.0f}s "
        f"reference={results['marginal_dr']['reference']:.1f} "
        f"(SE {results['if_se']['reference']:.1f})"
    )


if __name__ == "__main__":
    main()
