"""Run the pre-registered WCF fits for Project STAR.

    PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python research/applied/project_star/03_run_jobs.py --jobs smoke --workers 1
    ... --jobs primary placebo --workers 2
    ... --jobs min15 reg2 reading qstarall --workers 2

Each fit saves one JSON under results/applied_study_exploration/project_star/.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from star_common import RESULTS_DIR, run_and_save  # noqa: E402

from wasserstein_causal_forests.applied.adapter import AppliedDataset  # noqa: E402

DATA = RESULTS_DIR / "data"

# job -> (dataset directory, output subdir, seeds, placebo, extra run kwargs)
JOBS: dict[str, dict] = {
    "smoke": {
        "data": DATA,
        "out": "misc",
        "seeds": (99,),
        "placebo": False,
        "kwargs": {"n_estimators": 30, "n_folds": 2, "n_particles": 5},
    },
    "primary": {
        "data": DATA,
        "out": "primary",
        "seeds": (0, 1, 2),
        "placebo": False,
        "kwargs": {},
    },
    "placebo": {
        "data": DATA,
        "out": "placebo",
        "seeds": (0, 1, 2),
        "placebo": True,
        "kwargs": {},
    },
    "min15": {
        "data": DATA / "math_min15",
        "out": "sensitivity",
        "seeds": (0,),
        "placebo": False,
        "kwargs": {},
    },
    "reg2": {
        "data": DATA / "math_reg2",
        "out": "sensitivity",
        "seeds": (0,),
        "placebo": False,
        "kwargs": {},
    },
    "reading": {
        "data": DATA / "reading",
        "out": "sensitivity",
        "seeds": (0,),
        "placebo": False,
        "kwargs": {},
    },
    "qstarall": {
        "data": DATA / "math_qstarall",
        "out": "sensitivity",
        "seeds": (0,),
        "placebo": False,
        "kwargs": {},
    },
    "schoolsat": {
        "data": DATA / "math_schoolsat",
        "out": "sensitivity",
        "seeds": (0,),
        "placebo": False,
        "kwargs": {},
    },
}


def run_job(job: str, only_seeds: tuple[int, ...] | None = None) -> list[str]:
    spec = JOBS[job]
    ds = AppliedDataset.load(spec["data"])
    out_dir = RESULTS_DIR / spec["out"]
    written = []
    seeds = spec["seeds"] if only_seeds is None else tuple(
        s for s in spec["seeds"] if s in only_seeds
    )
    for seed in seeds:
        override = None
        tag = job
        if spec["placebo"]:
            rng = np.random.default_rng(10_000 + seed)
            override = rng.permutation(ds.A)
            tag = f"{job}_permuted"
        path = out_dir / f"{job}_seed{seed}.json"
        if path.exists():
            print("skip existing", path, flush=True)
            written.append(str(path))
            continue
        run_and_save(
            ds,
            path,
            random_state=seed,
            treatment_override=override,
            tag=tag,
            **spec["kwargs"],
        )
        written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True, help="job names or 'all'")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--only-seeds",
        type=int,
        nargs="+",
        default=None,
        help="restrict to these seed values (inspection/measurement only)",
    )
    args = parser.parse_args()

    jobs = list(JOBS) if args.jobs == ["all"] else args.jobs
    only = tuple(args.only_seeds) if args.only_seeds is not None else None
    for job in jobs:
        if job not in JOBS:
            raise SystemExit(f"unknown job {job}; known: {sorted(JOBS)}")
    if args.workers <= 1:
        for job in jobs:
            print("done", job, run_job(job, only), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, only): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                print("done", job, future.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print("FAILED", job, repr(exc), flush=True)
                raise
    print("all jobs complete")


if __name__ == "__main__":
    main()
