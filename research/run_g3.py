#!/usr/bin/env python3
"""Launcher for the G3 tournament that pins threads before anything imports.

    python research/run_g3.py run --workers 6
    python research/run_g3.py freeze
    python research/run_g3.py merge

Use this rather than `python -m wasserstein_causal_forests.g3.cli`. Running the
CLI as a module executes `wasserstein_causal_forests/__init__.py` first, which
imports NumPy through `.cwdb`, so any environment variable the CLI module body
sets afterwards arrives too late: OpenMP has already sized its pool from the
hardware. That failure is silent and expensive. It left six workers running
thirty-nine threads each on a twenty-CPU machine, driving the load average past
one hundred and cutting throughput to roughly one fortieth of the single-thread
rate measured in the cost pilot.

Nothing above the environment block may import the package, directly or
indirectly. Keep the imports below it.
"""

from __future__ import annotations

import os
import sys

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "R_NUM_THREADS",
):
    os.environ[_variable] = "1"

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wasserstein_causal_forests.g3.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
