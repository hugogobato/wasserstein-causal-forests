"""Subprocess bridge to the two R forest baselines.

Arrays cross the boundary as flat column-major doubles, the same convention
`wasserstein_causal_forests.pta_bcf.mvbcf` already uses for the MVBCF bridge.
What comes back is the pair of arm weight matrices over the shared training
bank, which is exactly the input `LawPrediction.from_forest_weights` wants: the
baselines and C-WDB then flow through one metric implementation rather than two.

The Causal-DRF driver uses the authors' causal-clean `drf` package through the
original simulation code path. The paper-DRF driver uses the supplied
`drfinference` half-sample implementation. The legacy `warm_rcpp_cache` helper
remains available for historical adapters but is not needed by the current
runner.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DRIVER_RELATIVE_PATH = "research/baselines/g3_driver.R"
ORIGINAL_CAUSAL_DRF_DRIVER_RELATIVE_PATH = (
    "research/baselines/g3_causal_drf_original_driver.R"
)
RETN_CAUSAL_DRF_DRIVER_RELATIVE_PATH = (
    "research/baselines/g3_causal_drf_retn_driver.R"
)
ORIGINAL_DRF_DRIVER_RELATIVE_PATH = "research/baselines/g3_drf_original_driver.R"
RCPP_CACHE_VARIABLE = "WCF_RCPP_CACHE"


class RBridgeError(RuntimeError):
    """The R driver failed, or returned something the schema does not allow."""


@dataclass(frozen=True)
class ForestBaselineResult:
    """Arm weight matrices over the training bank, plus driver telemetry."""

    weights: dict[int, NDArray[np.float64]]
    fit_seconds: float
    total_seconds: float
    peak_ram_mb: float


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def driver_path() -> Path:
    return repository_root() / DRIVER_RELATIVE_PATH


def rscript_executable() -> str | None:
    return shutil.which("Rscript")


def bridge_available() -> bool:
    """True when Rscript, the driver, and the pinned `drf` package are present."""

    executable = rscript_executable()
    if executable is None or not driver_path().exists():
        return False
    probe = subprocess.run(
        [
            executable,
            "-e",
            'quit(status = as.integer(!requireNamespace("drf", quietly = TRUE)))',
        ],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _write_array(array: NDArray[np.float64], path: Path) -> None:
    np.asarray(array, dtype=float).ravel(order="F").tofile(path)


def _read_array(path: Path, shape: tuple[int, ...]) -> NDArray[np.float64]:
    values = np.fromfile(path, dtype=np.float64)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise RBridgeError(
            f"{path.name} holds {values.size} values, expected {expected}"
        )
    return np.reshape(values, shape, order="F")


def _environment(cache_directory: Path | None) -> dict[str, str]:
    environment = dict(os.environ)
    # One thread per worker: the tournament parallelises over cells, and BLAS
    # oversubscription inside a worker would fight the process pool for cores.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "R_NUM_THREADS",
    ):
        environment[variable] = "1"
    if cache_directory is not None:
        environment[RCPP_CACHE_VARIABLE] = str(cache_directory)
    return environment


def warm_rcpp_cache(cache_directory: Path, *, timeout_seconds: float = 1800.0) -> None:
    """Compile the Causal-DRF translation unit once, before any fan-out."""

    executable = rscript_executable()
    if executable is None:
        raise RBridgeError("Rscript is not available on PATH")
    cache_directory.mkdir(parents=True, exist_ok=True)
    source = repository_root() / "research/baselines/causal_drf_r/causal_drf.R"
    completed = subprocess.run(
        [executable, "-e", f'source("{source.as_posix()}", chdir = TRUE)'],
        capture_output=True,
        text=True,
        env=_environment(cache_directory),
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RBridgeError(
            "warming the Rcpp cache failed:\n" + completed.stderr[-4000:]
        )


def fit_predict(
    method: str,
    *,
    X_train: NDArray[np.float64],
    treatment: NDArray[np.int64],
    Q_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    reference_quantiles: NDArray[np.float64],
    functionals: tuple[str, ...] = (),
    hyperparameters: dict[str, object] | None = None,
    seed: int = 1,
    cache_directory: Path | None = None,
    timeout_seconds: float = 3600.0,
) -> ForestBaselineResult:
    """Fit one R baseline and return its per-arm weights over the bank.

    ``causal_drf_retn`` is the Phase 6.5 retune control: the authors' fit call
    with an explicit kernel bandwidth of
    ``spec["bandwidth_multiplier"]`` times the data-driven default. The
    multiplier must be supplied through ``hyperparameters``; the driver fails
    loudly without it.
    """

    if method not in {"wdrft", "causal_drf", "drf", "causal_drf_retn"}:
        raise ValueError(
            "method must be 'wdrft', 'causal_drf', 'drf', or 'causal_drf_retn'"
        )
    executable = rscript_executable()
    if executable is None:
        raise RBridgeError("Rscript is not available on PATH")

    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]
    n_grid = Q_train.shape[1]
    spec: dict[str, object] = {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_features": int(n_features),
        "n_grid": int(n_grid),
        "seed": int(seed),
        "functionals": list(functionals),
        "num_trees": 1000,
        "min_node_size": 15,
        "min_arm_leaf": 5,
    }
    if method in {"causal_drf", "causal_drf_retn"}:
        # These are the authors' simulation settings in
        # code/causal_drf_paper-main/R/simulation_study.R.
        spec.update({"num_trees": 2500, "ci_group_size": 50})
    elif method == "drf":
        # Exact separate-forest settings from the Causal-DRF paper's DRF
        # benchmark: 50 half-sample groups, 50 trees per group.
        spec.update({"num_trees": 2500, "ci_group_size": 50})
    spec.update(hyperparameters or {})

    directory = Path(tempfile.mkdtemp(prefix=f"g3_{method}_"))
    try:
        _write_array(X_train, directory / "X_train.bin")
        _write_array(np.asarray(treatment, dtype=float)[:, None],
                     directory / "A_train.bin")
        _write_array(Q_train, directory / "Q_train.bin")
        _write_array(X_test, directory / "X_test.bin")
        _write_array(np.asarray(quad_weights, dtype=float)[:, None],
                     directory / "quad_weights.bin")
        _write_array(np.asarray(reference_quantiles, dtype=float)[:, None],
                     directory / "reference.bin")
        (directory / "spec.json").write_text(json.dumps(spec), encoding="utf-8")

        if method == "causal_drf":
            selected_driver = (
                repository_root() / ORIGINAL_CAUSAL_DRF_DRIVER_RELATIVE_PATH
            )
            command = [executable, str(selected_driver), str(directory)]
        elif method == "causal_drf_retn":
            selected_driver = (
                repository_root() / RETN_CAUSAL_DRF_DRIVER_RELATIVE_PATH
            )
            command = [executable, str(selected_driver), str(directory)]
        else:
            selected_driver = repository_root() / (
                ORIGINAL_DRF_DRIVER_RELATIVE_PATH if method == "drf"
                else DRIVER_RELATIVE_PATH
            )
            command = [executable, str(selected_driver), str(directory), method]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=_environment(cache_directory),
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RBridgeError(
                f"{method} driver exited {completed.returncode}:\n"
                + completed.stderr[-4000:]
            )
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        weights = {
            0: _read_array(directory / "weights_control.bin", (n_test, n_train)),
            1: _read_array(directory / "weights_treated.bin", (n_test, n_train)),
        }
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    for arm, matrix in weights.items():
        total = matrix.sum(axis=1)
        if not np.allclose(total, 1.0, atol=1e-6, rtol=0.0):
            raise RBridgeError(
                f"{method} arm {arm} weights do not sum to one "
                f"(worst row deviates by {np.max(np.abs(total - 1.0)):.2e})"
            )
    return ForestBaselineResult(
        weights=weights,
        fit_seconds=float(result["fit_seconds"]),
        total_seconds=float(result["total_seconds"]),
        peak_ram_mb=float(result.get("peak_ram_mb", 0.0)),
    )
