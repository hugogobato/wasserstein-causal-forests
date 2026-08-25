"""Python driver for the forced-shared MVBCF baseline (PTA-F).

The sampler itself lives in the MIT-licensed `mvbcf` R package and is reached
through `research/pta_bcf/mvbcf_bridge.R`. This module owns only the fixed
target-matrix interface, the binary array exchange, and the common posterior
draw schema shared with PTA-S.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

PTA_F_METHOD_ID = "PTA-F"
BRIDGE_RELATIVE_PATH = Path("research/pta_bcf/mvbcf_bridge.R")
MVBCF_PINNED_REVISION = "fc3b89b0a78ce8a31ae75c43a6ec75f1945ca0c8"
MVBCF_PACKAGE_LICENSE = "MIT"

ARRAY_NAMES = ("tau_train", "tau_test", "mu_train", "mu_test", "sigma")


class MVBCFBridgeError(RuntimeError):
    """Raised when the R bridge fails or is unavailable."""


@dataclass(frozen=True)
class MVBCFBudget:
    """Sampler budget for the forced-shared endpoint."""

    n_iter: int = 1000
    n_burn: int = 500
    n_tree: int = 50
    n_tree_tau: int = 20
    min_nodesize: int = 1
    sigma_mu_scale: float = 1.0
    sigma_tau_scale: float = 0.375

    def __post_init__(self) -> None:
        if self.n_burn >= self.n_iter:
            raise ValueError("n_burn must be smaller than n_iter")
        if min(self.n_tree, self.n_tree_tau, self.min_nodesize) < 1:
            raise ValueError("tree counts and min_nodesize must be positive")

    @property
    def n_draws(self) -> int:
        return int(self.n_iter - self.n_burn)


@dataclass(frozen=True)
class MVBCFResult:
    """Posterior draws in the schema shared with PTA-S."""

    control_train: NDArray[np.float64]
    contrast_train: NDArray[np.float64]
    control_test: NDArray[np.float64]
    contrast_test: NDArray[np.float64]
    residual_covariance: NDArray[np.float64]
    meta: dict[str, object]

    @property
    def n_draws(self) -> int:
        return int(self.contrast_test.shape[2])

    def contrast_mean(self, which: str = "test") -> NDArray[np.float64]:
        draws = self.contrast_test if which == "test" else self.contrast_train
        return draws.mean(axis=2)

    def control_mean(self, which: str = "test") -> NDArray[np.float64]:
        draws = self.control_test if which == "test" else self.control_train
        return draws.mean(axis=2)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def bridge_path() -> Path:
    return repository_root() / BRIDGE_RELATIVE_PATH


def rscript_executable() -> str | None:
    return shutil.which("Rscript")


def bridge_available() -> bool:
    """True when Rscript, the bridge script, and the pinned package are present."""

    executable = rscript_executable()
    if executable is None or not bridge_path().exists():
        return False
    probe = subprocess.run(
        [executable, "-e", 'quit(status = as.integer(!requireNamespace("mvbcf", quietly = TRUE)))'],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _write_array(array: NDArray[np.float64], path: Path) -> dict[str, object]:
    # R reads a flat double stream in column-major order.
    values = np.asarray(array, dtype=float)
    values.ravel(order="F").tofile(path)
    return {"path": str(path), "shape": list(values.shape)}


def _read_array(path: Path, shape: list[int]) -> NDArray[np.float64]:
    values = np.fromfile(path, dtype=np.float64)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise MVBCFBridgeError(
            f"{path.name} holds {values.size} values, expected {expected}"
        )
    return np.reshape(values, shape, order="F")


class MVBCFForcedShared:
    """Forced-shared multivariate BCF over the whole PTA target matrix."""

    def __init__(
        self,
        *,
        budget: MVBCFBudget = MVBCFBudget(),
        random_state: int = 1,
        rscript: str | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.budget = budget
        self.random_state = int(random_state)
        self.rscript = rscript
        self.timeout_seconds = float(timeout_seconds)

    def fit_predict(
        self,
        X_control: ArrayLike,
        targets: ArrayLike,
        treatment: ArrayLike,
        X_moderator: ArrayLike | None = None,
        *,
        X_control_test: ArrayLike | None = None,
        X_moderator_test: ArrayLike | None = None,
        workspace: str | os.PathLike[str] | None = None,
    ) -> MVBCFResult:
        """Fit on the training rows and return draws for train and test rows."""

        executable = self.rscript or rscript_executable()
        if executable is None:
            raise MVBCFBridgeError("Rscript is not available on PATH")
        script = bridge_path()
        if not script.exists():
            raise MVBCFBridgeError(f"missing bridge script at {script}")

        X_control = np.asarray(X_control, dtype=float)
        Y = np.asarray(targets, dtype=float)
        Z = np.asarray(treatment, dtype=float)
        X_moderator = (
            X_control if X_moderator is None else np.asarray(X_moderator, dtype=float)
        )
        X_control_test = (
            X_control
            if X_control_test is None
            else np.asarray(X_control_test, dtype=float)
        )
        X_moderator_test = (
            X_moderator
            if X_moderator_test is None
            else np.asarray(X_moderator_test, dtype=float)
        )
        if Y.ndim != 2 or Y.shape[0] != X_control.shape[0]:
            raise ValueError("targets must have shape (n, D) matching X_control")
        if Z.shape != (X_control.shape[0],):
            raise ValueError("treatment must have one entry per training row")
        if not np.isin(np.unique(Z), (0.0, 1.0)).all():
            raise ValueError("treatment must be binary in {0, 1}")
        # The package requires equal column counts for train and test designs.
        if X_control_test.shape[1] != X_control.shape[1]:
            raise ValueError("X_control_test must have the same number of columns")
        if X_moderator_test.shape[1] != X_moderator.shape[1]:
            raise ValueError("X_moderator_test must have the same number of columns")

        created = workspace is None
        directory = Path(
            tempfile.mkdtemp(prefix="mvbcf_") if created else workspace
        )
        directory.mkdir(parents=True, exist_ok=True)
        try:
            arrays = {
                "X_con": _write_array(X_control, directory / "X_con.bin"),
                "Y": _write_array(Y, directory / "Y.bin"),
                "Z": _write_array(Z[:, None], directory / "Z.bin"),
                "X_mod": _write_array(X_moderator, directory / "X_mod.bin"),
                "X_con_test": _write_array(
                    X_control_test, directory / "X_con_test.bin"
                ),
                "X_mod_test": _write_array(
                    X_moderator_test, directory / "X_mod_test.bin"
                ),
            }
            spec = {
                "arrays": arrays,
                "params": {**asdict(self.budget), "seed": self.random_state},
                "output_dir": str(directory / "out"),
            }
            spec_path = directory / "spec.json"
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            started = time.perf_counter()
            completed = subprocess.run(
                [executable, str(script), "fit", str(spec_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=repository_root(),
            )
            wall_time = time.perf_counter() - started
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            if completed.returncode != 0:
                raise MVBCFBridgeError(
                    f"mvbcf_bridge.R failed: {completed.stderr.strip()[-2000:]}"
                )

            output_dir = directory / "out"
            meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
            draws = {
                name: _read_array(
                    output_dir / f"{name}.bin", list(meta["shapes"][name])
                )
                for name in ARRAY_NAMES
            }
            meta.update(
                {
                    "method_id": PTA_F_METHOD_ID,
                    "wall_time_seconds": wall_time,
                    "child_peak_ram_mb": max(after, before) / 1024.0,
                    "package_license": MVBCF_PACKAGE_LICENSE,
                }
            )
            return MVBCFResult(
                control_train=draws["mu_train"],
                contrast_train=draws["tau_train"],
                control_test=draws["mu_test"],
                contrast_test=draws["tau_test"],
                residual_covariance=draws["sigma"],
                meta=meta,
            )
        finally:
            if created:
                shutil.rmtree(directory, ignore_errors=True)
