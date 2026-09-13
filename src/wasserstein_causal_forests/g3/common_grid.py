"""Score fitted methods against one dense oracle truth on a common grid.

Native-grid metrics change their estimand when K changes: an RMSE against a
K-level truth is not comparable across K. This adapter scores a fitted method
against a single 199-level truth built directly from the DGP specification, so
every method is compared on the same target whatever grid it was fitted on.
Predicted particles, forest atoms, and barycenters are interpolated linearly in
quantile level, with constant endpoint continuation outside the fitted grid's
range, and interpolation never extrapolates beyond the fitted range.

Two level sets are emitted under target identifiers that cannot collide with
any native-grid identifier:

* ``COMMON199``: the 199 midpoint levels ``(k + 0.5) / 199``;
* ``INTERIOR``: the same levels remapped to ``0.1 + 0.8 u``, so every proposed
  coarse K grid lies inside the diagnostic range and no fitted atom is
  extrapolated when that diagnostic is computed.

Truth is always quadratured on the dense level set itself; a coarse oracle is
never interpolated. Native functional columns are never relabelled as dense
targets: a dense functional is either integrated against an interpolated law or
read from a declared dense-grid calibration payload, and a row that cannot be
computed is emitted with ``status = "not_applicable"`` and a stated reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..pta_bcf.targets import GRID_FUNCTIONALS
from .dgps import (
    N_MODERATOR_BINS,
    DistributionalDGP,
    build_dgp,
    moderator_bins,
    resolve_dgp_spec,
)
from .evaluation import (
    NO_LAW_REASON,
    _bin_means,
    _oracle_energy_risk,
    _restrict,
    _rmse,
    _row,
    _truth_nodes,
)
from .laws import LawPrediction, kernel_law_error, median_heuristic_bandwidth

if TYPE_CHECKING:
    from .manifest import Cell
    from .methods import MethodOutput

#: Dense midpoint levels every method is scored on.
COMMON_LEVELS: NDArray[np.float64] = (np.arange(199) + 0.5) / 199.0

#: The same resolution compressed into (0.1, 0.9), where no proposed coarse
#: grid needs interpolation beyond its own support.
INTERIOR_LEVELS: NDArray[np.float64] = 0.1 + 0.8 * COMMON_LEVELS

#: Collision scale used by the dense oracle-energy floor. Mirrors the frozen
#: evaluation manifest so the cached truth stays comparable with native rows.
COLLISION_EPSILON = 1e-3

#: The Phase 6 income regimes use a benchmark-economy reference rather than the
#: standard normal. The constants duplicate `phase6_dgps` so this adapter
#: depends only on the modules it may import.
_INCOME_DGPS = ("IC0", "IC1", "IC2", "IC3")
_REFERENCE_LOCATION = 1.10
_REFERENCE_LOG_SCALE = 0.60
_REFERENCE_SHAPE = 0.30

_LEVEL_SETS = (("COMMON199", COMMON_LEVELS), ("INTERIOR", INTERIOR_LEVELS))

_DR_DETAIL = "dense-grid DR calibration"

_NO_OUTPUT_REASON = "cell produced no method output"

_NO_DENSE_LAW_REASON = (
    "method supplies no conditional law and no dense-grid calibration payload, "
    "so the dense target cannot be integrated; native-grid values are never "
    "relabelled as dense-grid targets"
)


class LevelGridSpec:
    """A supplied level set with uniform weights and a regime reference.

    The interface matches :class:`~wasserstein_causal_forests.g3.dgps.GridSpec`
    where the oracle machinery touches it: ``n_grid``, ``levels``, ``base_z``,
    ``weights``, and ``reference_quantiles``. The reference follows the regime
    family: the Phase 6 income tracks ``IC0`` through ``IC3`` use the frozen
    benchmark-economy formula

        location + log_scale * (z + shape * ((z^2 - 1) / 2 + (z^3 - 3 z) / 6)),

    with ``z`` the standard normal quantile at each level, and every other
    regime references the standard normal itself.
    """

    def __init__(self, levels: NDArray[np.float64], dgp_id: str) -> None:
        levels = np.asarray(levels, dtype=float)
        if levels.ndim != 1 or levels.size < 2:
            raise ValueError("levels must be one-dimensional with at least two entries")
        if np.any(np.diff(levels) <= 0.0):
            raise ValueError("levels must be strictly increasing")
        if levels[0] <= 0.0 or levels[-1] >= 1.0:
            raise ValueError("levels must lie strictly inside (0, 1)")
        self._levels = levels
        self.dgp_id = str(dgp_id)

    @property
    def n_grid(self) -> int:
        return int(self._levels.size)

    @property
    def levels(self) -> NDArray[np.float64]:
        return self._levels

    @property
    def base_z(self) -> NDArray[np.float64]:
        return norm.ppf(self._levels)

    @property
    def weights(self) -> NDArray[np.float64]:
        return np.full(self.n_grid, 1.0 / self.n_grid)

    @property
    def max_abs_z(self) -> float:
        return float(np.max(np.abs(self.base_z)))

    def reference_quantiles(self) -> NDArray[np.float64]:
        """The regime's external reference law, evaluated at the dense levels."""

        z = self.base_z
        if self.dgp_id not in _INCOME_DGPS:
            return z
        hermite = (z * z - 1.0) / 2.0 + (z * z * z - 3.0 * z) / 6.0
        return _REFERENCE_LOCATION + _REFERENCE_LOG_SCALE * (
            z + _REFERENCE_SHAPE * hermite
        )


def interpolate_quantile_curves(
    values: NDArray[np.float64],
    source_levels: NDArray[np.float64],
    target_levels: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate quantile vectors linearly along the last axis.

    ``values`` may have any leading shape, e.g. ``(K,)``, ``(m, K)``, or
    ``(A, K)``; the result has the same leading shape with the last axis
    replaced by ``target_levels``. Target levels outside the source range are
    continued constantly from the nearest endpoint, so no value is ever
    extrapolated beyond the fitted grid.
    """

    values = np.asarray(values, dtype=float)
    source = np.asarray(source_levels, dtype=float)
    target = np.asarray(target_levels, dtype=float)
    if source.ndim != 1 or source.size < 2:
        raise ValueError(
            "source_levels must be one-dimensional with at least two entries"
        )
    if np.any(np.diff(source) <= 0.0):
        raise ValueError("source_levels must be strictly increasing")
    if values.ndim == 0 or values.shape[-1] != source.size:
        raise ValueError("the last axis of values must match source_levels")
    if target.ndim != 1:
        raise ValueError("target_levels must be one-dimensional")

    upper = np.clip(
        np.searchsorted(source, target, side="right") - 1, 0, source.size - 2
    )
    width = source[upper + 1] - source[upper]
    fraction = np.clip((target - source[upper]) / width, 0.0, 1.0)
    lower = values[..., upper]
    above = values[..., upper + 1]
    return lower + fraction * (above - lower)


def subsample_law(
    law: LawPrediction, target_levels: NDArray[np.float64]
) -> LawPrediction:
    """Interpolate a law's atoms onto ``target_levels`` along the grid axis.

    The source levels are the law's own midpoint grid ``(k + 0.5) / K``:
    :class:`LawPrediction` stores grid vectors without their level labels, and
    every producing adapter declares the uniform midpoint grid. Weights and the
    shared-atom flag are carried over unchanged.
    """

    source_levels = (np.arange(law.n_grid, dtype=float) + 0.5) / law.n_grid
    atoms = interpolate_quantile_curves(law.atoms, source_levels, target_levels)
    return LawPrediction(
        atoms=atoms, weights=law.weights, shared_atoms=law.shared_atoms
    )


def build_dgp_at_levels(
    dgp_id: str, levels: NDArray[np.float64]
) -> DistributionalDGP:
    """Build a dense DGP directly on supplied levels from the regime spec.

    A registered specification gives
    ``DistributionalDGP(resolve_dgp_spec(dgp_id), LevelGridSpec(levels, dgp_id))``,
    which reproduces the income reference for ``IC0`` through ``IC3`` and the
    standard normal reference elsewhere. An identifier available only through
    ``_EXTRA_BUILDERS`` has no standalone spec, so it falls back to
    ``build_dgp(dgp_id, 199)`` and only when the requested levels are exactly
    ``COMMON_LEVELS``, whose midpoint grid the builder reproduces.
    """

    levels = np.asarray(levels, dtype=float)
    try:
        spec = resolve_dgp_spec(dgp_id)
    except ValueError:
        spec = None
    if spec is not None:
        return DistributionalDGP(spec, LevelGridSpec(levels, dgp_id))
    if not np.array_equal(levels, COMMON_LEVELS):
        raise ValueError(
            f"{dgp_id!r} has no registered specification; the builder fallback "
            "can serve only COMMON_LEVELS"
        )
    dense = build_dgp(dgp_id, COMMON_LEVELS.size)
    if not np.array_equal(np.asarray(dense.grid.levels, dtype=float), COMMON_LEVELS):
        raise ValueError(
            f"the builder for {dgp_id!r} does not reproduce COMMON_LEVELS"
        )
    return dense


def evaluate_common_grid(
    cell: Cell,
    output: MethodOutput | None,
    dgp: DistributionalDGP,
    X_test: NDArray[np.float64],
    *,
    n_law_rows: int = 200,
) -> list[dict[str, object]]:
    """Return the dense-grid rows for one fitted method.

    Every declared dense target is emitted for both level sets. A row is
    computed from the dense oracle and either the interpolated predicted law or,
    when the output carries an ``output.common_grid`` payload for that level
    set, from its calibrated marginals and moderator-bin contrasts. Rows that
    cannot be computed carry ``status = "not_applicable"`` and a reason.
    """

    X_test = np.asarray(X_test, dtype=float)
    if output is None:
        return [
            _row(
                metric,
                target,
                None,
                status="not_applicable",
                failure_reason=_NO_OUTPUT_REASON,
            )
            for metric, target in _DECLARED_TARGETS
        ]

    rows: list[dict[str, object]] = []
    bins = moderator_bins(X_test)
    native_levels = np.asarray(dgp.grid.levels, dtype=float)
    payload = getattr(output, "common_grid", None)

    for level_set_name, levels in _LEVEL_SETS:
        try:
            dense = build_dgp_at_levels(cell.dgp, levels)
        except ValueError as error:
            reason = f"dense truth unavailable: {error}"
            rows.extend(
                _row(
                    metric,
                    target,
                    None,
                    status="not_applicable",
                    failure_reason=reason,
                )
                for metric, target in _level_set_targets(level_set_name)
            )
            continue
        entry = _select_payload_entry(payload, level_set_name, levels)
        rows.extend(
            _mean_quantile_rows(
                output, native_levels, levels, dense, X_test, level_set_name
            )
        )
        rows.extend(
            _dense_law_rows(
                output, cell, dense, levels, X_test, level_set_name, n_law_rows
            )
        )
        if level_set_name == "COMMON199":
            rows.extend(
                _functional_rows(
                    output, dense, levels, X_test, bins, entry, level_set_name
                )
            )
        rows.extend(
            _reference_rows(
                output, dense, levels, X_test, bins, entry, level_set_name
            )
        )
    return rows


_DENSE_TRUTH_CACHE: dict[tuple, tuple] = {}
#: Two arms across two level sets stay resident, so successive methods on one
#: replication reuse the most expensive part of the evaluation.
_DENSE_TRUTH_CACHE_LIMIT = 4


def _dense_truth(
    cell: Cell,
    dense: DistributionalDGP,
    level_set_name: str,
    arm: int,
    X_test: NDArray[np.float64],
    n_law_rows: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Cached dense truth for one arm: nodes, weights, oracle risk, bandwidth.

    The cache is keyed by the test design, regime, level set, arm, and test row
    count, matching the native truth cache's granularity. Law-level rows are
    restricted to ``n_law_rows`` exactly as ``evaluation._law_rows`` does, so
    the cached nodes cover that subset.
    """

    key = (cell.test_seed, cell.dgp, level_set_name, arm, X_test.shape[0])
    cached = _DENSE_TRUTH_CACHE.get(key)
    if cached is not None:
        return cached
    n_rows = min(int(n_law_rows), X_test.shape[0])
    nodes, node_weights = _truth_nodes(dense, X_test[:n_rows], arm)
    oracle = _oracle_energy_risk(
        nodes, node_weights, dense.grid.weights, COLLISION_EPSILON
    )
    bandwidth = median_heuristic_bandwidth(
        nodes[: min(32, n_rows)], dense.grid.weights
    )
    entry = (nodes, node_weights, oracle, bandwidth)
    if len(_DENSE_TRUTH_CACHE) >= _DENSE_TRUTH_CACHE_LIMIT:
        _DENSE_TRUTH_CACHE.pop(next(iter(_DENSE_TRUTH_CACHE)))
    _DENSE_TRUTH_CACHE[key] = entry
    return entry


def _mean_quantile_rows(
    output: MethodOutput,
    native_levels: NDArray[np.float64],
    levels: NDArray[np.float64],
    dense: DistributionalDGP,
    X_test: NDArray[np.float64],
    suffix: str,
) -> list[dict[str, object]]:
    target = f"MEANQ-A-{suffix}"
    try:
        arm_zero = np.asarray(output.mean_quantiles[0], dtype=float)
        arm_one = np.asarray(output.mean_quantiles[1], dtype=float)
    except (AttributeError, KeyError, TypeError):
        return [
            _row(
                "mean_quantile_rmse",
                target,
                None,
                status="not_applicable",
                failure_reason="method output carries no mean quantile curves",
            )
        ]
    estimate = interpolate_quantile_curves(
        arm_one, native_levels, levels
    ) - interpolate_quantile_curves(arm_zero, native_levels, levels)
    truth = dense.mean_quantile_contrast(X_test)
    return [
        _row(
            "mean_quantile_rmse",
            target,
            _rmse(estimate, truth),
            detail="arm contrast against the dense truth",
        )
    ]


def _dense_law_rows(
    output: MethodOutput,
    cell: Cell,
    dense: DistributionalDGP,
    levels: NDArray[np.float64],
    X_test: NDArray[np.float64],
    suffix: str,
    n_law_rows: int,
) -> list[dict[str, object]]:
    target = f"LAW-A-{suffix}"
    laws = getattr(output, "law", None)
    if laws is None or not _has_both_arms(laws):
        reason = (
            NO_LAW_REASON
            if laws is None
            else "method output is missing an arm's conditional law"
        )
        return [
            _row(
                "kernel_law_error",
                target,
                None,
                status="not_applicable",
                failure_reason=reason,
            )
        ]
    n_rows = min(int(n_law_rows), X_test.shape[0])
    subset = slice(0, n_rows)
    rows: list[dict[str, object]] = []
    for arm in (0, 1):
        nodes, node_weights, _, bandwidth = _dense_truth(
            cell, dense, suffix, arm, X_test, n_law_rows
        )
        law = subsample_law(_restrict(laws[arm], subset), levels)
        error = kernel_law_error(
            law, nodes, node_weights, dense.grid.weights, bandwidth=bandwidth
        )
        rows.append(
            _row(
                "kernel_law_error",
                target,
                float(np.mean(error)),
                arm=arm,
                detail="squared MMD, Gaussian kernel, dense median bandwidth",
            )
        )
    return rows


def _functional_rows(
    output: MethodOutput,
    dense: DistributionalDGP,
    levels: NDArray[np.float64],
    X_test: NDArray[np.float64],
    bins: NDArray[np.int64],
    entry: dict | None,
    suffix: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    weight = dense.grid.weights
    laws = getattr(output, "law", None)
    for name, functional in GRID_FUNCTIONALS.items():
        truth = dense.functional_contrast(X_test, name)
        truth_mean = float(np.mean(truth))
        truth_bins = _bin_means(truth, bins)
        law_contrast = None
        if laws is not None and _has_both_arms(laws):
            law_contrast = _law_functional_contrast(laws, levels, weight, functional)

        marginal = _payload_scalar(entry, name)
        tate_target = f"TATE-{suffix}-{name}"
        if marginal is not None:
            rows.append(
                _row(
                    "tate_functional_rmse",
                    tate_target,
                    abs(marginal - truth_mean),
                    detail=_DR_DETAIL,
                )
            )
        elif law_contrast is not None:
            rows.append(
                _row(
                    "tate_functional_rmse",
                    tate_target,
                    abs(float(np.mean(law_contrast)) - truth_mean),
                    detail="arm contrast integrated against the dense law",
                )
            )
        else:
            rows.append(
                _row(
                    "tate_functional_rmse",
                    tate_target,
                    None,
                    status="not_applicable",
                    failure_reason=_NO_DENSE_LAW_REASON,
                )
            )

        contrasts = _payload_vector(entry, name, N_MODERATOR_BINS)
        tcate_target = f"TCATE-{suffix}-{name}"
        value: float | None = None
        detail = ""
        if contrasts is not None:
            value = _bin_rmse(contrasts, truth_bins)
            detail = _DR_DETAIL
        elif law_contrast is not None:
            value = _bin_rmse(_bin_means(law_contrast, bins), truth_bins)
            detail = "across moderator bins, integrated against the dense law"
        if value is None:
            rows.append(
                _row(
                    "tcate_functional_rmse",
                    tcate_target,
                    None,
                    status="not_applicable",
                    failure_reason=_NO_DENSE_LAW_REASON,
                )
            )
        else:
            rows.append(
                _row("tcate_functional_rmse", tcate_target, value, detail=detail)
            )
    return rows


def _reference_rows(
    output: MethodOutput,
    dense: DistributionalDGP,
    levels: NDArray[np.float64],
    X_test: NDArray[np.float64],
    bins: NDArray[np.int64],
    entry: dict | None,
    suffix: str,
) -> list[dict[str, object]]:
    reference = dense.grid.reference_quantiles()
    weight = dense.grid.weights
    truth = dense.reference_contrast(X_test)
    truth_mean = float(np.mean(truth))
    truth_bins = _bin_means(truth, bins)
    laws = getattr(output, "law", None)
    law_contrast = None
    if laws is not None and _has_both_arms(laws):
        law_contrast = _law_reference_contrast(laws, levels, weight, reference)

    rows: list[dict[str, object]] = []
    if suffix == "COMMON199":
        marginal = _payload_scalar(entry, "reference")
        target = f"REF-ATE-{suffix}"
        if marginal is not None:
            rows.append(
                _row(
                    "reference_effect_rmse",
                    target,
                    abs(marginal - truth_mean),
                    detail=_DR_DETAIL,
                )
            )
        elif law_contrast is not None:
            rows.append(
                _row(
                    "reference_effect_rmse",
                    target,
                    abs(float(np.mean(law_contrast)) - truth_mean),
                    detail="integrated against the dense reference and law",
                )
            )
        else:
            rows.append(
                _row(
                    "reference_effect_rmse",
                    target,
                    None,
                    status="not_applicable",
                    failure_reason=_NO_DENSE_LAW_REASON,
                )
            )

    contrasts = _payload_vector(entry, "reference", N_MODERATOR_BINS)
    target = f"REF-TCATE-{suffix}"
    value: float | None = None
    detail = ""
    if contrasts is not None:
        value = _bin_rmse(contrasts, truth_bins)
        detail = _DR_DETAIL
    elif law_contrast is not None:
        value = _bin_rmse(_bin_means(law_contrast, bins), truth_bins)
        detail = "across moderator bins, integrated against the dense reference"
    if value is None:
        rows.append(
            _row(
                "reference_tcate_rmse",
                target,
                None,
                status="not_applicable",
                failure_reason=_NO_DENSE_LAW_REASON,
            )
        )
    else:
        rows.append(_row("reference_tcate_rmse", target, value, detail=detail))
    return rows


def _law_functional_contrast(
    laws: dict[int, LawPrediction],
    levels: NDArray[np.float64],
    weight: NDArray[np.float64],
    functional,
) -> NDArray[np.float64]:
    estimates = []
    for arm in (0, 1):
        law = subsample_law(laws[arm], levels)
        estimates.append(
            law.scalar_expectation(lambda block, h=functional: h(block, weight))
        )
    return estimates[1] - estimates[0]


def _reference_distance(
    block: NDArray[np.float64],
    reference: NDArray[np.float64],
    weight: NDArray[np.float64],
) -> NDArray[np.float64]:
    difference = block - reference
    return np.sqrt(np.sum(weight * difference * difference, axis=-1))


def _law_reference_contrast(
    laws: dict[int, LawPrediction],
    levels: NDArray[np.float64],
    weight: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> NDArray[np.float64]:
    estimates = []
    for arm in (0, 1):
        law = subsample_law(laws[arm], levels)
        estimates.append(
            law.scalar_expectation(
                lambda block: _reference_distance(block, reference, weight)
            )
        )
    return estimates[1] - estimates[0]


def _has_both_arms(laws) -> bool:
    try:
        return 0 in laws and 1 in laws
    except TypeError:
        return False


def _select_payload_entry(
    payload,
    level_set_name: str,
    levels: NDArray[np.float64],
) -> dict | None:
    """The DR payload block for a level set, or None when it does not apply.

    The block is used only when its declared levels match the level set the
    adapter is scoring; a mismatched or absent block falls back to the
    interpolated law rather than comparing estimates on different targets.
    """

    if not isinstance(payload, dict):
        return None
    entry = payload.get(level_set_name)
    if not isinstance(entry, dict):
        return None
    stored = entry.get("levels")
    if stored is not None:
        stored = np.asarray(stored, dtype=float)
        if stored.shape != levels.shape or not np.allclose(stored, levels):
            return None
    return entry


def _payload_scalar(entry: dict | None, name: str) -> float | None:
    if not isinstance(entry, dict):
        return None
    marginal = entry.get("marginal")
    if not isinstance(marginal, dict) or name not in marginal:
        return None
    try:
        value = float(marginal[name])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _payload_vector(
    entry: dict | None, name: str, size: int
) -> NDArray[np.float64] | None:
    if not isinstance(entry, dict):
        return None
    contrasts = entry.get("bin_contrasts")
    if not isinstance(contrasts, dict) or name not in contrasts:
        return None
    block = np.asarray(contrasts[name], dtype=float)
    if block.ndim != 1 or block.size != size or not np.any(np.isfinite(block)):
        return None
    return block


def _bin_rmse(
    estimate: NDArray[np.float64], truth: NDArray[np.float64]
) -> float | None:
    """Bin-level RMSE over the bins the payload marks as nonempty."""

    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    finite = np.isfinite(estimate) & np.isfinite(truth)
    if not np.any(finite):
        return None
    return _rmse(estimate[finite], truth[finite])


def _level_set_targets(suffix: str) -> list[tuple[str, str]]:
    targets = [
        ("mean_quantile_rmse", f"MEANQ-A-{suffix}"),
        ("kernel_law_error", f"LAW-A-{suffix}"),
    ]
    if suffix == "COMMON199":
        for name in GRID_FUNCTIONALS:
            targets.append(("tate_functional_rmse", f"TATE-{suffix}-{name}"))
            targets.append(("tcate_functional_rmse", f"TCATE-{suffix}-{name}"))
        targets.append(("reference_effect_rmse", f"REF-ATE-{suffix}"))
    targets.append(("reference_tcate_rmse", f"REF-TCATE-{suffix}"))
    return targets


_DECLARED_TARGETS = _level_set_targets("COMMON199") + _level_set_targets("INTERIOR")
