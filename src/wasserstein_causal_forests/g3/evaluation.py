"""Turn one fitted method into preregistered result rows.

Every row carries a metric ID and a target ID from
`research/estimand_contract.md`, and every metric is computed against the DGP's
oracle truth rather than against another method's output. Two rules from the
contract are enforced here rather than left to the analysis:

* A metric a method cannot supply produces a row with `status =
  "not_applicable"` and a stated reason, never a substitute quantity. The PTA
  endpoints estimate conditional means, so every law-level metric is
  inapplicable to them; that absence is a finding, not a hole.
* Barycenter-level and outcome-level errors keep separate metric IDs even when
  they are numerically related, so no row can be read as the other.

Aggregation across seeds happens later, in the merge and analysis steps. One
call here yields the rows for one manifest cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..pta_bcf.targets import GRID_FUNCTIONALS
from .dgps import N_MODERATOR_BINS, DistributionalDGP, moderator_bins
from .laws import (
    LawPrediction,
    chunk_rows,
    energy_risk_against_truth,
    kernel_law_error,
    median_heuristic_bandwidth,
    mode_coverage,
    tail_probability,
)
from .methods import MethodOutput

#: Reason strings recorded on inapplicable rows.
NO_LAW_REASON = (
    "method estimates conditional means of a fixed target vector, not the "
    "conditional law; contract section 4 forbids relabelling a mean surface"
)
NOT_IN_MANIFEST_REASON = (
    "functional was not in the method's frozen target manifest at fit time"
)


@dataclass(frozen=True)
class EvaluationManifest:
    """Everything the metric layer needs beyond the DGP and the fitted method."""

    manifest_id: str
    functionals: tuple[str, ...]
    #: Grid coordinate and threshold defining the `tail_calibration` event.
    tail_level_index: int
    tail_threshold: float
    #: Radius and mass floor defining a covered mode for `mode_coverage`.
    mode_radius: float
    mode_mass_floor: float
    collision_epsilon: float = 1e-3
    #: Test rows scored by the law-level metrics, whose cost is quadratic in
    #: the atom count and linear in the truth's node count.
    n_law_rows: int = 200
    #: A fitted atom counts as representing a degenerate-at-zero outcome law
    #: when its largest grid coordinate is at most this absolute tolerance.
    #: Frozen before any Phase 6.5 decisive run; the loose value is the primary
    #: definition because a boosted particle is an average, not a training row,
    #: and demanding exact zeros would rig the comparison toward empirical-law
    #: methods by construction.
    zero_mass_tolerance: float = 0.05


def _rmse(estimate: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    difference = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean(difference * difference)))


def _bin_means(values: NDArray[np.float64], bins: NDArray[np.int64]) -> NDArray[np.float64]:
    """Conditional mean of `values` in each moderator bin, shape (B,)."""

    means = np.full(N_MODERATOR_BINS, np.nan)
    for index in range(N_MODERATOR_BINS):
        rows = bins == index
        if np.any(rows):
            means[index] = float(np.mean(values[rows]))
    return means


def _truth_nodes(
    dgp: DistributionalDGP, X: NDArray[np.float64], arm: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pairs = list(dgp.iter_law_nodes(X, arm))
    nodes = np.stack([block for _, block in pairs], axis=1)
    weights = np.array([weight for weight, _ in pairs])
    # A mixture regime with covariate-dependent component weights supplies an
    # (n, J) matrix aligned with the block order; it replaces the shared vector.
    row_weights = dgp.law_node_weights(X, arm)
    if row_weights is not None:
        if row_weights.shape != (X.shape[0], weights.shape[0]):
            raise ValueError(
                f"{dgp.spec.dgp_id}: law_node_weights must have shape "
                f"({X.shape[0]}, {weights.shape[0]})"
            )
        weights = row_weights
    return nodes, weights


def _oracle_energy_risk(
    nodes: NDArray[np.float64],
    node_weights: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
    epsilon: float,
) -> NDArray[np.float64]:
    """Energy risk of the true law against itself, shape (n,).

    Substituting Phat = P into the score collapses the attraction and repulsion
    terms into one:

        E_{y ~ P} S_eps(P, y) = 0.5 sum_{j,l} w_j w_l d_eps(t_j, t_l),

    so the oracle floor costs one pairwise pass rather than two. It is also the
    only irreducible part of the risk, which is why every method is scored on
    its excess over this number rather than on the raw risk.

    `node_weights` may be a shared `(J,)` vector or per-row `(n, J)` weights
    for mixture regimes; the contraction gains an `n` index and nothing else
    changes.
    """

    from .laws import _squared_distances

    total = np.empty(nodes.shape[0])
    step = chunk_rows(nodes.shape[1], nodes.shape[1])
    for start in range(0, nodes.shape[0], step):
        rows = slice(start, start + step)
        block = nodes[rows]
        squared = _squared_distances(block, block, grid_weights)
        distances = np.sqrt(squared + epsilon * epsilon) - epsilon
        if node_weights.ndim == 1:
            total[rows] = 0.5 * np.einsum(
                "j,njl,l->n", node_weights, distances, node_weights
            )
        else:
            total[rows] = 0.5 * np.einsum(
                "nj,njl,nl->n", node_weights[rows], distances, node_weights[rows]
            )
    return total


#: Oracle truth for one arm of one test design is identical for every method in
#: a replication, and costs more than most fits. Caching it turns a per-method
#: expense into a per-replication one; the dispatcher groups a replication's
#: methods into one worker so the cache actually hits.
_TRUTH_CACHE: dict[tuple, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = {}
_TRUTH_CACHE_LIMIT = 4


def _cached_truth(
    dgp: DistributionalDGP,
    X: NDArray[np.float64],
    arm: int,
    manifest: "EvaluationManifest",
    cache_key: tuple | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    key = (
        None
        if cache_key is None
        else (*cache_key, dgp.spec.dgp_id, dgp.grid.n_grid, arm, X.shape[0])
    )
    if key is not None and key in _TRUTH_CACHE:
        return _TRUTH_CACHE[key]
    nodes, node_weights = _truth_nodes(dgp, X, arm)
    oracle = _oracle_energy_risk(
        nodes, node_weights, dgp.grid.weights, manifest.collision_epsilon
    )
    entry = (nodes, node_weights, oracle)
    if key is not None:
        if len(_TRUTH_CACHE) >= _TRUTH_CACHE_LIMIT:
            _TRUTH_CACHE.pop(next(iter(_TRUTH_CACHE)))
        _TRUTH_CACHE[key] = entry
    return entry


def _row(
    metric: str,
    target_id: str,
    value: float | None,
    *,
    arm: int | None = None,
    status: str = "ok",
    failure_reason: str = "",
    detail: str = "",
) -> dict[str, object]:
    return {
        "metric": metric,
        "target_id": target_id,
        "arm": arm,
        "detail": detail,
        "value": None if value is None else float(value),
        "status": status,
        "failure_reason": failure_reason,
    }


def implied_zero_mass(
    law: "LawPrediction", tolerance: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mass a fitted law puts on degenerate-at-zero atoms.

    Returns the loose-tolerance mass, the primary definition, and the exact
    fraction under a strict tolerance, as a diagnostic. Both work for shared
    atom banks (forests: weight on all-zero training rows) and row-specific
    particles (boosters: particle mass near the origin).
    """

    if law.shared_atoms:
        indicator = np.max(np.abs(law.atoms), axis=-1) <= tolerance
        exact = np.max(np.abs(law.atoms), axis=-1) <= 1e-9
        loose = law.weights @ indicator.astype(float)
        exact_mass = law.weights @ exact.astype(float)
    else:
        indicator = np.max(np.abs(law.atoms), axis=-1) <= tolerance
        exact = np.max(np.abs(law.atoms), axis=-1) <= 1e-9
        loose = np.einsum("na,na->n", law.weights, indicator.astype(float))
        exact_mass = np.einsum("na,na->n", law.weights, exact.astype(float))
    return loose, exact_mass


def evaluate(
    output: MethodOutput,
    dgp: DistributionalDGP,
    X_test: NDArray[np.float64],
    manifest: EvaluationManifest,
    *,
    cache_key: tuple | None = None,
) -> list[dict[str, object]]:
    """Return every preregistered metric row for one fitted method.

    `cache_key` identifies the test design, normally the test seed. Passing it
    lets successive methods on the same replication reuse one oracle truth
    instead of recomputing the most expensive part of the evaluation.
    """

    rows: list[dict[str, object]] = []
    bins = moderator_bins(X_test)
    grid_weights = dgp.grid.weights

    # ------------------------------------------------ barycenter-level targets
    for arm in (0, 1):
        rows.append(
            _row(
                "barycenter_rmse",
                "MEANQ-A-K",
                _rmse(output.mean_quantiles[arm], dgp.mean_quantiles(X_test, arm)),
                arm=arm,
            )
        )
    estimated_contrast = output.mean_quantiles[1] - output.mean_quantiles[0]
    rows.append(
        _row(
            "mean_quantile_rmse",
            "MEANQ-A-K",
            _rmse(estimated_contrast, dgp.mean_quantile_contrast(X_test)),
            detail="arm contrast, the mandatory common target",
        )
    )

    # ------------------------------------------------- outcome-level functionals
    for name in GRID_FUNCTIONALS:
        if name not in output.supported_functionals:
            rows.append(
                _row(
                    "tate_functional_rmse",
                    f"TATE-K-{name}",
                    None,
                    status="not_applicable",
                    failure_reason=NOT_IN_MANIFEST_REASON,
                )
            )
            rows.append(
                _row(
                    "tcate_functional_rmse",
                    f"TCATE-K-{name}",
                    None,
                    status="not_applicable",
                    failure_reason=NOT_IN_MANIFEST_REASON,
                )
            )
            continue
        estimated = output.functionals[name][1] - output.functionals[name][0]
        truth = dgp.functional_contrast(X_test, name)
        rows.append(
            _row(
                "tate_functional_rmse",
                f"TATE-K-{name}",
                abs(float(np.mean(estimated) - np.mean(truth))),
                detail="absolute error of the marginal effect",
            )
        )
        rows.append(
            _row(
                "tcate_functional_rmse",
                f"TCATE-K-{name}",
                _rmse(_bin_means(estimated, bins), _bin_means(truth, bins)),
                detail="across moderator bins",
            )
        )

    # -------------------------------------------------------- reference targets
    if output.reference is None:
        for metric, target in (
            ("reference_effect_rmse", "REF-ATE-K"),
            ("reference_tcate_rmse", "REF-TCATE-K"),
        ):
            rows.append(
                _row(metric, target, None, status="not_applicable",
                     failure_reason="reference coordinate absent from the manifest")
            )
    else:
        estimated = output.reference[1] - output.reference[0]
        truth = dgp.reference_contrast(X_test)
        rows.append(
            _row(
                "reference_effect_rmse",
                "REF-ATE-K",
                abs(float(np.mean(estimated) - np.mean(truth))),
            )
        )
        rows.append(
            _row(
                "reference_tcate_rmse",
                "REF-TCATE-K",
                _rmse(_bin_means(estimated, bins), _bin_means(truth, bins)),
            )
        )

    rows.extend(_law_rows(output, dgp, X_test, manifest, cache_key))
    rows.extend(_zero_mass_rows(output, dgp, X_test, bins, manifest))

    # ------------------------------------------------------- operational rows
    rows.append(
        _row("runtime", "NONE_OPERATIONAL", output.fit_seconds + output.predict_seconds,
             detail="fit plus predict, seconds")
    )
    rows.append(
        _row("peak_ram", "NONE_OPERATIONAL", output.peak_ram_mb, detail="megabytes")
    )
    for name, value in output.diagnostics.items():
        rows.append(_row(f"diagnostic_{name}", "NONE_OPERATIONAL", value))
    return rows


def _law_rows(
    output: MethodOutput,
    dgp: DistributionalDGP,
    X_test: NDArray[np.float64],
    manifest: EvaluationManifest,
    cache_key: tuple | None = None,
) -> list[dict[str, object]]:
    """Law-level metrics, or an explicit inapplicability record."""

    law_metrics = (
        ("arm_energy_risk", "LAW-A-M-K"),
        ("kernel_law_error", "LAW-A-K"),
        ("tail_calibration", "LAW-A-K"),
        ("mode_coverage", "LAW-A-K"),
    )
    if output.law is None:
        return [
            _row(metric, target, None, status="not_applicable",
                 failure_reason=NO_LAW_REASON)
            for metric, target in law_metrics
        ]

    rows: list[dict[str, object]] = []
    n_rows = min(manifest.n_law_rows, X_test.shape[0])
    subset = slice(0, n_rows)
    X_subset = X_test[subset]

    for arm in (0, 1):
        law = _restrict(output.law[arm], subset)
        nodes, node_weights, oracle_risk = _cached_truth(
            dgp, X_subset, arm, manifest, cache_key
        )
        bandwidth = median_heuristic_bandwidth(nodes[: min(32, n_rows)], dgp.grid.weights)

        risk = energy_risk_against_truth(
            law, nodes, node_weights, dgp.grid.weights,
            epsilon=manifest.collision_epsilon,
        )
        # Excess risk over the oracle is the comparable quantity: the raw risk
        # contains an irreducible term that varies by regime and arm.
        rows.append(
            _row("arm_energy_risk", "LAW-A-M-K",
                 float(np.mean(risk - oracle_risk)), arm=arm,
                 detail="excess energy risk over the true law")
        )
        rows.append(
            _row("kernel_law_error", "LAW-A-K",
                 float(np.mean(kernel_law_error(
                     law, nodes, node_weights, dgp.grid.weights, bandwidth=bandwidth
                 ))),
                 arm=arm, detail="squared MMD, Gaussian kernel, median bandwidth")
        )
        rows.append(
            _row("tail_calibration", "LAW-A-K",
                 float(np.mean(np.abs(
                     tail_probability(law, level_index=manifest.tail_level_index,
                                      threshold=manifest.tail_threshold)
                     - dgp.tail_probability(X_subset, arm,
                                            level_index=manifest.tail_level_index,
                                            threshold=manifest.tail_threshold)
                 ))),
                 arm=arm,
                 detail=f"|P̂ - P| at grid index {manifest.tail_level_index}")
        )
        rows.append(
            _row("mode_coverage", "LAW-A-K",
                 float(np.mean(mode_coverage(
                     law, dgp.conditional_mode_centres(X_subset, arm),
                     dgp.grid.weights,
                     radius=manifest.mode_radius,
                     mass_floor=manifest.mode_mass_floor,
                 ))),
                 arm=arm, detail="fraction of true outer modes carrying mass")
        )
    return rows


def _restrict(law: LawPrediction, rows: slice) -> LawPrediction:
    """Restrict a law estimate to a block of test rows."""

    if law.shared_atoms:
        return LawPrediction(
            atoms=law.atoms, weights=law.weights[rows], shared_atoms=True
        )
    return LawPrediction(
        atoms=law.atoms[rows], weights=law.weights[rows], shared_atoms=False
    )


def _zero_mass_rows(
    output: MethodOutput,
    dgp: DistributionalDGP,
    X_test: NDArray[np.float64],
    bins: NDArray[np.int64],
    manifest: EvaluationManifest,
) -> list[dict[str, object]]:
    """Zero-inflation metrics, or an explicit inapplicability record.

    Three cases, each recorded rather than silently dropped: the method holds
    no law (PTA endpoints), the regime has no degenerate component (every
    frozen suite member), or both quantities exist and are compared. The
    contrast metric is the moderator-binned RMSE of the arm difference in
    degenerate-component probability.
    """

    if output.law is None:
        reasons = (
            ("zero_mass_abs_error", "method produces no conditional law"),
            ("mass_contrast_rmse", "method produces no conditional law"),
        )
        return [
            _row(metric, "LAW-A-K", None, status="not_applicable",
                 failure_reason=reason)
            for metric, reason in reasons
        ]
    truth_zero = dgp.zero_type_probability(X_test, 0)
    if truth_zero is None:
        return [
            _row("zero_mass_abs_error", "LAW-A-K", None,
                 status="not_applicable",
                 failure_reason="regime has no degenerate component"),
            _row("mass_contrast_rmse", "LAW-A-K", None,
                 status="not_applicable",
                 failure_reason="regime has no degenerate component"),
        ]

    rows: list[dict[str, object]] = []
    implied = {}
    exact = {}
    for arm in (0, 1):
        loose, strict = implied_zero_mass(output.law[arm],
                                          manifest.zero_mass_tolerance)
        implied[arm] = loose
        exact[arm] = float(np.mean(strict))
        rows.append(
            _row("zero_mass_abs_error", "LAW-A-K",
                 _rmse(implied[arm], dgp.zero_type_probability(X_test, arm)),
                 arm=arm,
                 detail=(f"loose tolerance {manifest.zero_mass_tolerance}; "
                         f"exact-atom fraction {exact[arm]:.4f}"))
        )
    contrast = implied[1] - implied[0]
    truth_contrast = (
        dgp.zero_type_probability(X_test, 1)
        - dgp.zero_type_probability(X_test, 0)
    )
    rows.append(
        _row("mass_contrast_rmse", "LAW-A-K",
             _rmse(_bin_means(contrast, bins), _bin_means(truth_contrast, bins)),
             detail="degenerate-probability contrast across moderator bins")
    )
    return rows
