#!/usr/bin/env python3
"""Deterministic Phase G0 checks for the C-WDB score and estimand contract.

Run from the repository root:

    python research/checks/check_proper_score.py

The script uses only NumPy and the Python standard library. It exits nonzero
when a G0 obligation fails and prints a JSON certificate on success.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GRADIENT_TOL = 1e-6
PERMUTATION_TOL = 1e-12
PROJECTION_TOL = 1e-10
COLLAPSE_TOL = 1e-12


def validate_weights(weights: np.ndarray) -> np.ndarray:
    """Return a validated one-dimensional positive weight vector."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be finite and strictly positive")
    if not np.isclose(weights.sum(), 1.0, atol=1e-14, rtol=0.0):
        raise ValueError("weights must sum to one under contract G0-WP0-A-v1")
    return weights


def squared_weighted_norm(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute sum_k w_k values_k^2 along the last axis."""
    weights = validate_weights(weights)
    values = np.asarray(values, dtype=float)
    if values.shape[-1] != weights.size:
        raise ValueError("last coordinate dimension does not match weights")
    return np.einsum("...k,k,...k->...", values, weights, values)


def rho_epsilon(
    differences: np.ndarray, weights: np.ndarray, epsilon: float = 0.0
) -> np.ndarray:
    """Certified collision-smoothed weighted distance.

    rho_epsilon(v) = sqrt(||v||_W^2 + epsilon^2) - epsilon.
    """
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and nonnegative")
    squared = squared_weighted_norm(differences, weights)
    return np.sqrt(np.maximum(squared, 0.0) + epsilon * epsilon) - epsilon


def energy_score(
    particles: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 0.0,
) -> float:
    """Equal-weight empirical energy loss S_epsilon(P_M, outcome)."""
    particles = np.asarray(particles, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if particles.ndim != 2 or outcome.shape != (particles.shape[1],):
        raise ValueError("particles must be M by K and outcome must have length K")
    attraction = np.mean(rho_epsilon(particles - outcome, weights, epsilon))
    pairwise = particles[:, None, :] - particles[None, :, :]
    repulsion = 0.5 * np.mean(rho_epsilon(pairwise, weights, epsilon))
    return float(attraction - repulsion)


def energy_gradient(
    particles: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Full ensemble-coupled gradient of ``energy_score``.

    For epsilon=0, the selected subgradient of a zero displacement is zero.
    For epsilon>0, this is the ordinary gradient of the certified smooth score.
    """
    particles = np.asarray(particles, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    weights = validate_weights(weights)
    if particles.ndim != 2 or outcome.shape != (particles.shape[1],):
        raise ValueError("particles must be M by K and outcome must have length K")

    m_particles = particles.shape[0]
    attraction_difference = particles - outcome
    attraction_squared = squared_weighted_norm(attraction_difference, weights)
    attraction_denominator = np.sqrt(attraction_squared + epsilon * epsilon)
    attraction_direction = np.divide(
        attraction_difference * weights,
        attraction_denominator[:, None],
        out=np.zeros_like(attraction_difference),
        where=attraction_denominator[:, None] > 0,
    )

    pairwise_difference = particles[:, None, :] - particles[None, :, :]
    pairwise_squared = squared_weighted_norm(pairwise_difference, weights)
    pairwise_denominator = np.sqrt(pairwise_squared + epsilon * epsilon)
    pairwise_direction = np.divide(
        pairwise_difference * weights,
        pairwise_denominator[:, :, None],
        out=np.zeros_like(pairwise_difference),
        where=pairwise_denominator[:, :, None] > 0,
    )

    return (
        attraction_direction / m_particles
        - pairwise_direction.sum(axis=1) / (m_particles * m_particles)
    )


def finite_difference_gradient(
    particles: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
    step: float = 1e-6,
) -> np.ndarray:
    """Central finite-difference gradient for an away-from-kink test case."""
    numerical = np.zeros_like(particles, dtype=float)
    for index in np.ndindex(particles.shape):
        plus = particles.copy()
        minus = particles.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (
            energy_score(plus, outcome, weights, epsilon)
            - energy_score(minus, outcome, weights, epsilon)
        ) / (2.0 * step)
    return numerical


def weighted_isotonic_projection(
    values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Project onto q_1 <= ... <= q_K using weighted pool-adjacent violators."""
    values = np.asarray(values, dtype=float)
    weights = validate_weights(weights)
    if values.shape != weights.shape or not np.all(np.isfinite(values)):
        raise ValueError("values and weights must be finite vectors of equal length")

    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([index, index, float(weight), float(value)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left[2]) + float(right[2])
            merged_value = (
                float(left[2]) * float(left[3])
                + float(right[2]) * float(right[3])
            ) / merged_weight
            blocks.append([int(left[0]), int(right[1]), merged_weight, merged_value])

    projected = np.empty_like(values)
    for start, end, _, level in blocks:
        projected[int(start) : int(end) + 1] = float(level)
    return projected


def projection_kkt_residual(
    values: np.ndarray, projected: np.ndarray, weights: np.ndarray
) -> float:
    """Return a KKT residual for weighted projection onto the monotone cone."""
    values = np.asarray(values, dtype=float)
    projected = np.asarray(projected, dtype=float)
    weights = validate_weights(weights)
    stationarity_residual = weights * (projected - values)
    multipliers = -np.cumsum(stationarity_residual)[:-1]
    primal_violation = np.maximum(projected[:-1] - projected[1:], 0.0)
    dual_violation = np.maximum(-multipliers, 0.0)
    complementarity = multipliers * (projected[:-1] - projected[1:])
    return float(
        max(
            np.max(primal_violation, initial=0.0),
            np.max(dual_violation, initial=0.0),
            np.max(np.abs(complementarity), initial=0.0),
            abs(np.sum(stationarity_residual)),
        )
    )


def empirical_energy_distance(
    sample_p: np.ndarray,
    sample_q: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 0.0,
) -> float:
    """Energy distance between two equally or unequally sized empirical laws."""
    sample_p = np.asarray(sample_p, dtype=float)
    sample_q = np.asarray(sample_q, dtype=float)
    cross = rho_epsilon(sample_p[:, None, :] - sample_q[None, :, :], weights, epsilon)
    within_p = rho_epsilon(
        sample_p[:, None, :] - sample_p[None, :, :], weights, epsilon
    )
    within_q = rho_epsilon(
        sample_q[:, None, :] - sample_q[None, :, :], weights, epsilon
    )
    return float(2.0 * cross.mean() - within_p.mean() - within_q.mean())


def law_invariant_outputs(
    particles: np.ndarray,
    outcome: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Small public-output vector used by the permutation certificate."""
    pairwise = particles[:, None, :] - particles[None, :, :]
    return np.concatenate(
        [
            np.asarray([energy_score(particles, outcome, weights, epsilon)]),
            particles.mean(axis=0),
            np.asarray(
                [
                    np.mean(rho_epsilon(particles - reference, weights, epsilon)),
                    np.mean(rho_epsilon(pairwise, weights, epsilon)),
                ]
            ),
        ]
    )


def validate_metric_semantics(target_level: str, prediction_level: str) -> None:
    """Reject use of a barycenter prediction for an outcome-level target."""
    if target_level.strip().lower() == "outcome" and prediction_level.strip().lower() in {
        "barycenter",
        "barycenter_draw",
        "barycenter_draws",
        "mean_quantile",
    }:
        raise ValueError(
            "barycenter or mean-quantile predictions cannot be used as "
            "outcome-level draws"
        )


FORBIDDEN_BARYCENTER_PATTERNS = (
    re.compile(r"outcome_draws\s*=\s*barycenter_draws", re.IGNORECASE),
    re.compile(r"tate_out\s*=\s*e\s*\[\s*t\s*\(\s*m_", re.IGNORECASE),
    re.compile(
        r"target_level\s*[:=]\s*outcome[\s,;]+"
        r"prediction_level\s*[:=]\s*barycenter",
        re.IGNORECASE,
    ),
)


def lint_barycenter_substitution(text: str) -> None:
    """Reject explicit text forms of the barycenter plug-in fallacy."""
    for pattern in FORBIDDEN_BARYCENTER_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"forbidden barycenter-as-outcome substitution: {pattern.pattern}"
            )


def check_weighted_geometry() -> dict[str, float]:
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    q_left = np.asarray([-2.0, -0.5, 0.25, 3.0])
    q_right = np.asarray([-1.5, -0.25, 1.0, 2.0])
    z_left = np.sqrt(weights) * q_left
    z_right = np.sqrt(weights) * q_right
    weighted_squared = float(squared_weighted_norm(q_left - q_right, weights))
    rescaled_squared = float(np.sum((z_left - z_right) ** 2))
    error = abs(weighted_squared - rescaled_squared)
    if error >= PERMUTATION_TOL:
        raise AssertionError(f"weighted geometry mismatch: {error}")
    return {"weighted_rescaling_error": error}


def check_gradients() -> dict[str, float]:
    weights = np.asarray([0.07, 0.13, 0.2, 0.25, 0.35])
    particles = np.asarray(
        [
            [-2.0, -1.0, -0.2, 0.3, 1.0],
            [-1.5, -0.4, 0.1, 0.9, 1.7],
            [-0.8, 0.0, 0.8, 1.4, 2.2],
            [-2.4, -1.7, -0.9, -0.1, 0.5],
        ]
    )
    outcome = np.asarray([-1.2, -0.6, 0.4, 1.1, 1.9])
    errors = {}
    for epsilon in (0.0, 1e-3):
        analytic = energy_gradient(particles, outcome, weights, epsilon)
        numerical = finite_difference_gradient(
            particles, outcome, weights, epsilon, step=1e-6
        )
        error = float(np.max(np.abs(analytic - numerical)))
        errors[f"gradient_error_epsilon_{epsilon:g}"] = error
        if error >= GRADIENT_TOL:
            raise AssertionError(f"gradient error {error} exceeds {GRADIENT_TOL}")
    return errors


def check_collision_behavior() -> dict[str, float]:
    weights = np.asarray([0.2, 0.3, 0.5])
    outcome = np.asarray([-1.0, 0.0, 1.0])
    particles = np.asarray(
        [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-0.5, 0.4, 1.4],
        ]
    )
    diagnostics = {}
    for epsilon in (0.0, 1e-4):
        score = energy_score(particles, outcome, weights, epsilon)
        gradient = energy_gradient(particles, outcome, weights, epsilon)
        if not np.isfinite(score) or not np.all(np.isfinite(gradient)):
            raise AssertionError("score or gradient is nonfinite at a collision")
        diagnostics[f"collision_score_epsilon_{epsilon:g}"] = score
        diagnostics[f"collision_gradient_max_epsilon_{epsilon:g}"] = float(
            np.max(np.abs(gradient))
        )
    return diagnostics


def check_particle_permutations() -> dict[str, float]:
    weights = np.asarray([0.2, 0.3, 0.5])
    particles = np.asarray(
        [
            [-2.0, -1.0, 0.0],
            [-1.0, 0.2, 1.0],
            [0.0, 0.7, 2.0],
            [1.0, 1.5, 3.0],
        ]
    )
    outcome = np.asarray([-0.7, 0.1, 1.2])
    reference = np.asarray([-1.0, 0.0, 1.0])
    permutation = np.asarray([2, 0, 3, 1])
    epsilon = 1e-4
    original = law_invariant_outputs(
        particles, outcome, reference, weights, epsilon
    )
    permuted = law_invariant_outputs(
        particles[permutation], outcome, reference, weights, epsilon
    )
    output_error = float(np.max(np.abs(original - permuted)))

    original_gradient = energy_gradient(particles, outcome, weights, epsilon)
    permuted_gradient = energy_gradient(
        particles[permutation], outcome, weights, epsilon
    )
    equivariance_error = float(
        np.max(np.abs(permuted_gradient - original_gradient[permutation]))
    )
    if max(output_error, equivariance_error) >= PERMUTATION_TOL:
        raise AssertionError("particle permutation certificate failed")
    return {
        "permutation_output_error": output_error,
        "permutation_gradient_equivariance_error": equivariance_error,
    }


def check_monotone_projection() -> dict[str, float]:
    weights = np.asarray([0.05, 0.15, 0.25, 0.25, 0.30])
    values = np.asarray([2.0, -1.0, 1.5, 0.5, 4.0])
    projected = weighted_isotonic_projection(values, weights)
    residual = projection_kkt_residual(values, projected, weights)
    if np.any(np.diff(projected) < -PROJECTION_TOL):
        raise AssertionError("projection is outside the monotone cone")
    if residual >= PROJECTION_TOL:
        raise AssertionError(f"projection KKT residual is {residual}")
    score = energy_score(
        np.vstack([projected, projected]),
        projected,
        weights,
        epsilon=0.0,
    )
    if not np.isfinite(score):
        raise AssertionError("projected particles yield nonfinite loss")
    return {
        "projection_kkt_residual": residual,
        "projected_minimum_increment": float(np.min(np.diff(projected))),
        "projected_collision_score": score,
    }


def check_strict_propriety_witness() -> dict[str, float]:
    weights = np.asarray([0.25, 0.25, 0.25, 0.25])
    base = np.asarray([-2.0, -0.5, 0.5, 2.0])
    truth = np.vstack([base - 1.0, base + 1.0])
    collapsed = np.vstack([base, base])
    same_distance = empirical_energy_distance(truth, truth, weights, epsilon=0.0)
    collapsed_distance = empirical_energy_distance(
        truth, collapsed, weights, epsilon=0.0
    )
    smooth_collapsed_distance = empirical_energy_distance(
        truth, collapsed, weights, epsilon=1e-3
    )
    if abs(same_distance) >= PERMUTATION_TOL:
        raise AssertionError("energy distance of a law from itself is nonzero")
    if min(collapsed_distance, smooth_collapsed_distance) <= 0:
        raise AssertionError("energy distance failed to separate a collapsed law")
    return {
        "same_law_energy_distance": same_distance,
        "collapsed_energy_distance": collapsed_distance,
        "smooth_collapsed_energy_distance": smooth_collapsed_distance,
    }


def uniform_midpoint_risk(m_particles: int) -> float:
    """Exact energy risk for midpoint particles against Uniform[-1, 1]."""
    if m_particles <= 0:
        raise ValueError("m_particles must be positive")
    centers = -1.0 + (2.0 * np.arange(m_particles) + 1.0) / m_particles
    attraction = np.mean((centers * centers + 1.0) / 2.0)
    repulsion = 0.5 * np.mean(
        np.abs(centers[:, None] - centers[None, :])
    )
    return float(attraction - repulsion)


def check_fixed_m_risk_ladder() -> dict[str, object]:
    particle_counts = np.asarray([2, 5, 10, 25])
    risks = np.asarray([uniform_midpoint_risk(int(m)) for m in particle_counts])
    formula = 1.0 / 3.0 + 1.0 / (6.0 * particle_counts**2)
    formula_error = float(np.max(np.abs(risks - formula)))
    if formula_error >= PERMUTATION_TOL:
        raise AssertionError("known-law finite-M risk does not match exact formula")
    if not np.all(np.diff(risks) < 0):
        raise AssertionError("known-law approximation risk is not strictly decreasing")
    if not np.all(risks > 1.0 / 3.0):
        raise AssertionError("finite-M risk must exceed unrestricted oracle risk")
    return {
        "particle_counts": particle_counts.tolist(),
        "risks": risks.tolist(),
        "oracle_risk": 1.0 / 3.0,
        "formula_max_error": formula_error,
    }


def check_squared_w2_collapse() -> dict[str, float]:
    """Show that independent squared-W2 particles all target the barycenter."""
    m_particles = 10
    outcomes = np.linspace(-1.0, 1.0, 1001)
    outcome_mean = float(np.mean(outcomes))
    particles = np.linspace(-2.0, 2.0, m_particles)
    initial_spread = float(np.ptp(particles))
    learning_rate = m_particles / 4.0
    for _ in range(60):
        gradient = (2.0 / m_particles) * (particles - outcome_mean)
        particles -= learning_rate * gradient
    final_spread = float(np.ptp(particles))
    center_error = float(np.max(np.abs(particles - outcome_mean)))
    if final_spread >= COLLAPSE_TOL or center_error >= COLLAPSE_TOL:
        raise AssertionError("squared-W2 ablation did not collapse to the barycenter")
    return {
        "collapse_initial_spread": initial_spread,
        "collapse_final_spread": final_spread,
        "collapse_center_error": center_error,
    }


def check_estimand_contract_lint() -> dict[str, object]:
    contract_path = ROOT / "research" / "estimand_contract.md"
    contract = contract_path.read_text(encoding="utf-8")
    lint_barycenter_substitution(contract)

    rejected_structured = False
    try:
        validate_metric_semantics("outcome", "barycenter_draws")
    except ValueError:
        rejected_structured = True

    rejected_text = False
    try:
        lint_barycenter_substitution("outcome_draws = barycenter_draws")
    except ValueError:
        rejected_text = True

    if not rejected_structured or not rejected_text:
        raise AssertionError("barycenter substitution witness was not rejected")

    required_metric_ids = {
        "arm_energy_risk",
        "kernel_law_error",
        "mean_quantile_rmse",
        "tate_functional_rmse",
        "tcate_functional_rmse",
        "reference_effect_rmse",
        "reference_tcate_rmse",
        "barycenter_rmse",
        "tail_calibration",
        "mode_coverage",
        "runtime",
        "peak_ram",
    }
    missing = sorted(metric for metric in required_metric_ids if metric not in contract)
    if missing:
        raise AssertionError(f"metric registry is missing {missing}")
    return {
        "structured_barycenter_witness_rejected": rejected_structured,
        "text_barycenter_witness_rejected": rejected_text,
        "registered_metric_count": len(required_metric_ids),
    }


CHECKS: tuple[tuple[str, Callable[[], dict[str, object]]], ...] = (
    ("weighted_geometry", check_weighted_geometry),
    ("analytic_gradient", check_gradients),
    ("collision_behavior", check_collision_behavior),
    ("particle_permutation", check_particle_permutations),
    ("monotone_projection", check_monotone_projection),
    ("strict_propriety_witness", check_strict_propriety_witness),
    ("fixed_m_risk_ladder", check_fixed_m_risk_ladder),
    ("squared_w2_collapse", check_squared_w2_collapse),
    ("estimand_contract_lint", check_estimand_contract_lint),
)


def main() -> int:
    started = time.perf_counter()
    results: dict[str, object] = {}
    failures: dict[str, str] = {}
    for name, check in CHECKS:
        try:
            results[name] = check()
        except Exception as error:  # noqa: BLE001 - certificate must collect all failures
            failures[name] = f"{type(error).__name__}: {error}"

    certificate = {
        "certificate_id": "G0-WP0-B-v1",
        "status": "PASS" if not failures else "FAIL",
        "thresholds": {
            "gradient": GRADIENT_TOL,
            "permutation": PERMUTATION_TOL,
            "projection": PROJECTION_TOL,
            "collapse": COLLAPSE_TOL,
        },
        "results": results,
        "failures": failures,
        "runtime_seconds": time.perf_counter() - started,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
