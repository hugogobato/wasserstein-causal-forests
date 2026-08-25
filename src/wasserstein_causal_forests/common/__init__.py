"""Shared validation and finite-grid quantile utilities."""

from .quantiles import (
    canonicalize_particles,
    is_monotone,
    validate_quantiles,
    validate_weights,
)

__all__ = [
    "canonicalize_particles",
    "is_monotone",
    "validate_quantiles",
    "validate_weights",
]

