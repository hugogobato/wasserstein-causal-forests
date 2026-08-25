"""Phase G3 preregistered simulation tournament."""

from __future__ import annotations

from .dgps import (
    DGP_IDS,
    DGPSample,
    DGPSpec,
    DistributionalDGP,
    GridSpec,
    OuterLaw,
    build_dgp,
    moderator_bins,
)

__all__ = [
    "DGP_IDS",
    "DGPSample",
    "DGPSpec",
    "DistributionalDGP",
    "GridSpec",
    "OuterLaw",
    "build_dgp",
    "moderator_bins",
]
