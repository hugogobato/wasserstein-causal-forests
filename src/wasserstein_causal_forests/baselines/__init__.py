"""Python drivers for the WP2-C forest baselines.

The baselines themselves are R: `research/baselines/drf_tlearner.R` wraps the
pinned GPL-3.0 `drf` package for W-DRF-T, and `research/baselines/causal_drf_r/`
holds the Causal-DRF reimplementation. This package owns only the reproduction
driver and the result schema shared with the rest of the project.

Nothing is imported here, so that `python -m
wasserstein_causal_forests.baselines.reproduction` runs the module as a script
without importing it twice.
"""

__all__ = ["reproduction"]
