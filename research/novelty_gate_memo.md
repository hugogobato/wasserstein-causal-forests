# G1 Novelty Gate Memo

**Date:** 2026-07-30  
**Gate:** G1, direct-hit prior-art audit  
**Decision:** `GO` for narrow C-WDB scope; `INCREMENTAL-ONLY` for PTA-BCF as a standalone novelty claim.

## Decision

The broad idea is not novel: causal inference for distribution-valued outcomes, Wasserstein causal effects, conditional distribution forests, shared multivariate Bayesian trees, and particle distributional regression all have material precedents. The strongest target collision is Du et al.'s Wasserstein Random Forest, which estimates conditional marginal laws for treatment and control groups separately. The strongest treatment-aware forest collision is Causal-DRF, which targets a conditional kernel treatment effect from a shared forest.

A narrower C-WDB claim survives. C-WDB is a treatment-aware learner for the finite-grid conditional law of a distribution-valued potential outcome, with an explicit fixed-\(M\) equal-weight particle output obtained by a strictly proper energy score on observed quantile-vector outcomes. The inspected WRF, Causal-DRF, and WGBoost sources do not implement this same combination. In particular, the score's ensemble-coupled particle repulsion and post-fit law-invariant functional extraction are plausible capability differences. The claim is still `promising but unproven` until comparative experiments establish value beyond WRF and Causal-DRF.

PTA-BCF does not clear an independent novelty gate. MVBCF, shared Bayesian forest work, and multiVCBART cover shared or multivariate Bayesian-tree ingredients. PTA remains a useful baseline or a conditional adaptive-pooling extension, but its novelty must be earned through a pre-registered crossover result against separate and forced-sharing Bayesian learners.

## Mandatory novelty boundary

The paper must not claim the first Wasserstein causal method, the first causal distribution-valued outcome method, the first distributional forest, or the first multivariate causal forest. The defensible claim is narrower: a treatment-aware boosted-tree conditional-law learner using strictly proper finite-grid energy-score particle projection, with explicit downstream outcome-level functionals.

## Required follow-up

G2 must compare C-WDB with WRF, Causal-DRF on the same rescaled quantile grid, and separate fixed-target baselines. It must stress particle count, grid resolution, misspecification, arm imbalance, and computational cost. G3 must establish whether the extracted functionals improve an applied decision or scientific question. No software provenance audit was performed or recorded in this memo.

See [`research/prior_art_matrix.csv`](prior_art_matrix.csv) for the full source-by-source evidence ledger and the phase file for exact supporting locations and references.
