#!/usr/bin/env python3
"""Tutorial 01: Quickstart on Synthetic Distributional Data.

Demonstrates how to use Wasserstein Causal Forests (WCF) end to end:
1. Representing each unit i by a K-grid quantile vector Q_i in Q_K on the
   uniform midpoint grid u_k = (k - 0.5) / K (default K = 25).
2. Declaring distributional functionals (mean, standard deviation, skewness,
   upper-half mean, 10th/90th percentiles, and reference Wasserstein distance
   d_{W,K}(q, q_*^K) to a normative benchmark law q_*^K).
3. Fitting DRCalibratedCWDB, which combines:
   - Cross-fitted M-particle Wasserstein energy-score boosting on the monotone
     cone Q_K with arm-shared trees and out-of-fold contrast-shrinkage selection
     lambda^* in {0, 50, 500}, and
   - Three-fold augmented inverse-propensity weighting (AIPW) calibration with
     influence-function standard errors.
4. Predicting conditional particle laws P_hat_{a,M}^K(x) and conditional mean
   quantile curves q_bar_{a,M}^K(x) on held-out covariates.
5. Inspecting marginal TATE and moderator-quartile TCATE estimates alongside
   their influence-function standard errors and saving a summary figure.

Run from the repository root:
    python tutorials/01_quickstart_synthetic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from wasserstein_causal_forests import (
    DRCalibratedCWDB,
    ecdf_moderator,
    grid_weights,
    make_functionals,
    midpoint_levels,
    plugin_contrasts,
)


def simulate_distributional_sample(
    n: int = 300,
    n_grid: int = 25,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a synthetic observational sample of unit-level quantile curves.

    Each unit i has covariates X_i in R^4 (with X_{i,0} rank-transformed into
    (-1, 1) so the four bins (-1, -0.5], (-0.5, 0], (0, 0.5], (0.5, 1) are
    moderator quartiles), confounded binary treatment A_i in {0, 1}, and a
    realized K-point quantile curve Q_i on u_k = (k - 0.5) / K.

    Treatment increases location (heterogeneously in the moderator X_{i,0}),
    reduces dispersion, and shifts units closer to an external benchmark law
    q_*^K = Phi^{-1}(u_k) * 0.85 + 1.0.
    """
    rng = np.random.default_rng(seed)
    u = midpoint_levels(n_grid)
    z = norm.ppf(u)

    raw_x = rng.normal(size=(n, 4))
    moderator_raw = raw_x[:, 0].copy()
    X = raw_x.copy()
    X[:, 0] = ecdf_moderator(moderator_raw)

    # Confounded propensity e(X) in (0.15, 0.85)
    logits = 0.55 * X[:, 0] - 0.35 * X[:, 1] + 0.25 * np.sin(X[:, 2])
    propensity = 1.0 / (1.0 + np.exp(-logits))
    A = rng.binomial(1, propensity, size=n).astype(np.int64)

    # Unit-specific random location and scale shocks (making Q_i a random law)
    xi_loc = rng.normal(0.0, 0.28, size=n)
    xi_scale = np.exp(rng.normal(0.0, 0.14, size=n))

    # Baseline and treated location/scale surfaces
    mu_0 = 0.30 * X[:, 0] - 0.25 * X[:, 1] + 0.15 * X[:, 2]
    tau_loc = 0.45 + 0.30 * (X[:, 0] > 0.0).astype(float)
    sigma_0 = 1.10 + 0.15 * np.abs(X[:, 1])
    sigma_1 = 0.88 + 0.10 * np.abs(X[:, 1])

    loc = mu_0 + A * tau_loc + xi_loc
    scale = ((1 - A) * sigma_0 + A * sigma_1) * xi_scale

    # Realized monotone quantile curves Q_i in Q_K
    Q = loc[:, None] + scale[:, None] * z[None, :]
    Q = np.maximum.accumulate(Q, axis=1)

    # Normative reference benchmark law q_*^K
    q_star = 1.00 + 0.85 * z

    return {
        "X": X,
        "A": A,
        "Q": Q,
        "q_star": q_star,
        "u": u,
        "propensity": propensity,
    }


def main() -> None:
    print("=" * 72)
    print("Tutorial 01: Wasserstein Causal Forests (WCF) Synthetic Quickstart")
    print("=" * 72)

    data = simulate_distributional_sample(n=300, n_grid=25, seed=42)
    X, A, Q, q_star, u = (
        data["X"],
        data["A"],
        data["Q"],
        data["q_star"],
        data["u"],
    )
    weights = grid_weights(Q.shape[1])

    # Declare the 5 standard WCF functionals + 10th and 90th percentiles
    functionals = make_functionals(q_star, weights)
    idx_p10 = int(np.argmin(np.abs(u - 0.10)))
    idx_p90 = int(np.argmin(np.abs(u - 0.90)))
    functionals["p10"] = lambda q, k=idx_p10: np.asarray(q, dtype=float)[:, k]
    functionals["p90"] = lambda q, k=idx_p90: np.asarray(q, dtype=float)[:, k]

    # Fit DRCalibratedCWDB (using 60 trees here for a fast tutorial run;
    # paper default is n_estimators=100, n_particles=10, learning_rate=0.12)
    model = DRCalibratedCWDB(
        functionals=functionals,
        contrast_candidates=(0.0, 50.0, 500.0),
        n_folds=3,
        n_particles=10,
        n_estimators=60,
        learning_rate=0.12,
        max_depth=4,
        min_samples_leaf=10,
        min_arm_leaf=5,
        collision_epsilon=1e-3,
        arm_shrinkage=5.0,
        architecture="v1",
        sharing="partial",
        init_sharing="pooled",
        random_state=0,
    )
    model.fit(X, A, Q, weights)

    print(f"\nSample size: n = {X.shape[0]} (treated = {int(A.sum())}, control = {int((A == 0).sum())})")
    print(f"Grid resolution: K = {Q.shape[1]}, Particles: M = {model.n_particles}")
    print(f"Selected contrast shrinkage lambda*: {model.selected_contrast_shrinkage_:.1f}")
    print("Out-of-fold energy-risk selection table:")
    for rec in model.selection_records_:
        print(
            f"  lambda = {rec.contrast_shrinkage:5.1f} | "
            f"held-out energy risk = {rec.held_out_risk:.5f} (n_scored = {rec.n_scored})"
        )

    plugin = plugin_contrasts(model, X, functionals)

    print("\nMarginal Distributional Treatment Effects (TATE):")
    print(f"{'Functional':<14} {'WCF (DR)':>10} {'IF SE':>10} {'95% CI':>23} {'Plug-in':>10}")
    print("-" * 72)
    for name in ("mean", "sd", "skew", "tail", "reference", "p10", "p90"):
        dr_est = model.dr_marginal(name)
        se = model.dr_if_se(name)
        lo, hi = dr_est - 1.96 * se, dr_est + 1.96 * se
        pi_est = plugin[name]["marginal"]
        print(
            f"{name:<14} {dr_est:+10.3f} {se:10.3f} "
            f"[{lo:+8.3f}, {hi:+8.3f}] {pi_est:+10.3f}"
        )

    print("\nModerator-Quartile Contrasts (TCATE bins Q1..Q4 for 'mean' and 'reference'):")
    for name in ("mean", "reference"):
        bins_dr = model.aipw_.bin_contrasts_[name]
        print(
            f"  {name:<10}: "
            + " | ".join(f"Q{b + 1} = {bins_dr[b]:+.3f}" for b in range(4))
        )

    # Predict conditional particle laws on test units
    particles_0 = model.predict_particles(X[:5], arm=0)  # shape (5, M, K)
    particles_1 = model.predict_particles(X[:5], arm=1)  # shape (5, M, K)
    mean_q0 = model.predict_mean_quantile(X, arm=0).mean(axis=0)
    mean_q1 = model.predict_mean_quantile(X, arm=1).mean(axis=0)
    print(
        f"\nPredicted particle cloud shapes for 5 units: "
        f"arm 0 -> {particles_0.shape}, arm 1 -> {particles_1.shape}"
    )

    # Save a diagnostic plot
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "tutorial_01_quickstart.png"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(u, mean_q0, color="#D55E00", lw=2.0, label="Predicted control barycenter")
    ax.plot(u, mean_q1, color="#0072B2", lw=2.0, label="Predicted treated barycenter")
    ax.plot(u, q_star, color="#000000", lw=1.6, ls="--", label="Reference benchmark $q_\\star^K$")
    ax.set_xlabel("Midpoint quantile level $u_k$")
    ax.set_ylabel("Outcome quantile")
    ax.set_title("(a) Predicted marginal quantile curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    names = ["mean", "sd", "tail", "reference", "p10", "p90"]
    y_pos = np.arange(len(names))
    ests = [model.dr_marginal(k) for k in names]
    ses = [model.dr_if_se(k) for k in names]
    ax.axvline(0.0, color="0.3", ls="--", lw=1.0)
    ax.errorbar(
        ests,
        y_pos,
        xerr=1.96 * np.asarray(ses),
        fmt="o",
        color="#0072B2",
        capsize=4,
        lw=1.6,
        label="WCF (DR) $\\pm 1.96$ IF SE",
    )
    ax.set_yticks(y_pos, names)
    ax.invert_yaxis()
    ax.set_xlabel("Estimated treatment effect")
    ax.set_title("(b) Doubly robust functional contrasts")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)

    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to: {output_file}")


if __name__ == "__main__":
    main()
