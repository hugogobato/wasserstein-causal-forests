"""Method adapters for Phase 6.5.

Three families, one interface each inherited from the frozen adapter contract:

* ``LogForestAdapter`` fits an incumbent forest on ``log Y`` and maps back
  through ``exp``. Because the map is monotone, the atom bank after mapping is
  byte-for-byte the training sample's own grid vectors, so the control changes
  exactly one thing, the kernel and splitting geometry, and nothing else. It is
  the honest answer to "every applied researcher would have taken logs".
* ``RetunedCausalDRFAdapter`` runs the incumbent through a driver that accepts
  an explicit kernel bandwidth multiplier. The multiplier is selected per
  (regime, n) cell on held-out energy score with pilot seeds outside every
  manifest, and the frozen selection file must exist before any decisive cell;
  a missing file fails loudly rather than silently running at unity.
* ``ZIPTAdapter`` is the two-part assembly: cross-fitted per-arm classifiers
  for the degenerate component, the R3 booster refitted on positive rows only,
  and the law reassembled as ``(1 - phat_a(x)) delta_0 + phat_a(x) * cloud``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..cwdb.cross_fitted import CrossFittedCWDBRegressor
from . import r_bridge
from .dgps import DGPSample, DistributionalDGP
from .laws import LawPrediction, energy_risk_against_truth
from .methods import MethodOutput, _output_from_laws, peak_ram_mb
from .phase6 import PHASE6_CONTRAST_CANDIDATES, PHASE6_SELECTION_FOLDS

#: Frozen candidate multipliers for the bandwidth-retune control. Selection may
#: only pick from this grid, never tune around it.
BANDWIDTH_CANDIDATES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

#: Pilot seeds for the selection rule. They live outside every manifest seed
#: range (decisive seeds are 0-9), so selecting on them touches no decisive row.
SELECTION_SEEDS: tuple[int, ...] = (100, 101)

#: Location of the frozen selection document, relative to the repository root.
#: The Colab notebooks generate this file in their pilot cell before any
#: decisive cell runs; local runs expect the committed file.
SELECTION_RELATIVE_PATH = Path("results") / "manifests" / \
    "phase65_bandwidth_selection.json"

DEGENERATE_TOLERANCE = 1e-12


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_bandwidth_selection() -> dict[str, float]:
    """The frozen multipliers, keyed by ``"{dgp}|{n_train}"``."""

    path = repository_root() / SELECTION_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the bandwidth-selection pilot "
            "(seeds outside the manifest) before any decisive retune cell."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value)
            for key, value in document["multipliers"].items()}


def select_bandwidth_multiplier(
    dgp: DistributionalDGP,
    n_train: int,
    *,
    seeds: tuple[int, ...] = SELECTION_SEEDS,
    candidates: tuple[float, ...] = BANDWIDTH_CANDIDATES,
    cache_directory: Path | None = None,
) -> tuple[float, dict[str, float]]:
    """Pick the retune multiplier on held-out energy score.

    For each pilot seed the sample is split arm-stratified into a fitting half
    and a scoring half. One forest is fitted per candidate on the fitting half,
    and the candidate's score is the mean energy risk of its arm-law estimate
    against the held-out rows' realised outcome distributions, a proper score,
    so the rule cannot be gamed by oracle quantities. Oracle truth is never
    read.

    The multiplier enters as exact outcome rescaling: with
    ``response.scaling = FALSE`` and the data-driven bandwidth, every kernel
    quantity of the fit on ``Q / m`` equals the corresponding quantity of a fit
    at bandwidth ``m`` times the default, so the untouched published driver
    implements the whole candidate grid without modification.
    """

    scores: dict[float, list[float]] = {c: [] for c in candidates}
    for seed in seeds:
        sample = dgp.sample(n_train, seed=seed)
        rng = np.random.default_rng(seed + 500_000)
        fold = rng.random(sample.n_rows)
        validation = fold >= 0.5
        for candidate in candidates:
            result = r_bridge.fit_predict(
                "causal_drf",
                X_train=sample.X[~validation],
                treatment=sample.treatment[~validation],
                Q_train=sample.quantiles[~validation] / candidate,
                X_test=sample.X[validation],
                quad_weights=dgp.grid.weights,
                reference_quantiles=dgp.grid.reference_quantiles(),
                hyperparameters={},
                seed=seed,
                cache_directory=cache_directory,
            )
            total = 0.0
            for arm, matrix in result.weights.items():
                law = LawPrediction.from_forest_weights(
                    sample.quantiles[~validation] / candidate, matrix
                )
                realised = (
                    sample.quantiles[validation][:, None, :] / candidate
                )
                total += float(np.mean(energy_risk_against_truth(
                    law, realised, np.ones(int(validation.sum())),
                    dgp.grid.weights, epsilon=1e-3,
                )))
            scores[candidate].append(total / 2.0)
    means = {c: float(np.mean(v)) for c, v in scores.items()}
    best = min(means, key=lambda c: means[c])
    return best, means


class LogForestAdapter:
    """Incumbent forests fitted under a shifted-log geometry.

    Income-like panels contain non-positive coordinates, so the control fits
    on ``log(Q - floor)`` with the frozen rule

        floor = min(Q_train) - 0.05 * sd(Q_train),

    and maps every predicted atom back through the exact inverse,
    ``exp(.) + floor``. Because the composite map is strictly monotone, the
    mapped bank is byte-for-byte the original-scale training sample: the
    control changes exactly one thing, the geometry the forest sees during
    splitting and weighting, and nothing else. Every metric is computed
    against original-scale truth.
    """

    produces_law = True

    def __init__(
        self,
        method: str,
        *,
        cache_directory: Path | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        if method not in {"causal_drf", "drf"}:
            raise ValueError("method must be 'causal_drf' or 'drf'")
        self.method = method
        self.cache_directory = cache_directory
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _floor(quantiles: NDArray[np.float64]) -> float:
        return float(np.min(quantiles)) - 0.05 * float(np.std(quantiles))

    def fit_predict(
        self,
        train: DGPSample,
        X_test: NDArray[np.float64],
        dgp: DistributionalDGP,
        functionals: tuple[str, ...],
        *,
        seed: int,
    ) -> MethodOutput:
        floor = self._floor(train.quantiles)
        before = peak_ram_mb()
        result = r_bridge.fit_predict(
            self.method,
            X_train=train.X,
            treatment=train.treatment,
            Q_train=np.log(train.quantiles - floor),
            X_test=X_test,
            quad_weights=dgp.grid.weights,
            reference_quantiles=dgp.grid.reference_quantiles(),
            hyperparameters={},
            seed=seed,
            cache_directory=self.cache_directory,
            timeout_seconds=self.timeout_seconds,
        )
        # exp maps the log-space bank back onto exactly the original vectors,
        # so the weights land on the untouched atom bank.
        laws = {
            arm: LawPrediction.from_forest_weights(train.quantiles, matrix)
            for arm, matrix in result.weights.items()
        }
        output = _output_from_laws(
            laws,
            dgp.grid.weights,
            dgp.grid.reference_quantiles(),
            functionals,
            fit_seconds=result.fit_seconds,
            predict_seconds=max(result.total_seconds - result.fit_seconds, 0.0),
            peak_ram_mb=max(result.peak_ram_mb,
                            max(peak_ram_mb() - before, 0.0)),
        )
        diagnostics = {
            **output.diagnostics,
            "log_floor": float(floor),
        }
        object.__setattr__(output, "diagnostics", diagnostics)
        return output


class RetunedCausalDRFAdapter:
    """Causal-DRF at a frozen, held-out-selected kernel bandwidth multiplier.

    The multiplier enters as exact outcome rescaling. With
    ``response.scaling = FALSE`` and the data-driven bandwidth, fitting on
    ``Q / m`` reproduces, kernel value for kernel value, the fit that would
    have run at bandwidth ``m`` times the default; the arm weights returned by
    the untouched published driver therefore apply to the original-scale atom
    bank directly. Everything except the effective bandwidth follows the
    authors' call path.
    """

    produces_law = True

    def __init__(
        self,
        *,
        cache_directory: Path | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.cache_directory = cache_directory
        self.timeout_seconds = timeout_seconds

    def fit_predict(
        self,
        train: DGPSample,
        X_test: NDArray[np.float64],
        dgp: DistributionalDGP,
        functionals: tuple[str, ...],
        *,
        seed: int,
    ) -> MethodOutput:
        key = f"{dgp.spec.dgp_id}|{train.n_rows}"
        multiplier = load_bandwidth_selection().get(key)
        if multiplier is None:
            raise KeyError(
                f"no frozen bandwidth multiplier for {key!r}; the selection "
                "document covers the cells that ran its pilot"
            )
        before = peak_ram_mb()
        result = r_bridge.fit_predict(
            "causal_drf",
            X_train=train.X,
            treatment=train.treatment,
            Q_train=train.quantiles / multiplier,
            X_test=X_test,
            quad_weights=dgp.grid.weights,
            reference_quantiles=dgp.grid.reference_quantiles(),
            hyperparameters={},
            seed=seed,
            cache_directory=self.cache_directory,
            timeout_seconds=self.timeout_seconds,
        )
        laws = {
            arm: LawPrediction.from_forest_weights(train.quantiles, matrix)
            for arm, matrix in result.weights.items()
        }
        output = _output_from_laws(
            laws,
            dgp.grid.weights,
            dgp.grid.reference_quantiles(),
            functionals,
            fit_seconds=result.fit_seconds,
            predict_seconds=max(result.total_seconds - result.fit_seconds, 0.0),
            peak_ram_mb=max(result.peak_ram_mb,
                            max(peak_ram_mb() - before, 0.0)),
        )
        diagnostics = {
            **output.diagnostics,
            "bandwidth_multiplier": float(multiplier),
        }
        object.__setattr__(output, "diagnostics", diagnostics)
        return output


class ZIPTAdapter:
    """``cwdb_zipt``: two-part assembly over a degenerate component.

    Part one is a per-arm classifier for ``P(degenerate | X, arm)``. Part two
    is the cross-fitted R3 booster refitted on positive-component rows only.
    The assembled conditional law puts mass ``1 - phat_a(x)`` on the zero grid
    vector and spreads ``phat_a(x)`` uniformly over the predicted particles,
    so every downstream quantity, including the unseen functionals and the
    reference target, integrates against the mixture rather than against
    either part alone. This is a mixture assembly in law space, not a
    relabelling of posterior draws.
    """

    produces_law = True

    def __init__(
        self,
        *,
        n_particles: int = 10,
        classifier_c: float = 1.0,
        **budget: object,
    ) -> None:
        # The frozen registry forwards the selection settings alongside the
        # boosting budget; lift them out so the booster constructor receives
        # each keyword exactly once.
        self.n_folds = int(budget.pop("n_folds", PHASE6_SELECTION_FOLDS))
        candidates = budget.pop(
            "contrast_candidates", PHASE6_CONTRAST_CANDIDATES
        )
        self.contrast_candidates = tuple(candidates)
        self.n_particles = n_particles
        self.classifier_c = classifier_c
        self.budget = dict(budget)

    def _fit_classifier(self, features, targets):
        from sklearn.linear_model import LogisticRegression

        degenerate = np.max(np.abs(targets), axis=1) <= DEGENERATE_TOLERANCE
        rate = float(np.mean(degenerate))
        if rate <= 0.0 or rate >= 1.0:
            # A degenerate arm carries no covariate information about the
            # component; the honest fallback is the constant empirical rate,
            # recorded so no reader mistakes it for a fitted surface.
            constant = True
            surface = None
        else:
            surface = LogisticRegression(
                C=self.classifier_c, solver="lbfgs", max_iter=2000,
            )
            surface.fit(features, degenerate.astype(int))
            constant = False
        return surface, constant, rate

    def _predict_probability(self, model, constant, rate, X):
        """P(degenerate | X): the fitted surface, or the constant rate."""

        if constant or model is None:
            return np.full(X.shape[0], float(np.clip(rate, 0.0, 1.0)))
        return model.predict_proba(X)[:, 1]

    def fit_predict(
        self,
        train: DGPSample,
        X_test: NDArray[np.float64],
        dgp: DistributionalDGP,
        functionals: tuple[str, ...],
        *,
        seed: int,
    ) -> MethodOutput:
        weights = dgp.grid.weights
        before = peak_ram_mb()

        classifiers: dict[int, object] = {}
        constants: dict[int, bool] = {}
        rates: dict[int, float] = {}
        positive_any = np.zeros(train.n_rows, dtype=bool)
        for arm in (0, 1):
            mask = train.treatment == arm
            classifiers[arm], constants[arm], rates[arm] = (
                self._fit_classifier(train.X[mask], train.quantiles[mask])
            )
            positive_any |= mask & (
                np.max(np.abs(train.quantiles), axis=1) > DEGENERATE_TOLERANCE
            )

        started = time.perf_counter()
        model = None
        if int(positive_any.sum()) >= 8:
            model = CrossFittedCWDBRegressor(
                contrast_candidates=self.contrast_candidates,
                n_folds=self.n_folds,
                architecture="v1",
                sharing="partial",
                init_sharing="pooled",
                arm_shrinkage=5.0,
                n_particles=self.n_particles,
                random_state=seed,
                **self.budget,
            )
            model.fit(
                train.X[positive_any],
                train.treatment[positive_any],
                train.quantiles[positive_any],
                weights,
            )
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        laws: dict[int, LawPrediction] = {}
        probability_mean: dict[int, float] = {}
        for arm in (0, 1):
            # The classifier models P(degenerate | X, arm) directly.
            degenerate_probability = self._predict_probability(
                classifiers[arm], constants[arm], rates[arm], X_test
            )
            degenerate_probability = np.clip(
                degenerate_probability, 1e-9, 1.0 - 1e-9
            )
            probability_mean[arm] = float(np.mean(1.0 - degenerate_probability))
            if model is None:
                particles = np.zeros((X_test.shape[0], 1, dgp.grid.n_grid))
            else:
                particles = model.predict_particles(X_test, arm)
            atoms = np.concatenate(
                [np.zeros((X_test.shape[0], 1, dgp.grid.n_grid)), particles],
                axis=1,
            )
            row = np.concatenate(
                [
                    degenerate_probability[:, None],
                    np.repeat(
                        (1.0 - degenerate_probability)[:, None]
                        / particles.shape[1],
                        particles.shape[1],
                        axis=1,
                    ),
                ],
                axis=1,
            )
            laws[arm] = LawPrediction(
                atoms=atoms,
                weights=row / row.sum(axis=1, keepdims=True),
                shared_atoms=False,
            )
        predict_seconds = time.perf_counter() - started

        output = _output_from_laws(
            laws,
            weights,
            dgp.grid.reference_quantiles(),
            functionals,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_ram_mb=max(peak_ram_mb() - before, 0.0),
        )
        diagnostics = {
            **output.diagnostics,
            "classifier_constant_arm0": float(constants[0]),
            "classifier_constant_arm1": float(constants[1]),
            "positive_rate_arm0": probability_mean[0],
            "positive_rate_arm1": probability_mean[1],
        }
        if model is not None:
            diagnostics.update({
                "train_risk": float(model.train_risk_),
                "selected_contrast_shrinkage": float(
                    model.selected_contrast_shrinkage_
                ),
            })
        object.__setattr__(output, "diagnostics", diagnostics)
        return output
