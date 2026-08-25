#!/usr/bin/env Rscript
#
# WP2-C1. Two-arm Wasserstein DRF baseline (W-DRF-T).
#
# This is a T-learner: one *ordinary predictive* Distributional Random Forest
# (Cevid, Michel, Naf, Meinshausen, Buhlmann, 2022) is fitted per treatment
# arm on the rescaled quantile coordinates z = diag(sqrt(w)) q, and the two
# forests never share a split. It is the cheapest full-law forest comparator
# and it is *not* Causal-DRF: there is no treatment-aware splitting and no
# signed prediction weight. The label `W-DRF-T` is enforced below and must be
# carried unchanged into every result schema and manuscript table.
#
# The `W-` prefix records the input geometry only. Because
# ||z - z'||_2^2 = W_{2,K}^2(q, q'), the Gaussian kernel the DRF splitting rule
# already uses is exactly a discretized Wasserstein-RBF kernel on q. That is a
# representation map, not a methodological contribution.
#
# Provenance and licensing are recorded in `research/baselines/PROVENANCE.md`.
# The GPL-3.0 `drf` package is used through its public R API only; no DRF
# source is copied into this repository.

WDRFT_METHOD_ID <- "W-DRF-T"
WDRFT_BRIDGE_ID <- "W-DRF-T-BRIDGE-v1"
WDRFT_PACKAGE_SOURCE <- "CRAN::drf"
WDRFT_PINNED_VERSION <- "1.3.1"
WDRFT_PACKAGE_LICENSE <- "GPL-3"

suppressPackageStartupMessages({
  library(drf)
})

#' Resolve a file that sits next to this script, falling back to a
#' repository-root-relative path when the script directory is unknown
#' (`source()` from an arbitrary working directory, `Rscript -e`, ...).
wdrft_sibling_path <- function(filename) {
  script <- NULL
  frames <- sys.frames()
  for (index in rev(seq_along(frames))) {
    candidate <- frames[[index]]$ofile
    if (!is.null(candidate) && nzchar(candidate)) {
      script <- candidate
      break
    }
  }
  if (is.null(script)) {
    arguments <- commandArgs(trailingOnly = FALSE)
    flag <- grep("^--file=", arguments, value = TRUE)
    if (length(flag) == 1L) {
      script <- sub("^--file=", "", flag)
    }
  }
  if (!is.null(script) && nzchar(script)) {
    beside <- file.path(dirname(normalizePath(script, mustWork = FALSE)), filename)
    if (file.exists(beside)) {
      return(beside)
    }
  }
  file.path("research", "baselines", filename)
}

if (!exists("BASELINE_SCHEMA_VERSION")) {
  source(wdrft_sibling_path("baseline_common.R"))
}

# ---------------------------------------------------------------------------
# Pinned-environment reporting
# ---------------------------------------------------------------------------

#' Report the pinned DRF environment used for this baseline.
#'
#' @return Named list with the package source, version, and license.
wdrft_environment <- function() {
  installed <- tryCatch(
    as.character(utils::packageVersion("drf")),
    error = function(e) NA_character_
  )
  list(
    bridge_id = WDRFT_BRIDGE_ID,
    method_id = WDRFT_METHOD_ID,
    package_source = WDRFT_PACKAGE_SOURCE,
    pinned_version = WDRFT_PINNED_VERSION,
    installed_version = installed,
    version_matches_pin = identical(installed, WDRFT_PINNED_VERSION),
    license = WDRFT_PACKAGE_LICENSE,
    r_version = paste(R.version$major, R.version$minor, sep = ".")
  )
}

# ---------------------------------------------------------------------------
# Ordinary-example reproduction (WP2-C1 action 3)
# ---------------------------------------------------------------------------

#' Reproduce the canonical example shipped in `?drf`.
#'
#' The example fits an ordinary predictive DRF on a three-dimensional response
#' whose first coordinate has conditional mean X_1, then reads E[Y_1 | X] off
#' the forest weights. Passing this establishes that the pinned package builds,
#' predicts, and exposes usable weights before any causal wrapper is trusted.
#'
#' @param seed Integer seed.
#' @return List of reproduction diagnostics.
wdrft_reproduce_package_example <- function(seed = 1L) {
  set.seed(seed)
  n <- 500L
  p <- 10L
  d <- 3L
  X <- matrix(rnorm(n * p), nrow = n)
  Y <- matrix(rnorm(n * d), nrow = n)
  Y[, 1] <- Y[, 1] + X[, 1]
  Y[, 2] <- Y[, 2] * X[, 2]
  Y[, 3] <- Y[, 3] * X[, 1] + X[, 2]

  forest <- drf(X, Y, num.trees = 500L, seed = seed)
  X_test <- matrix(rnorm(10L * p), nrow = 10L)

  weighted <- predict(forest, newdata = X_test)
  from_weights <- as.numeric(weighted$weights %*% weighted$y[, 1])
  from_functional <- predict(forest, newdata = X_test, functional = "mean")[, 1]

  list(
    weight_row_sum_error = max(abs(rowSums(as.matrix(weighted$weights)) - 1)),
    weights_nonnegative = min(as.matrix(weighted$weights)) >= 0,
    mean_agreement_error = max(abs(from_weights - from_functional)),
    # E[Y_1 | X] = X_1 in this example, so the fitted mean must track X_test[, 1].
    mean_signal_correlation = as.numeric(cor(from_weights, X_test[, 1])),
    n_test = nrow(X_test)
  )
}

# ---------------------------------------------------------------------------
# W-DRF-T fit
# ---------------------------------------------------------------------------

#' Fit one ordinary DRF per treatment arm on rescaled quantile coordinates.
#'
#' `response.scaling` is forced off. The DRF default standardizes each response
#' column, which would replace the exact discretized W_2 geometry of z by an
#' arbitrary coordinatewise rescaling and make the `W-` prefix false.
#'
#' @param X Numeric n x p covariate matrix.
#' @param A Binary treatment vector of length n.
#' @param Q Numeric n x K matrix of monotone quantile vectors.
#' @param quad_weights Numeric vector of length K of positive quadrature weights.
#' @param num_trees Trees per arm forest.
#' @param min_node_size Minimum leaf size inside each arm forest.
#' @param num_features Random Fourier features used by the MMD split rule.
#' @param mtry Candidate variables per split; defaults to the DRF rule.
#' @param sample_fraction Subsample fraction per tree.
#' @param honesty Whether to use honest splitting.
#' @param seed Integer seed. Arm 0 uses `seed`, arm 1 uses `seed + 1`.
#' @return A `wdrft_fit` object.
wdrft_fit <- function(X,
                      A,
                      Q,
                      quad_weights,
                      num_trees = 1000L,
                      min_node_size = 15L,
                      num_features = 10L,
                      mtry = NULL,
                      sample_fraction = 0.5,
                      honesty = TRUE,
                      seed = 1L) {
  X <- as.matrix(X)
  Q <- validate_quantile_matrix(Q)
  quad_weights <- validate_quad_weights(quad_weights, ncol(Q))
  A <- as.integer(round(as.numeric(A)))

  if (nrow(X) != nrow(Q) || length(A) != nrow(X)) {
    stop("X, A, and Q must describe the same units")
  }
  if (!all(A %in% c(0L, 1L))) {
    stop("A must be binary")
  }
  control_index <- which(A == 0L)
  treated_index <- which(A == 1L)
  if (length(control_index) < 2L || length(treated_index) < 2L) {
    stop("each treatment arm needs at least two units")
  }

  Z <- rescale_quantiles(Q, quad_weights)
  if (is.null(mtry)) {
    mtry <- min(ceiling(sqrt(ncol(X)) + 20), ncol(X))
  }

  fit_arm <- function(index, arm_seed) {
    drf(
      X = X[index, , drop = FALSE],
      Y = Z[index, , drop = FALSE],
      num.trees = num_trees,
      min.node.size = min_node_size,
      num.features = num_features,
      mtry = mtry,
      sample.fraction = sample_fraction,
      honesty = honesty,
      response.scaling = FALSE,
      # The drf default is as.integer(num.trees / 30), which is zero for fewer
      # than thirty trees and makes its C++ core divide by zero (SIGFPE).
      ci.group.size = max(1L, as.integer(num_trees / 30L)),
      seed = arm_seed
    )
  }

  structure(
    list(
      method_id = WDRFT_METHOD_ID,
      bridge_id = WDRFT_BRIDGE_ID,
      forest_control = fit_arm(control_index, as.integer(seed)),
      forest_treated = fit_arm(treated_index, as.integer(seed) + 1L),
      control_index = control_index,
      treated_index = treated_index,
      Q_train = Q,
      quad_weights = quad_weights,
      n_train = nrow(Q),
      n_grid = ncol(Q),
      seed = as.integer(seed),
      hyperparameters = list(
        num_trees = num_trees,
        min_node_size = min_node_size,
        num_features = num_features,
        mtry = mtry,
        sample_fraction = sample_fraction,
        honesty = honesty,
        response_scaling = FALSE
      )
    ),
    class = "wdrft_fit"
  )
}

#' Expand an arm forest's weights onto the full training index space.
#'
#' @param forest A fitted `drf` object.
#' @param X_test Numeric n_test x p matrix.
#' @param index Integer training rows the forest was fitted on.
#' @param n_train Full training sample size.
#' @return Numeric n_test x n_train matrix with rows summing to 1.
wdrft_arm_weights <- function(forest, X_test, index, n_train) {
  X_test <- as.matrix(X_test)
  arm_weights <- as.matrix(predict(forest, newdata = X_test)$weights)
  full <- matrix(0.0, nrow = nrow(X_test), ncol = n_train)
  full[, index] <- arm_weights
  full
}

#' Predict arm laws and law-invariant summaries in the common schema.
#'
#' @param fit A `wdrft_fit` object.
#' @param X_test Numeric n_test x p matrix.
#' @param reference_quantiles Optional length-K quantile vector of nu_star.
#' @param functionals Optional n_train x J matrix of T_j(Y_i) values.
#' @return A `wp2c_prediction` list.
wdrft_predict <- function(fit,
                          X_test,
                          reference_quantiles = NULL,
                          functionals = NULL) {
  stopifnot(inherits(fit, "wdrft_fit"))
  X_test <- as.matrix(X_test)

  weights_control <- wdrft_arm_weights(
    fit$forest_control, X_test, fit$control_index, fit$n_train
  )
  weights_treated <- wdrft_arm_weights(
    fit$forest_treated, X_test, fit$treated_index, fit$n_train
  )

  baseline_prediction(
    method_id = WDRFT_METHOD_ID,
    weights_treated = weights_treated,
    weights_control = weights_control,
    Q_train = fit$Q_train,
    quad_weights = fit$quad_weights,
    reference_quantiles = reference_quantiles,
    functionals = functionals,
    extra = list(
      shared_partition = FALSE,
      treatment_aware_splitting = FALSE,
      seed = fit$seed,
      hyperparameters = fit$hyperparameters
    )
  )
}

#' Conditional witness function of the arm contrast at one test point.
#'
#' The witness is tau(y) = sum_i (w_i^1 - w_i^0) k(Y_i, y) with the Gaussian
#' kernel on rescaled coordinates. W-DRF-T forms it from two separately fitted
#' forests; Causal-DRF forms the same object from one shared forest. Having it
#' on both sides is what makes the WP2-C1 versus WP2-C2 comparison possible.
#'
#' @param record A `wp2c_prediction` list.
#' @param Z_train Numeric n_train x d matrix of training responses.
#' @param Z_eval Numeric n_eval x d matrix of evaluation points.
#' @param sigma Gaussian bandwidth.
#' @return Numeric n_test x n_eval matrix.
baseline_witness <- function(record, Z_train, Z_eval, sigma) {
  kernel <- gaussian_kernel_matrix(as.matrix(Z_train), as.matrix(Z_eval), sigma)
  record$weights_signed %*% kernel
}
