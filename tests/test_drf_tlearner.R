#!/usr/bin/env Rscript
#
# WP2-C1 verification for the two-arm Wasserstein DRF baseline (W-DRF-T).
# Run from the repository root in a clean session:
#
#   Rscript tests/test_drf_tlearner.R
#
# Covers the six verification items named in the phase file: exact geometry
# identity, ordinary example reproduction, forest-weight normalization, arm
# separation, deterministic seed behavior, and the common prediction schema.

source("research/baselines/drf_tlearner.R")

failures <- 0L
checks <- 0L

check <- function(label, condition) {
  checks <<- checks + 1L
  if (isTRUE(condition)) {
    cat(sprintf("ok   %s\n", label))
  } else {
    failures <<- failures + 1L
    cat(sprintf("FAIL %s\n", label))
  }
}

# A location-scale cell with heterogeneous treatment effects on both the
# location and the spread, so the arm laws genuinely differ in shape.
location_scale_cell <- function(seed = 11L, n = 300L, p = 4L, n_grid = 9L) {
  set.seed(seed)
  levels <- seq_len(n_grid) / (n_grid + 1L)
  quad_weights <- rep(1 / n_grid, n_grid)
  X <- matrix(runif(n * p), n, p)
  A <- rbinom(n, 1L, 0.5)
  location <- 1 + 2 * X[, 1] + A * (0.8 * X[, 3])
  scale <- (0.5 + X[, 2]) * (1 + 0.5 * A * X[, 4])
  Q <- t(vapply(
    seq_len(n),
    function(i) qnorm(levels, location[i], scale[i]),
    numeric(n_grid)
  ))
  list(
    X = X, A = A, Q = Q, levels = levels, quad_weights = quad_weights,
    location = location, scale = scale
  )
}

# --- provenance and label -------------------------------------------------

environment_report <- wdrft_environment()

check(
  "baseline is labelled W-DRF-T and not Causal-DRF",
  identical(WDRFT_METHOD_ID, "W-DRF-T") &&
    !grepl("causal", WDRFT_METHOD_ID, ignore.case = TRUE)
)
check(
  "pinned GPL-3.0 drf revision is the installed one",
  isTRUE(environment_report$version_matches_pin) &&
    identical(environment_report$license, "GPL-3")
)

# --- exact geometry identity ----------------------------------------------

cell <- location_scale_cell()
identity_report <- check_geometry_identity(cell$Q[1:60, ], cell$quad_weights)

check(
  "squared Euclidean distance on z equals weighted W_2 distance on q",
  identity_report$max_squared_distance_error < 1e-10
)
check(
  "Gaussian kernel on z equals the discretized Wasserstein-RBF kernel on q",
  identity_report$max_kernel_error < 1e-12
)
check(
  "rescaling is invertible",
  max(abs(unscale_quantiles(
    rescale_quantiles(cell$Q, cell$quad_weights), cell$quad_weights
  ) - cell$Q)) < 1e-12
)
check(
  "unequal quadrature weights are honoured",
  {
    uneven <- seq(0.5, 1.5, length.out = ncol(cell$Q))
    uneven <- uneven / sum(uneven)
    report <- check_geometry_identity(cell$Q[1:40, ], uneven)
    report$max_squared_distance_error < 1e-10
  }
)
check(
  "nonpositive quadrature weights are rejected",
  inherits(try(validate_quad_weights(c(0.5, 0.0)), silent = TRUE), "try-error")
)

# --- ordinary DRF example reproduction ------------------------------------

example_report <- wdrft_reproduce_package_example(seed = 7L)

check(
  "package example forest weights sum to one",
  example_report$weight_row_sum_error < 1e-10
)
check(
  "package example forest weights are nonnegative",
  isTRUE(example_report$weights_nonnegative)
)
check(
  "mean read off the weights equals the built-in mean functional",
  example_report$mean_agreement_error < 1e-10
)
check(
  "package example recovers its own E[Y_1 | X] = X_1 signal",
  example_report$mean_signal_correlation > 0.8
)

# --- fit, weights, and arm separation -------------------------------------

fit <- wdrft_fit(
  cell$X, cell$A, cell$Q, cell$quad_weights,
  num_trees = 200L, seed = 3L
)
X_test <- matrix(runif(20L * ncol(cell$X)), 20L, ncol(cell$X))
reference <- qnorm(cell$levels, 0, 1)
functionals <- cbind(mean = rowMeans(cell$Q), sd = apply(cell$Q, 1L, sd))
prediction <- wdrft_predict(
  fit, X_test,
  reference_quantiles = reference, functionals = functionals
)
invariants <- check_prediction_invariants(prediction, fit$treated_index)

check(
  "arm weight rows are normalized",
  max(invariants$row_error_treated, invariants$row_error_control) < 1e-10
)
check(
  "treated weights place no mass on control units and vice versa",
  max(invariants$leak_treated, invariants$leak_control) < 1e-12
)
check(
  "arm forests are fitted on disjoint index sets",
  length(intersect(fit$control_index, fit$treated_index)) == 0L &&
    length(fit$control_index) + length(fit$treated_index) == nrow(cell$X)
)
check(
  "the two arm forests are separate objects with no shared partition",
  isFALSE(prediction$shared_partition) &&
    isFALSE(prediction$treatment_aware_splitting) &&
    !identical(fit$forest_control, fit$forest_treated)
)
check(
  "response scaling is disabled so the W_2 geometry is not overwritten",
  isFALSE(fit$hyperparameters$response_scaling)
)

# --- common prediction schema ---------------------------------------------

required_fields <- c(
  "method_id", "schema_version", "n_test", "n_train", "n_grid",
  "weights_treated", "weights_control", "weights_signed",
  "barycenter_treated", "barycenter_control", "barycenter_contrast",
  "reference_treated", "reference_control", "reference_effect",
  "functional_treated", "functional_control", "functional_effect"
)

check(
  "prediction record exposes every common schema field",
  all(required_fields %in% names(prediction))
)
check(
  "prediction record is versioned and labelled",
  identical(prediction$method_id, "W-DRF-T") &&
    identical(prediction$schema_version, BASELINE_SCHEMA_VERSION)
)
check(
  "conditional barycenters stay monotone without projection",
  all(apply(prediction$barycenter_treated, 1L, function(v) all(diff(v) >= -1e-12))) &&
    all(apply(prediction$barycenter_control, 1L, function(v) all(diff(v) >= -1e-12)))
)
check(
  "signed weights are the arm difference",
  max(abs(
    prediction$weights_signed -
      (prediction$weights_treated - prediction$weights_control)
  )) < 1e-15
)
check(
  "reference levels are nonnegative distances",
  all(prediction$reference_treated >= 0) && all(prediction$reference_control >= 0)
)
check(
  "functional names survive into the record",
  identical(prediction$functional_names, c("mean", "sd"))
)
check(
  "functional effect equals the difference of arm functionals",
  max(abs(
    prediction$functional_effect -
      (prediction$functional_treated - prediction$functional_control)
  )) < 1e-12
)

# --- recovery of the arm laws ---------------------------------------------

check(
  "the scale effect is recovered with the correct sign",
  {
    spread_treated <- prediction$functional_treated[, "sd"]
    spread_control <- prediction$functional_control[, "sd"]
    mean(spread_treated - spread_control) > 0
  }
)
check(
  "the location effect is recovered with the correct sign",
  mean(prediction$functional_effect[, "mean"]) > 0
)
check(
  "a null treatment produces a near-zero contrast",
  {
    null_cell <- location_scale_cell(seed = 5L)
    null_cell$Q <- t(vapply(
      seq_len(nrow(null_cell$X)),
      function(i) {
        qnorm(null_cell$levels, 1 + 2 * null_cell$X[i, 1], 0.5 + null_cell$X[i, 2])
      },
      numeric(ncol(null_cell$Q))
    ))
    null_fit <- wdrft_fit(
      null_cell$X, null_cell$A, null_cell$Q, null_cell$quad_weights,
      num_trees = 200L, seed = 3L
    )
    null_prediction <- wdrft_predict(null_fit, X_test)
    signal <- mean(abs(prediction$barycenter_contrast))
    mean(abs(null_prediction$barycenter_contrast)) < 0.5 * signal
  }
)

# --- deterministic seed behaviour -----------------------------------------

repeat_fit <- wdrft_fit(
  cell$X, cell$A, cell$Q, cell$quad_weights,
  num_trees = 200L, seed = 3L
)
repeat_prediction <- wdrft_predict(
  repeat_fit, X_test,
  reference_quantiles = reference, functionals = functionals
)
other_fit <- wdrft_fit(
  cell$X, cell$A, cell$Q, cell$quad_weights,
  num_trees = 200L, seed = 99L
)
other_prediction <- wdrft_predict(other_fit, X_test)

check(
  "the same seed reproduces the weights exactly",
  max(abs(prediction$weights_signed - repeat_prediction$weights_signed)) == 0
)
check(
  "the same seed reproduces every derived summary exactly",
  max(abs(prediction$reference_effect - repeat_prediction$reference_effect)) == 0
)
check(
  "a different seed changes the fit",
  max(abs(prediction$weights_signed - other_prediction$weights_signed)) > 0
)
check(
  "particle export is deterministic and stays monotone",
  {
    particles <- baseline_particles(prediction, cell$Q, arm = 1L, n_particles = 5L, seed = 2L)
    again <- baseline_particles(prediction, cell$Q, arm = 1L, n_particles = 5L, seed = 2L)
    identical(dim(particles), c(20L, 5L, ncol(cell$Q))) &&
      identical(particles, again) &&
      all(apply(particles, c(1L, 2L), function(v) all(diff(v) >= -1e-12)))
  }
)

# --- input validation ------------------------------------------------------

check(
  "a nonbinary treatment vector is rejected",
  inherits(
    try(
      wdrft_fit(cell$X, rep(2L, nrow(cell$X)), cell$Q, cell$quad_weights),
      silent = TRUE
    ),
    "try-error"
  )
)
check(
  "an empty treatment arm is rejected",
  inherits(
    try(
      wdrft_fit(cell$X, rep(1L, nrow(cell$X)), cell$Q, cell$quad_weights),
      silent = TRUE
    ),
    "try-error"
  )
)
check(
  "nonmonotone quantile rows are rejected",
  {
    broken <- cell$Q
    broken[1L, ] <- rev(broken[1L, ])
    inherits(
      try(wdrft_fit(cell$X, cell$A, broken, cell$quad_weights), silent = TRUE),
      "try-error"
    )
  }
)

cat(sprintf("\n%d checks, %d failures\n", checks, failures))
if (failures > 0L) {
  quit(status = 1L)
}
