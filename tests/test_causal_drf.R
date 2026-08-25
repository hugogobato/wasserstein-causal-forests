#!/usr/bin/env Rscript
#
# WP2-C2 verification for the faithful shared Causal-DRF reimplementation.
# Run from the repository root in a clean session:
#
#   Rscript tests/test_causal_drf.R
#
# Covers the verification items named in the phase file: split-score hand case,
# arm-count constraints, honesty disjointness, null signed weights, and the
# W-DRF-T versus shared-forest comparison. The published-DGP reproduction is a
# separate long-running artefact, `results/smoke/causal_drf_reproduction.parquet`;
# this file checks the pieces the reproduction depends on.

source("research/baselines/causal_drf_r/reproduction.R")

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

# --- provenance and labelling ---------------------------------------------

environment_report <- causal_drf_environment()

check(
  "the implementation declares itself a reimplementation with no author code",
  isTRUE(environment_report$reimplementation) &&
    identical(environment_report$author_code, "none published")
)
check(
  "the implementation does not share source with the GPL-3.0 drf package",
  isFALSE(environment_report$shares_source_with_drf_package)
)
check(
  "Causal-DRF and W-DRF-T carry distinct method labels",
  !identical(CAUSAL_DRF_METHOD_ID, WDRFT_METHOD_ID)
)

# --- split-score hand case -------------------------------------------------
#
# Four units, one treated and one control on each side, so every arm count is
# exactly one and the criterion collapses to a closed form:
#
#   nu_L = (+1 on the treated left unit, -1 on the control left unit)
#   scale = |I_L| |I_R| / (|I_L| + |I_R|)^2 = 2 * 2 / 16 = 1/4
#   value = (1/4) * || (k(y1) - k(y2)) - (k(y3) - k(y4)) ||_H^2

hand_Y <- matrix(c(0.0, 1.0, 0.0, 3.0), ncol = 1L)
hand_W <- c(1L, 0L, 1L, 0L)
hand_left <- c(TRUE, TRUE, FALSE, FALSE)
hand_sigma <- 1.0

hand_expected <- local({
  gram <- gaussian_kernel_matrix(hand_Y, hand_Y, hand_sigma)
  contrast <- c(1, -1, -1, 1)
  0.25 * as.numeric(t(contrast) %*% gram %*% contrast)
})

check(
  "reference split criterion matches the closed-form hand case",
  abs(causal_drf_reference_criterion(hand_Y, hand_W, hand_left, hand_sigma) -
        hand_expected) < 1e-12
)
check(
  "the split criterion is zero when the two sides carry the same contrast",
  {
    mirrored_Y <- matrix(c(0.0, 1.0, 0.0, 1.0), ncol = 1L)
    abs(causal_drf_reference_criterion(mirrored_Y, hand_W, hand_left, hand_sigma)) < 1e-12
  }
)
check(
  "the split criterion vanishes when a child has only one arm",
  {
    one_arm <- c(1L, 1L, 1L, 0L)
    causal_drf_reference_criterion(hand_Y, one_arm, hand_left, hand_sigma) == 0
  }
)
check(
  "the Fourier criterion converges to the exact-kernel criterion",
  {
    set.seed(21)
    n <- 200L
    Y <- matrix(rnorm(n), ncol = 1L)
    W <- rep(c(0L, 1L), length.out = n)
    left <- seq_len(n) <= n / 2
    sigma <- median_heuristic_bandwidth(Y, seed = 1L)
    features <- causal_drf_fourier_features(Y, 4000L, sigma)
    exact <- causal_drf_reference_criterion(Y, W, left, sigma)
    approximate <- causal_drf_fourier_criterion(features$Phi, W, left)
    abs(exact - approximate) < 0.02 * max(exact, 1e-6) + 1e-4
  }
)
check(
  "the compiled search returns the argmax of the criterion it approximates",
  {
    set.seed(4)
    n <- 200L
    n_covariates <- 3L
    X <- matrix(runif(n * n_covariates), n, n_covariates)
    A <- rbinom(n, 1L, 0.5)
    Y <- matrix(rnorm(n, (A - 0.5) * (X[, 1] > 0.5) * 3, 1), ncol = 1L)
    sigma <- median_heuristic_bandwidth(Y, seed = 1L)
    set.seed(9)
    features <- causal_drf_fourier_features(Y, 10L, sigma)
    rows <- as.integer(seq_len(n) - 1L)
    set.seed(5)
    tree <- causal_drf_grow_tree(
      X, A, features$Phi, rows, rows,
      mtry = n_covariates, min_arm_build = 2L, min_arm_pop = 2L,
      alpha = 0.05, max_depth = 1L
    )
    best_variable <- -1L
    best_value <- NA_real_
    best_criterion <- -1
    min_side <- max(1, ceiling(0.05 * n))
    for (variable in seq_len(n_covariates)) {
      for (candidate in sort(unique(X[, variable]))) {
        left <- X[, variable] <= candidate
        if (sum(left) < min_side || sum(!left) < min_side) next
        arm_counts <- c(
          sum(left & A == 1L), sum(left & A == 0L),
          sum(!left & A == 1L), sum(!left & A == 0L)
        )
        if (min(arm_counts) < 2L) next
        value <- causal_drf_fourier_criterion(features$Phi, A, left)
        if (value > best_criterion) {
          best_criterion <- value
          best_variable <- variable
          best_value <- candidate
        }
      }
    }
    tree$split_var[1L] + 1L == best_variable &&
      abs(tree$split_value[1L] - best_value) < 1e-12
  }
)
check(
  "the criterion rewards a split that separates the treatment effect",
  {
    set.seed(12)
    n <- 200L
    Y <- matrix(0.0, n, 1L)
    W <- rep(c(0L, 1L), length.out = n)
    informative <- seq_len(n) <= n / 2
    # Left half: a real arm contrast. Right half: none.
    Y[informative & W == 1L] <- 3
    sigma <- 1.0
    informative_value <- causal_drf_reference_criterion(Y, W, informative, sigma)
    set.seed(13)
    uninformative <- sample(c(TRUE, FALSE), n, replace = TRUE)
    uninformative_value <- causal_drf_reference_criterion(Y, W, uninformative, sigma)
    informative_value > 5 * uninformative_value
  }
)

# --- fitted forest: arm counts, honesty, invariants ------------------------

cell <- causal_drf_paper_dgp(n = 400L, regime = 3L, seed = 42L)
fit <- causal_drf_fit(
  cell$X, cell$A, cell$Y,
  num_trees = 200L, num_groups = 20L, min_arm_leaf = 5L, seed = 7L
)
X_test <- matrix(CAUSAL_DRF_TEST_POINT, nrow = 1L)

leaf_arm_counts <- function(fit) {
  counts <- list(treated = integer(0), control = integer(0))
  for (tree in fit$trees) {
    leaves <- which(tree$split_var < 0L)
    for (node in leaves) {
      members <- tree$leaf_members[
        seq_len(tree$leaf_size[node]) + tree$leaf_start[node]
      ] + 1L
      if (length(members) == 0L) next
      counts$treated <- c(counts$treated, sum(fit$A_train[members] == 1L))
      counts$control <- c(counts$control, sum(fit$A_train[members] == 0L))
    }
  }
  counts
}

arm_counts <- leaf_arm_counts(fit)

check(
  "every populated leaf holds at least min_arm_leaf units of each arm",
  min(arm_counts$treated) >= fit$hyperparameters$min_arm_leaf &&
    min(arm_counts$control) >= fit$hyperparameters$min_arm_leaf
)
check(
  "alpha-regularity is enforced within the paper's bound",
  fit$hyperparameters$alpha > 0 && fit$hyperparameters$alpha <= 0.2
)
check(
  "an alpha above the assumption bound is rejected",
  inherits(
    try(
      causal_drf_fit(cell$X, cell$A, cell$Y, num_trees = 10L, num_groups = 2L, alpha = 0.4),
      silent = TRUE
    ),
    "try-error"
  )
)
check(
  "honesty keeps the split-determining and leaf-populating rows disjoint",
  {
    # A leaf holds only rows from the populating half, so the union of a tree's
    # leaf members must miss every row used to determine that tree's splits.
    set.seed(31)
    subsample <- sample.int(nrow(cell$X), 200L)
    build_rows <- subsample[1:100]
    pop_rows <- subsample[101:200]
    sigma <- median_heuristic_bandwidth(cell$Y, seed = 1L)
    features <- causal_drf_fourier_features(cell$Y, 10L, sigma)
    tree <- causal_drf_grow_tree(
      cell$X, cell$A, features$Phi,
      as.integer(build_rows - 1L), as.integer(pop_rows - 1L),
      mtry = 5L, min_arm_build = 2L, min_arm_pop = 5L,
      alpha = 0.05, max_depth = 30L
    )
    members <- sort(tree$leaf_members + 1L)
    identical(members, sort(pop_rows)) &&
      length(intersect(members, build_rows)) == 0L
  }
)
check(
  "subforest grouping produces the requested number of groups",
  length(unique(fit$tree_group)) == fit$num_groups &&
    length(fit$trees) == fit$num_groups * fit$trees_per_group
)
check(
  "every tree lands in exactly one group with an equal share of trees",
  {
    sizes <- table(fit$tree_group)
    length(unique(as.integer(sizes))) == 1L
  }
)

weights <- causal_drf_weights(fit, X_test, with_groups = TRUE)

check(
  "signed weights split into two normalized arm laws",
  max(abs(rowSums(weights$treated) - 1)) < 1e-10 &&
    max(abs(rowSums(weights$control) - 1)) < 1e-10
)
check(
  "each arm weight vector is supported on its own arm",
  max(abs(weights$treated[, cell$A == 0L])) == 0 &&
    max(abs(weights$control[, cell$A == 1L])) == 0
)
check(
  "arm weights are nonnegative before signing",
  min(weights$treated) >= 0 && min(weights$control) >= 0
)
check(
  "every subforest yields normalized arm weights of its own",
  {
    grouped <- array(weights$grouped_treated, dim = c(fit$num_groups, 1L, nrow(cell$X)))
    max(abs(apply(grouped, 1L, sum) - 1)) < 1e-10
  }
)
check(
  "the pooled weights are the average of the subforest weights",
  {
    grouped <- array(weights$grouped_treated, dim = c(fit$num_groups, 1L, nrow(cell$X)))
    pooled <- apply(grouped, 3L, mean)
    max(abs(pooled - weights$treated[1L, ])) < 1e-10
  }
)
check(
  "the fit is deterministic in its seed",
  {
    again <- causal_drf_fit(
      cell$X, cell$A, cell$Y,
      num_trees = 200L, num_groups = 20L, min_arm_leaf = 5L, seed = 7L
    )
    identical(causal_drf_weights(again, X_test), causal_drf_weights(fit, X_test))
  }
)
check(
  "a different seed changes the fit",
  {
    other <- causal_drf_fit(
      cell$X, cell$A, cell$Y,
      num_trees = 200L, num_groups = 20L, min_arm_leaf = 5L, seed = 8L
    )
    !identical(causal_drf_weights(other, X_test), causal_drf_weights(fit, X_test))
  }
)

# --- common prediction schema ---------------------------------------------

grid_levels <- seq_len(5L) / 6
quad_weights <- rep(1 / 5, 5L)
Q_train <- t(vapply(
  seq_len(nrow(cell$X)),
  function(i) qnorm(grid_levels, cell$Y[i, 1L], 1),
  numeric(5L)
))
prediction <- causal_drf_predict(
  fit, X_test, Q_train, quad_weights,
  reference_quantiles = qnorm(grid_levels, 0, 1),
  functionals = cbind(mean = rowMeans(Q_train))
)
invariants <- check_prediction_invariants(prediction, which(cell$A == 1L))

check(
  "Causal-DRF emits the same schema version as W-DRF-T",
  identical(prediction$schema_version, BASELINE_SCHEMA_VERSION) &&
    identical(prediction$method_id, CAUSAL_DRF_METHOD_ID)
)
check(
  "the record declares its shared, treatment-aware partition",
  isTRUE(prediction$shared_partition) && isTRUE(prediction$treatment_aware_splitting)
)
check(
  "the prediction record passes the shared arm invariants",
  max(invariants$row_error_treated, invariants$row_error_control) < 1e-10 &&
    max(invariants$leak_treated, invariants$leak_control) < 1e-12
)
check(
  "every tree contributes to the test point",
  all(prediction$contributing_trees == length(fit$trees))
)

# --- CKTE, null behaviour, and inference -----------------------------------

null_cell <- causal_drf_paper_dgp(n = 400L, regime = 1L, seed = 43L)
null_fit <- causal_drf_fit(
  null_cell$X, null_cell$A, null_cell$Y,
  num_trees = 500L, num_groups = 25L, seed = 7L
)
null_ckte <- causal_drf_ckte(null_fit, X_test)
effect_ckte <- causal_drf_ckte(fit, X_test)

check(
  "the CKTE norm is far smaller under the null regime than under an effect",
  null_ckte$squared_norm < 0.2 * effect_ckte$squared_norm
)
check(
  "the witness function is near zero under the null regime",
  {
    evaluation <- matrix(seq(-3, 3, length.out = 25L), ncol = 1L)
    witness <- causal_drf_ckte(null_fit, X_test, Y_eval = evaluation)$witness
    max(abs(witness)) < 0.15
  }
)
check(
  "the witness function has the sign of the effect under regime 3",
  {
    # Regime 3 shifts the treated arm upward at the test point, so the witness
    # must be positive above the common centre and negative below it.
    evaluation <- matrix(c(-2, 2), ncol = 1L)
    witness <- causal_drf_ckte(fit, X_test, Y_eval = evaluation)$witness
    witness[1L, 1L] < 0 && witness[1L, 2L] > 0
  }
)
check(
  "the resampling test rejects under a treatment effect",
  {
    inference <- causal_drf_inference(fit, X_test, level = 0.05)
    isTRUE(inference$reject[1L]) && inference$critical_value[1L] > 0
  }
)
check(
  "the resampling test does not reject under the null regime",
  {
    inference <- causal_drf_inference(null_fit, X_test, level = 0.05)
    isFALSE(as.logical(inference$reject[1L]))
  }
)
check(
  "the confidence band is centred on the witness and has positive width",
  {
    evaluation <- matrix(seq(-3, 3, length.out = 11L), ncol = 1L)
    inference <- causal_drf_inference(fit, X_test, Y_eval = evaluation, level = 0.05)
    centre <- (inference$band_lower + inference$band_upper) / 2
    max(abs(centre - inference$witness)) < 1e-12 &&
      all(inference$band_upper > inference$band_lower)
  }
)
check(
  "a smaller nominal level widens the band",
  {
    evaluation <- matrix(seq(-3, 3, length.out = 11L), ncol = 1L)
    wide <- causal_drf_inference(fit, X_test, Y_eval = evaluation, level = 0.01)
    narrow <- causal_drf_inference(fit, X_test, Y_eval = evaluation, level = 0.10)
    wide$critical_value[1L] >= narrow$critical_value[1L]
  }
)

# --- published data generating process and its truth -----------------------

check(
  "the published regimes carry the declared confounding and effect structure",
  {
    regimes <- lapply(1:4, function(r) causal_drf_paper_dgp(600L, r, seed = 5L))
    identical(vapply(regimes, function(d) d$confounded, logical(1L)),
              c(FALSE, TRUE, FALSE, TRUE)) &&
      identical(vapply(regimes, function(d) d$has_effect, logical(1L)),
                c(FALSE, FALSE, TRUE, TRUE))
  }
)
check(
  "the unconfounded regimes have constant propensity one half",
  {
    unconfounded <- causal_drf_paper_dgp(500L, 1L, seed = 5L)
    max(abs(unconfounded$propensity - 0.5)) < 1e-12
  }
)
check(
  "the confounded regimes keep the propensity strictly inside (0, 1)",
  {
    confounded <- causal_drf_paper_dgp(2000L, 2L, seed = 5L)
    min(confounded$propensity) > 0 && max(confounded$propensity) < 1
  }
)
check(
  "the published DGP is deterministic in its seed",
  identical(
    causal_drf_paper_dgp(300L, 4L, seed = 17L)$Y,
    causal_drf_paper_dgp(300L, 4L, seed = 17L)$Y
  )
)
check(
  "the Monte Carlo truth matches the closed-form witness",
  {
    evaluation <- matrix(seq(-5, 5, length.out = 41L), ncol = 1L)
    sigma <- 1.6
    approximate <- causal_drf_true_witness(3L, evaluation, sigma, n_draws = 200000L, seed = 1L)
    exact <- causal_drf_true_witness_exact(3L, evaluation, sigma)
    max(abs(approximate - exact)) < 0.005
  }
)
check(
  "the true witness is identically zero in the null regimes",
  {
    evaluation <- matrix(seq(-5, 5, length.out = 41L), ncol = 1L)
    max(abs(causal_drf_true_witness_exact(1L, evaluation, 1.6))) == 0 &&
      max(abs(causal_drf_true_witness_exact(2L, evaluation, 1.6))) == 0
  }
)

# --- bandwidth conventions -------------------------------------------------

check(
  "the stated median-distance rule is scale equivariant",
  {
    set.seed(2)
    Y <- matrix(rnorm(400), ncol = 2L)
    base <- median_heuristic_bandwidth(Y, rule = "median_distance")
    scaled <- median_heuristic_bandwidth(3 * Y, rule = "median_distance")
    abs(scaled - 3 * base) < 1e-10
  }
)
check(
  "the drf package rule is not scale equivariant, hence not used on z",
  {
    set.seed(2)
    Y <- matrix(rnorm(400), ncol = 2L)
    base <- median_heuristic_bandwidth(Y, rule = "drf_package")
    scaled <- median_heuristic_bandwidth(3 * Y, rule = "drf_package")
    abs(scaled - 3 * base) > 0.1 * base &&
      abs(scaled - 3^0.25 * base) < 1e-10
  }
)
check(
  "the fit records which bandwidth convention produced its kernel",
  identical(fit$bandwidth_rule, "median_distance") &&
    identical(
      causal_drf_fit(
        cell$X, cell$A, cell$Y, num_trees = 20L, num_groups = 2L,
        bandwidth_rule = "drf_package", seed = 1L
      )$bandwidth_rule,
      "drf_package"
    )
)

# --- W-DRF-T versus shared forest on identical information ------------------

check(
  "the shared forest and the two-forest baseline disagree on the same data",
  {
    separate <- wdrft_grouped_weights(
      cell$X, cell$A, cell$Y, X_test,
      num_trees = 200L, num_groups = 20L, seed = 7L
    )
    shared <- weights$treated - weights$control
    max(abs(separate$pooled - shared)) > 1e-6
  }
)
check(
  "both baselines recover the same sign of the effect at the test point",
  {
    separate <- wdrft_grouped_weights(
      cell$X, cell$A, cell$Y, X_test,
      num_trees = 200L, num_groups = 20L, seed = 7L
    )
    evaluation <- matrix(c(-2, 2), ncol = 1L)
    kernel <- gaussian_kernel_matrix(cell$Y, evaluation, fit$bandwidth)
    separate_witness <- separate$pooled %*% kernel
    shared_witness <- (weights$treated - weights$control) %*% kernel
    sign(separate_witness[1L, 1L]) == sign(shared_witness[1L, 1L]) &&
      sign(separate_witness[1L, 2L]) == sign(shared_witness[1L, 2L])
  }
)

# --- input validation ------------------------------------------------------

check(
  "a nonbinary treatment vector is rejected",
  inherits(
    try(causal_drf_fit(cell$X, rep(2L, nrow(cell$X)), cell$Y), silent = TRUE),
    "try-error"
  )
)
check(
  "a zero minimum arm count in the build sample is rejected",
  inherits(
    try(
      causal_drf_fit(cell$X, cell$A, cell$Y, num_trees = 10L, num_groups = 2L,
                     min_arm_build = 0L),
      silent = TRUE
    ),
    "try-error"
  )
)
check(
  "more groups than trees is rejected",
  inherits(
    try(causal_drf_fit(cell$X, cell$A, cell$Y, num_trees = 5L, num_groups = 10L),
        silent = TRUE),
    "try-error"
  )
)

cat(sprintf("\n%d checks, %d failures\n", checks, failures))
if (failures > 0L) {
  quit(status = 1L)
}
