#!/usr/bin/env Rscript
#
# WP2-C2 fidelity check: reproduce a published Causal-DRF simulation cell.
#
# Target: Näf, Park, Susmann (2026), Appendix B, Table 3. The published table
# reports, for the conditional witness function at the fixed test point
# x = (0.7, 0.3, 0.5, 0.68, 0.43), the mean absolute error and the empirical
# coverage of the 95% band, for Causal-DRF and for the two-separate-forests DRF
# benchmark of Näf et al. (2023).
#
# The discriminating signature of the paper is not the MAE, which is close for
# the two methods, but the coverage: under a treatment effect the benchmark
# undercovers badly (78-92%) while Causal-DRF holds near its nominal 95-97%.
# Reproducing that contrast is what establishes that the shared, CKTE-targeted
# splitting was implemented and not merely renamed.
#
# The DRF benchmark is built here from the WP2-C1 forests under the *same*
# half-sampling group structure as Causal-DRF, so the two methods differ in
# exactly one respect: whether the partition is shared and treatment-aware.
# See DEVIATIONS.md.

suppressPackageStartupMessages({
  library(parallel)
})

repro_sibling_path <- function(filename) {
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
  file.path("research", "baselines", "causal_drf_r", filename)
}

source(repro_sibling_path("causal_drf.R"))
source(file.path(dirname(repro_sibling_path("causal_drf.R")), "..", "drf_tlearner.R"))

CAUSAL_DRF_TEST_POINT <- c(0.7, 0.3, 0.5, 0.68, 0.43)
CAUSAL_DRF_REPRODUCTION_ID <- "WP2-C2-REPRODUCTION-v1"

# Published Causal-DRF and DRF witness-function results, Table 3.
CAUSAL_DRF_PUBLISHED <- data.frame(
  regime = c(1L, 1L, 1L, 1L, 3L, 3L, 3L, 3L, 4L, 4L, 4L, 4L),
  n = c(250L, 500L, 1000L, 5000L, 250L, 500L, 1000L, 5000L,
        250L, 500L, 1000L, 5000L),
  causal_drf_mae = c(0.041, 0.033, 0.029, 0.022, 0.065, 0.059, 0.053, 0.041,
                     0.070, 0.061, 0.053, 0.037),
  causal_drf_coverage = c(1.000, 1.000, 1.000, 1.000, 0.970, 0.964, 0.974, 0.960,
                          0.974, 0.968, 0.968, 0.992),
  drf_mae = c(0.035, 0.030, 0.027, 0.019, 0.066, 0.060, 0.054, 0.040,
              0.072, 0.062, 0.055, 0.039),
  drf_coverage = c(1.000, 0.994, 1.000, 1.000, 0.782, 0.802, 0.844, 0.958,
                   0.776, 0.860, 0.918, 0.976),
  stringsAsFactors = FALSE
)

# ---------------------------------------------------------------------------
# Published data generating process, Appendix B
# ---------------------------------------------------------------------------

#' Conditional treatment effect function t(X) of the published regimes.
causal_drf_effect_function <- function(X) {
  eta <- function(x) 1 + (1 + exp(-20 * (x - 1 / 3)))^-1
  eta(X[, 1]) * eta(X[, 2])
}

#' Propensity function e(X) of the published regimes.
causal_drf_propensity <- function(X, confounded) {
  if (!confounded) {
    return(rep(0.5, nrow(X)))
  }
  0.25 * (1 + dbeta(X[, 3], 2, 4))
}

#' Draw one dataset from a published simulation regime.
#'
#' @param n Sample size.
#' @param regime 1 (nothing), 2 (confounding), 3 (effect), or 4 (both).
#' @param seed Integer seed.
#' @return List with X, A, Y, and the regime metadata.
causal_drf_paper_dgp <- function(n, regime, seed) {
  stopifnot(regime %in% 1:4)
  set.seed(seed)
  n_covariates <- 5L
  X <- matrix(runif(n * n_covariates), n, n_covariates)
  confounded <- regime %in% c(2L, 4L)
  has_effect <- regime %in% c(3L, 4L)
  propensity <- causal_drf_propensity(X, confounded)
  A <- rbinom(n, 1L, propensity)
  effect <- if (has_effect) causal_drf_effect_function(X) else rep(0.0, n)
  location <- 2 * X[, 3] - 1 + (A - 0.5) * effect
  Y <- matrix(rnorm(n, location, 1), ncol = 1L)
  list(
    X = X, A = A, Y = Y,
    propensity = propensity,
    confounded = confounded,
    has_effect = has_effect,
    regime = as.integer(regime)
  )
}

#' Monte Carlo approximation of the true conditional witness function.
#'
#' Follows the paper's own protocol: draw from the conditional laws at the test
#' point and average the kernel. The closed form is available for this Gaussian
#' design and is used in `tests/test_causal_drf.R` to check this routine.
#'
#' @param regime Published regime index.
#' @param y_eval Numeric matrix of evaluation points.
#' @param sigma Gaussian bandwidth.
#' @param n_draws Draws per arm.
#' @param seed Integer seed.
#' @return Numeric vector of true witness values at `y_eval`.
causal_drf_true_witness <- function(regime, y_eval, sigma, n_draws = 8000L, seed = 1L) {
  set.seed(seed)
  x <- matrix(CAUSAL_DRF_TEST_POINT, nrow = 1L)
  effect <- if (regime %in% c(3L, 4L)) causal_drf_effect_function(x) else 0.0
  base <- 2 * x[1, 3] - 1
  treated <- matrix(rnorm(n_draws, base + 0.5 * effect, 1), ncol = 1L)
  control <- matrix(rnorm(n_draws, base - 0.5 * effect, 1), ncol = 1L)
  y_eval <- as.matrix(y_eval)
  colMeans(gaussian_kernel_matrix(treated, y_eval, sigma)) -
    colMeans(gaussian_kernel_matrix(control, y_eval, sigma))
}

#' Closed-form true witness for the Gaussian design.
#'
#' For Y ~ N(mu, 1) and the Gaussian kernel of bandwidth sigma,
#' E[k(Y, y)] = sigma / sqrt(sigma^2 + 1) * exp(-(y - mu)^2 / (2 (sigma^2 + 1))).
causal_drf_true_witness_exact <- function(regime, y_eval, sigma) {
  x <- matrix(CAUSAL_DRF_TEST_POINT, nrow = 1L)
  effect <- if (regime %in% c(3L, 4L)) causal_drf_effect_function(x) else 0.0
  base <- 2 * x[1, 3] - 1
  y <- as.numeric(y_eval)
  scale <- sigma / sqrt(sigma^2 + 1)
  variance <- sigma^2 + 1
  scale * (exp(-(y - (base + 0.5 * effect))^2 / (2 * variance)) -
             exp(-(y - (base - 0.5 * effect))^2 / (2 * variance)))
}

# ---------------------------------------------------------------------------
# Two-separate-forests benchmark under matched grouping
# ---------------------------------------------------------------------------

#' Fit the W-DRF-T benchmark with the same half-sampling group structure.
#'
#' For each of B groups, a half-sample is drawn and two ordinary DRFs are fitted
#' on its arm subsets. The pooled signed weights and the B group-level signed
#' weights then enter exactly the same eqs. (4)-(6) inference as Causal-DRF.
#'
#' @param X Numeric n x p covariate matrix.
#' @param A Binary treatment vector.
#' @param Y Numeric n x d response matrix.
#' @param X_test Numeric n_test x p matrix.
#' @param num_trees Total trees per arm.
#' @param num_groups Number of half-sample groups B.
#' @param min_arm_leaf Minimum arm units per leaf, matched to Causal-DRF.
#' @param seed Integer seed.
#' @return List with pooled and grouped signed weights.
wdrft_grouped_weights <- function(X,
                                  A,
                                  Y,
                                  X_test,
                                  num_trees = 2500L,
                                  num_groups = 50L,
                                  min_arm_leaf = 5L,
                                  seed = 1L) {
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  A <- as.integer(A)
  X_test <- as.matrix(X_test)
  n <- nrow(X)
  n_test <- nrow(X_test)
  trees_per_group <- max(1L, as.integer(round(num_trees / num_groups)))

  set.seed(seed)
  grouped <- array(0.0, dim = c(num_groups, n_test, n))
  for (group in seq_len(num_groups)) {
    half_sample <- which(rbinom(n, 1L, 0.5) == 1L)
    if (sum(A[half_sample] == 1L) < 4L * min_arm_leaf ||
        sum(A[half_sample] == 0L) < 4L * min_arm_leaf) {
      half_sample <- seq_len(n)
    }
    control_index <- half_sample[A[half_sample] == 0L]
    treated_index <- half_sample[A[half_sample] == 1L]
    arm_forest <- function(index, arm_seed) {
      drf(
        X = X[index, , drop = FALSE],
        Y = Y[index, , drop = FALSE],
        num.trees = trees_per_group,
        min.node.size = 2L * min_arm_leaf,
        mtry = ncol(X),
        response.scaling = FALSE,
        # The B half-sample groups are managed here, so drf's own confidence
        # grouping is switched off. Leaving it at its default would also divide
        # by zero whenever a group holds fewer than thirty trees.
        ci.group.size = 1L,
        seed = arm_seed
      )
    }
    treated_weights <- matrix(0.0, n_test, n)
    control_weights <- matrix(0.0, n_test, n)
    treated_weights[, treated_index] <-
      as.matrix(predict(arm_forest(treated_index, seed + 2L * group), newdata = X_test)$weights)
    control_weights[, control_index] <-
      as.matrix(predict(arm_forest(control_index, seed + 2L * group + 1L), newdata = X_test)$weights)
    grouped[group, , ] <- treated_weights - control_weights
  }

  pooled <- apply(grouped, c(2L, 3L), mean)
  if (n_test == 1L) {
    pooled <- matrix(pooled, nrow = 1L)
  }
  list(pooled = pooled, grouped = grouped)
}

#' Apply the eqs. (4)-(6) resampling inference to any grouped signed weights.
#'
#' @param pooled Numeric n_test x n_train pooled signed weights.
#' @param grouped Numeric B x n_test x n_train grouped signed weights.
#' @param gram Numeric n_train x n_train Gram matrix.
#' @param y_eval_kernel Numeric n_train x n_eval kernel matrix.
#' @param level Nominal alpha.
#' @return List with the witness, the band, the statistic, and the decision.
grouped_witness_inference <- function(pooled, grouped, gram, y_eval_kernel, level = 0.05) {
  n_test <- nrow(pooled)
  n_groups <- dim(grouped)[1L]
  n_train <- ncol(pooled)
  statistic <- rowSums((pooled %*% gram) * pooled)
  critical <- numeric(n_test)
  for (i in seq_len(n_test)) {
    deviation <- grouped[, i, , drop = FALSE]
    dim(deviation) <- c(n_groups, n_train)
    deviation <- sweep(deviation, 2L, pooled[i, ], `-`)
    critical[i] <- as.numeric(quantile(
      rowSums((deviation %*% gram) * deviation), probs = 1 - level, type = 7
    ))
  }
  witness <- pooled %*% y_eval_kernel
  half_width <- sqrt(critical)
  list(
    statistic = as.numeric(statistic),
    critical_value = critical,
    reject = as.numeric(statistic) > critical,
    witness = witness,
    band_lower = witness - half_width,
    band_upper = witness + half_width
  )
}

# ---------------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------------

#' Run one replication of one published cell for both methods.
#'
#' @param regime Published regime index.
#' @param n Sample size.
#' @param replication Replication index; combined with `regime` and `n` into a
#'   distinct seed per cell.
#' @param num_trees Total trees.
#' @param num_groups Number of subforests B.
#' @param level Nominal alpha.
#' @return Data frame with one row per method.
causal_drf_reproduction_cell <- function(regime,
                                         n,
                                         replication,
                                         num_trees = 2500L,
                                         num_groups = 50L,
                                         level = 0.05,
                                         bandwidth_rule = "drf_package") {
  seed <- as.integer(1e6 * regime + 1e3 * replication + log2(n) * 7)
  data <- causal_drf_paper_dgp(n, regime, seed)
  X_test <- matrix(CAUSAL_DRF_TEST_POINT, nrow = 1L)

  # Both methods share one bandwidth, one Gram matrix, and one evaluation grid.
  # The witness is kernel dependent, so anything else would compare two
  # different estimands rather than two estimators.
  #
  # The reproduction cell defaults to the `drf` package's bandwidth chain, not
  # the median pairwise distance the appendix states in words. The two differ
  # by roughly a factor of two on this design and only the former reproduces
  # the published error levels; see DEVIATIONS.md.
  sigma <- median_heuristic_bandwidth(data$Y, rule = bandwidth_rule, seed = seed)
  gram <- gaussian_kernel_matrix(data$Y, data$Y, sigma)
  eval_kernel <- gram
  truth <- causal_drf_true_witness(regime, data$Y, sigma, seed = seed + 1L)

  summarize <- function(method, inference, elapsed) {
    estimate <- as.numeric(inference$witness[1L, ])
    data.frame(
      reproduction_id = CAUSAL_DRF_REPRODUCTION_ID,
      method = method,
      regime = as.integer(regime),
      n = as.integer(n),
      replication = as.integer(replication),
      seed = seed,
      num_trees = as.integer(num_trees),
      num_groups = as.integer(num_groups),
      bandwidth = sigma,
      bandwidth_rule = bandwidth_rule,
      witness_mae = mean(abs(estimate - truth)),
      band_covers = as.integer(all(
        truth >= inference$band_lower[1L, ] & truth <= inference$band_upper[1L, ]
      )),
      band_half_width = sqrt(inference$critical_value[1L]),
      statistic = inference$statistic[1L],
      critical_value = inference$critical_value[1L],
      reject = as.integer(inference$reject[1L]),
      runtime_seconds = elapsed,
      status = "ok",
      stringsAsFactors = FALSE
    )
  }

  elapsed_of <- function(timing) {
    wall <- timing[["elapsed"]]
    # WSL2 wall clocks occasionally resync mid-run and report a negative
    # interval; fall back to CPU time when that happens.
    if (!is.finite(wall) || wall <= 0) {
      wall <- timing[["user.self"]] + timing[["sys.self"]]
    }
    wall
  }

  causal_timing <- system.time({
    causal_fit <- causal_drf_fit(
      data$X, data$A, data$Y,
      num_trees = num_trees, num_groups = num_groups,
      mtry = ncol(data$X), bandwidth = sigma, seed = seed
    )
    causal_inference <- causal_drf_inference(causal_fit, X_test, Y_eval = data$Y, level = level)
  })

  benchmark_timing <- system.time({
    benchmark_weights <- wdrft_grouped_weights(
      data$X, data$A, data$Y, X_test,
      num_trees = num_trees, num_groups = num_groups, seed = seed
    )
    benchmark_inference <- grouped_witness_inference(
      benchmark_weights$pooled, benchmark_weights$grouped, gram, eval_kernel, level
    )
  })

  rbind(
    summarize("CAUSAL-DRF", causal_inference, elapsed_of(causal_timing)),
    summarize("W-DRF-T", benchmark_inference, elapsed_of(benchmark_timing))
  )
}

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

#' Run a grid of replications and return the stacked result frame.
causal_drf_reproduction_run <- function(regimes,
                                        sizes,
                                        replications,
                                        num_trees = 2500L,
                                        num_groups = 50L,
                                        workers = 1L) {
  grid <- expand.grid(
    replication = seq_len(replications),
    n = sizes,
    regime = regimes,
    KEEP.OUT.ATTRS = FALSE
  )
  jobs <- split(grid, seq_len(nrow(grid)))
  runner <- function(job) {
    tryCatch(
      causal_drf_reproduction_cell(
        regime = job$regime, n = job$n, replication = job$replication,
        num_trees = num_trees, num_groups = num_groups
      ),
      error = function(e) {
        data.frame(
          reproduction_id = CAUSAL_DRF_REPRODUCTION_ID,
          method = NA_character_, regime = as.integer(job$regime),
          n = as.integer(job$n), replication = as.integer(job$replication),
          seed = NA_integer_, num_trees = as.integer(num_trees),
          num_groups = as.integer(num_groups), bandwidth = NA_real_,
          bandwidth_rule = NA_character_,
          witness_mae = NA_real_, band_covers = NA_integer_,
          band_half_width = NA_real_, statistic = NA_real_,
          critical_value = NA_real_, reject = NA_integer_,
          runtime_seconds = NA_real_, status = paste("error:", conditionMessage(e)),
          stringsAsFactors = FALSE
        )
      }
    )
  }
  results <- if (workers > 1L) {
    parallel::mclapply(jobs, runner, mc.cores = workers, mc.preschedule = FALSE)
  } else {
    lapply(jobs, runner)
  }
  do.call(rbind, results)
}

# Run the grid only when this file is the script Rscript was pointed at. Being
# sourced by a test must never launch a multi-hour simulation.
local({
  invoked <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  is_entry_point <- length(invoked) == 1L &&
    identical(basename(sub("^--file=", "", invoked)), "reproduction.R")

  arguments <- commandArgs(trailingOnly = TRUE)
  value_of <- function(flag, default) {
    hit <- which(arguments == flag)
    if (length(hit) == 1L && length(arguments) > hit) arguments[hit + 1L] else default
  }
  if (is_entry_point) {
    regimes <- as.integer(strsplit(value_of("--regimes", "1,3"), ",")[[1L]])
    sizes <- as.integer(strsplit(value_of("--sizes", "250,1000"), ",")[[1L]])
    replications <- as.integer(value_of("--replications", "100"))
    num_trees <- as.integer(value_of("--trees", "2500"))
    num_groups <- as.integer(value_of("--groups", "50"))
    workers <- as.integer(value_of("--workers", "1"))
    output <- value_of("--output", "results/smoke/causal_drf_reproduction.csv")

    cat(sprintf(
      "regimes=%s sizes=%s replications=%d trees=%d groups=%d workers=%d\n",
      paste(regimes, collapse = ","), paste(sizes, collapse = ","),
      replications, num_trees, num_groups, workers
    ))
    frame <- causal_drf_reproduction_run(
      regimes, sizes, replications, num_trees, num_groups, workers
    )
    dir.create(dirname(output), showWarnings = FALSE, recursive = TRUE)
    write.csv(frame, output, row.names = FALSE)
    cat(sprintf("wrote %d rows to %s\n", nrow(frame), output))
  }
})
