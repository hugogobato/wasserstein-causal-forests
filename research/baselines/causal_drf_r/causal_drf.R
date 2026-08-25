#!/usr/bin/env Rscript
#
# WP2-C2. Faithful shared Causal-DRF.
#
# Reimplementation of Näf, Park, Susmann (2026), "Causal-DRF: Conditional
# Kernel Treatment Effect Estimation using Distributional Random Forest",
# AISTATS 2026 (PMLR v300), arXiv:2411.08778v2.
#
# The authors published no code (the AISTATS page records
# `Code Dataset Promise: No`), so this is written from the paper text. It
# shares no source with the GPL-3.0 `drf` package and does not load its
# namespace; see `research/baselines/PROVENANCE.md`.
#
# The methodological difference from W-DRF-T is the whole point of the
# baseline: Causal-DRF fits ONE forest whose split criterion targets the
# conditional kernel treatment effect directly, instead of two forests fitted
# separately per arm. Concretely it implements
#
#   * the treatment-aware weighted-MMD split criterion, eq. (2) and eq. (C.1),
#     approximated by random Fourier features, eq. (C.2);
#   * both-arm minimums in candidate children and leaves, (F4);
#   * honesty, (F1): one half of each tree's subsample determines the splits,
#     the other populates the leaves;
#   * signed prediction weights, eq. (3);
#   * subforest grouping and half-sampling, Section 3 and Algorithm 1;
#   * the resampled test and uniform confidence band, eqs. (4), (5), (6).
#
# Any variant that drops one of these must be labelled `DRF-inspired` and must
# not be presented as the published incumbent. Deviations that remain are
# listed in `DEVIATIONS.md`.

CAUSAL_DRF_METHOD_ID <- "CAUSAL-DRF"
CAUSAL_DRF_IMPLEMENTATION_ID <- "CAUSAL-DRF-REIMPLEMENTATION-v1"
CAUSAL_DRF_PAPER <- "Naf, Park, Susmann (2026), AISTATS, arXiv:2411.08778v2"
CAUSAL_DRF_AUTHOR_CODE <- "none published"

suppressPackageStartupMessages({
  library(Rcpp)
})

#' Resolve a file that sits next to this script.
cdrf_sibling_path <- function(filename) {
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

if (!exists("BASELINE_SCHEMA_VERSION")) {
  source(file.path(dirname(cdrf_sibling_path("causal_drf.R")), "..", "baseline_common.R"))
}

# The tree builder is compiled once and cached by content hash, so repeated
# sessions and the parallel workers of a tournament shard pay for it once.
if (!exists("causal_drf_grow_tree")) {
  local({
    cache <- Sys.getenv("WCF_RCPP_CACHE", unset = "")
    if (!nzchar(cache)) {
      cache <- file.path(
        tools::R_user_dir("wasserstein-causal-forests", which = "cache"),
        "causal-drf-rcpp"
      )
    }
    dir.create(cache, showWarnings = FALSE, recursive = TRUE)
    Rcpp::sourceCpp(
      cdrf_sibling_path("causal_tree.cpp"),
      cacheDir = cache,
      env = globalenv()
    )
  })
}

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

#' Report what this implementation is and is not.
#'
#' @return Named list recorded alongside every result row.
causal_drf_environment <- function() {
  list(
    method_id = CAUSAL_DRF_METHOD_ID,
    implementation_id = CAUSAL_DRF_IMPLEMENTATION_ID,
    paper = CAUSAL_DRF_PAPER,
    author_code = CAUSAL_DRF_AUTHOR_CODE,
    reimplementation = TRUE,
    shares_source_with_drf_package = FALSE,
    r_version = paste(R.version$major, R.version$minor, sep = ".")
  )
}

# ---------------------------------------------------------------------------
# Random Fourier features
# ---------------------------------------------------------------------------

#' Draw random Fourier frequencies and evaluate the feature map.
#'
#' Appendix C: for the Gaussian kernel with bandwidth sigma, Bochner's theorem
#' gives the spectral measure N_d(0, sigma^-2 I_d), and
#' phi_b(y) = exp(i omega_b' y) approximates k up to a scale factor. The real
#' and imaginary parts are returned as one n x 2S matrix so that the split
#' search never sees the response dimension.
#'
#' @param Y Numeric n x d response matrix.
#' @param n_features Number of frequencies S.
#' @param sigma Gaussian bandwidth.
#' @return List with the frequency matrix and the n x 2S feature matrix.
causal_drf_fourier_features <- function(Y, n_features, sigma) {
  Y <- as.matrix(Y)
  n_dimensions <- ncol(Y)
  omega <- matrix(
    rnorm(n_features * n_dimensions, mean = 0, sd = 1 / sigma),
    nrow = n_dimensions, ncol = n_features
  )
  projected <- Y %*% omega
  list(omega = omega, Phi = cbind(cos(projected), sin(projected)))
}

# ---------------------------------------------------------------------------
# Forest
# ---------------------------------------------------------------------------

#' Fit a shared treatment-aware Causal-DRF.
#'
#' @param X Numeric n x p covariate matrix.
#' @param A Binary treatment vector of length n.
#' @param Y Numeric n x d response matrix. For this project the response is the
#'   rescaled quantile vector z = diag(sqrt(w)) q, on which Euclidean geometry
#'   is exactly discretized W_2 geometry.
#' @param num_trees Total number of trees N.
#' @param num_groups Number of subforests B; N / B trees are built per group.
#' @param num_features Random Fourier frequencies S used by the split criterion.
#' @param mtry Candidate covariates per node.
#' @param min_arm_leaf Minimum units of each arm per leaf in the populating
#'   sample; the kappa of forest assumption (F4).
#' @param min_arm_build Minimum units of each arm per child in the split-
#'   determining sample. Must be at least 1 for the criterion to be defined.
#' @param alpha Minimum fraction of a parent's units kept by each child, (F4).
#'   Must not exceed 0.2.
#' @param sample_fraction Per-tree subsample fraction of its group half-sample,
#'   (F5).
#' @param honesty_fraction Share of a tree's subsample used to determine splits.
#' @param max_depth Depth cap.
#' @param bandwidth Gaussian bandwidth; derived from `bandwidth_rule` when
#'   `NULL`.
#' @param bandwidth_rule Median-heuristic convention. `"median_distance"` is the
#'   equivariant rule the paper states in words and is the default here;
#'   `"drf_package"` is the chain the `drf` package computes and is what
#'   reproduces the published numbers. See `median_heuristic_bandwidth`.
#' @param seed Integer seed controlling every random draw.
#' @return A `causal_drf_fit` object.
causal_drf_fit <- function(X,
                           A,
                           Y,
                           num_trees = 2500L,
                           num_groups = 50L,
                           num_features = 10L,
                           mtry = NULL,
                           min_arm_leaf = 5L,
                           min_arm_build = 2L,
                           alpha = 0.05,
                           sample_fraction = 0.5,
                           honesty_fraction = 0.5,
                           max_depth = 30L,
                           bandwidth = NULL,
                           bandwidth_rule = c("median_distance", "drf_package"),
                           seed = 1L) {
  bandwidth_rule <- match.arg(bandwidth_rule)
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  A <- as.integer(round(as.numeric(A)))
  n <- nrow(X)

  if (nrow(Y) != n || length(A) != n) {
    stop("X, A, and Y must describe the same units")
  }
  if (!all(A %in% c(0L, 1L))) {
    stop("A must be binary")
  }
  if (alpha <= 0 || alpha > 0.2) {
    stop("alpha-regularity requires 0 < alpha <= 0.2 (forest assumption F4)")
  }
  if (min_arm_build < 1L) {
    stop("min_arm_build must be at least 1 or the split criterion is undefined")
  }
  if (num_groups < 1L || num_trees < num_groups) {
    stop("num_trees must be at least num_groups, and num_groups at least one")
  }
  if (sum(A == 1L) < 2L * min_arm_leaf || sum(A == 0L) < 2L * min_arm_leaf) {
    stop("each treatment arm needs at least 2 * min_arm_leaf units")
  }
  if (is.null(mtry)) {
    mtry <- min(ceiling(sqrt(ncol(X)) + 20), ncol(X))
  }

  set.seed(seed)
  if (is.null(bandwidth)) {
    bandwidth <- median_heuristic_bandwidth(Y, rule = bandwidth_rule, seed = seed)
  }
  if (!is.finite(bandwidth) || bandwidth <= 0) {
    stop("bandwidth must be positive and finite")
  }

  features <- causal_drf_fourier_features(Y, num_features, bandwidth)
  trees_per_group <- max(1L, as.integer(round(num_trees / num_groups)))
  total_trees <- trees_per_group * num_groups

  trees <- vector("list", total_trees)
  tree_group <- integer(total_trees)
  cursor <- 0L
  for (group in seq_len(num_groups)) {
    # Half-sampling, Section 3: U_i ~ Bernoulli(1/2), S = {i : U_i = 1}. The
    # subforest is fitted on S alone, which is what makes the B subforest
    # predictions an approximation of the sampling distribution.
    half_sample <- which(rbinom(n, 1L, 0.5) == 1L)
    if (sum(A[half_sample] == 1L) < 2L * min_arm_leaf ||
        sum(A[half_sample] == 0L) < 2L * min_arm_leaf) {
      half_sample <- seq_len(n)
    }
    for (tree_index in seq_len(trees_per_group)) {
      # (F5) subsampling. The target size is `sample_fraction * n`, measured
      # against the FULL sample and then drawn from the group half-sample, so
      # that adding subforest grouping does not silently halve the data each
      # tree sees. This is the composition grf and drf use for their
      # confidence-interval groups; see DEVIATIONS.md.
      subsample_size <- max(
        4L * min_arm_leaf,
        as.integer(floor(sample_fraction * n))
      )
      subsample_size <- min(subsample_size, length(half_sample))
      subsample <- sample(half_sample, subsample_size, replace = FALSE)
      n_build <- max(1L, as.integer(floor(honesty_fraction * subsample_size)))
      build_rows <- subsample[seq_len(n_build)]
      pop_rows <- subsample[-seq_len(n_build)]
      cursor <- cursor + 1L
      trees[[cursor]] <- causal_drf_grow_tree(
        X = X,
        W = A,
        Phi = features$Phi,
        build_rows = as.integer(build_rows - 1L),
        pop_rows = as.integer(pop_rows - 1L),
        mtry = as.integer(mtry),
        min_arm_build = as.integer(min_arm_build),
        min_arm_pop = as.integer(min_arm_leaf),
        alpha = alpha,
        max_depth = as.integer(max_depth)
      )
      tree_group[cursor] <- group
    }
  }

  structure(
    list(
      method_id = CAUSAL_DRF_METHOD_ID,
      implementation_id = CAUSAL_DRF_IMPLEMENTATION_ID,
      trees = trees,
      tree_group = tree_group,
      num_groups = as.integer(num_groups),
      trees_per_group = trees_per_group,
      X_train = X,
      A_train = A,
      Y_train = Y,
      bandwidth = bandwidth,
      bandwidth_rule = bandwidth_rule,
      omega = features$omega,
      seed = as.integer(seed),
      hyperparameters = list(
        num_trees = total_trees,
        num_groups = as.integer(num_groups),
        num_features = as.integer(num_features),
        mtry = mtry,
        min_arm_leaf = as.integer(min_arm_leaf),
        min_arm_build = as.integer(min_arm_build),
        alpha = alpha,
        sample_fraction = sample_fraction,
        honesty_fraction = honesty_fraction,
        max_depth = as.integer(max_depth)
      )
    ),
    class = "causal_drf_fit"
  )
}

#' Compute the signed prediction weights of eq. (3).
#'
#' @param fit A `causal_drf_fit` object.
#' @param X_test Numeric n_test x p matrix.
#' @param with_groups Whether to also return per-subforest weights.
#' @return List with `treated`, `control`, and optionally the grouped arrays.
causal_drf_weights <- function(fit, X_test, with_groups = FALSE) {
  stopifnot(inherits(fit, "causal_drf_fit"))
  X_test <- as.matrix(X_test)
  if (ncol(X_test) != ncol(fit$X_train)) {
    stop("X_test must have the training number of covariates")
  }
  causal_drf_predict_weights(
    trees = fit$trees,
    tree_group = as.integer(fit$tree_group - 1L),
    n_groups = fit$num_groups,
    X_test = X_test,
    W = fit$A_train,
    n_train = nrow(fit$X_train),
    with_groups = with_groups
  )
}

#' Predict in the common WP2-C schema.
#'
#' The arm summaries are exposed because the signed weights of eq. (3) split
#' exactly into two normalized arm weight vectors, each supported on its own
#' arm. They are read off ONE shared, CKTE-targeted partition, so they are not
#' the same object as the two independently fitted arm laws of W-DRF-T and must
#' not be relabelled as such.
#'
#' @param fit A `causal_drf_fit` object.
#' @param X_test Numeric n_test x p matrix.
#' @param Q_train Numeric n_train x K matrix of training quantile vectors.
#' @param quad_weights Numeric length-K vector of quadrature weights.
#' @param reference_quantiles Optional length-K quantile vector of nu_star.
#' @param functionals Optional n_train x J matrix of T_j(Y_i) values.
#' @return A `wp2c_prediction` list.
causal_drf_predict <- function(fit,
                               X_test,
                               Q_train,
                               quad_weights,
                               reference_quantiles = NULL,
                               functionals = NULL) {
  weights <- causal_drf_weights(fit, X_test, with_groups = FALSE)
  baseline_prediction(
    method_id = CAUSAL_DRF_METHOD_ID,
    weights_treated = weights$treated,
    weights_control = weights$control,
    Q_train = Q_train,
    quad_weights = quad_weights,
    reference_quantiles = reference_quantiles,
    functionals = functionals,
    extra = list(
      shared_partition = TRUE,
      treatment_aware_splitting = TRUE,
      contributing_trees = weights$contributing_trees,
      bandwidth = fit$bandwidth,
      seed = fit$seed,
      hyperparameters = fit$hyperparameters
    )
  )
}

#' Conditional kernel treatment effect and its witness function.
#'
#' `tau_k(x)` is the RKHS element sum_i w_i(x) k(Y_i, .) of eq. (3). What is
#' reported is its squared RKHS norm, which is the test statistic of eq. (5),
#' and the witness function y -> tau_k(x)(y) whose sign says which outcomes are
#' locally more likely under treatment.
#'
#' @param fit A `causal_drf_fit` object.
#' @param X_test Numeric n_test x p matrix.
#' @param Y_eval Optional n_eval x d matrix of evaluation points for the
#'   witness function.
#' @return List with the signed weights, the squared norm, and the witness.
causal_drf_ckte <- function(fit, X_test, Y_eval = NULL) {
  weights <- causal_drf_weights(fit, X_test, with_groups = FALSE)
  signed <- weights$treated - weights$control
  gram <- gaussian_kernel_matrix(fit$Y_train, fit$Y_train, fit$bandwidth)
  squared_norm <- rowSums((signed %*% gram) * signed)
  witness <- NULL
  if (!is.null(Y_eval)) {
    witness <- signed %*% gaussian_kernel_matrix(
      fit$Y_train, as.matrix(Y_eval), fit$bandwidth
    )
  }
  list(
    weights_signed = signed,
    squared_norm = as.numeric(squared_norm),
    witness = witness
  )
}

#' Resampling test and uniform confidence band, eqs. (4), (5), (6).
#'
#' The B subforest predictions approximate the sampling distribution of the
#' statistic. `q_{n,alpha}` is the (1 - alpha) quantile of
#' `||tau^{S_b} - tau||^2 = (w^{S_b} - w)' K (w^{S_b} - w)` across the B
#' subforests; the test rejects when `||tau||^2` exceeds it, and the band is
#' `tau(y) +/- sqrt(q_{n,alpha})`. Boundedness of the Gaussian kernel gives
#' the constant C = sup_y k(y, y) = 1 used in eq. (20).
#'
#' @param fit A `causal_drf_fit` object.
#' @param X_test Numeric n_test x p matrix. Inference is valid for a fixed test
#'   point; several test points need a multiple-testing correction.
#' @param Y_eval Optional n_eval x d matrix of witness evaluation points.
#' @param level Nominal alpha.
#' @return List with the statistic, the critical value, the rejection flag, and
#'   the band.
causal_drf_inference <- function(fit, X_test, Y_eval = NULL, level = 0.05) {
  X_test <- as.matrix(X_test)
  weights <- causal_drf_weights(fit, X_test, with_groups = TRUE)
  signed <- weights$treated - weights$control
  gram <- gaussian_kernel_matrix(fit$Y_train, fit$Y_train, fit$bandwidth)
  statistic <- rowSums((signed %*% gram) * signed)

  n_test <- nrow(X_test)
  n_train <- nrow(fit$X_train)
  grouped <- array(
    weights$grouped_treated - weights$grouped_control,
    dim = c(fit$num_groups, n_test, n_train)
  )

  critical <- numeric(n_test)
  resampled <- matrix(0.0, fit$num_groups, n_test)
  for (i in seq_len(n_test)) {
    deviation <- grouped[, i, , drop = FALSE]
    dim(deviation) <- c(fit$num_groups, n_train)
    deviation <- sweep(deviation, 2L, signed[i, ], `-`)
    resampled[, i] <- rowSums((deviation %*% gram) * deviation)
    critical[i] <- as.numeric(quantile(resampled[, i], probs = 1 - level, type = 7))
  }

  witness <- NULL
  band_lower <- NULL
  band_upper <- NULL
  if (!is.null(Y_eval)) {
    witness <- signed %*% gaussian_kernel_matrix(
      fit$Y_train, as.matrix(Y_eval), fit$bandwidth
    )
    half_width <- sqrt(critical)
    band_lower <- witness - half_width
    band_upper <- witness + half_width
  }

  list(
    statistic = as.numeric(statistic),
    critical_value = critical,
    reject = as.numeric(statistic) > critical,
    level = level,
    resampled_statistics = resampled,
    weights_signed = signed,
    witness = witness,
    band_lower = band_lower,
    band_upper = band_upper
  )
}

# ---------------------------------------------------------------------------
# Reference split criterion, kept in R for verification
# ---------------------------------------------------------------------------

#' Evaluate the Fourier-approximated split criterion of eq. (C.2) in R.
#'
#' Mirrors what the compiled search maximizes, so a test can confirm that the
#' split the tree actually chose is the argmax over all admissible candidates.
#'
#' @param Phi Numeric n x 2S feature matrix restricted to the node.
#' @param W Binary treatment vector restricted to the node.
#' @param left Logical vector marking the left child.
#' @return Scalar criterion value.
causal_drf_fourier_criterion <- function(Phi, W, left) {
  Phi <- as.matrix(Phi)
  W <- as.integer(W)
  left <- as.logical(left)
  n_left <- sum(left)
  n_right <- sum(!left)
  if (n_left == 0L || n_right == 0L) {
    return(0.0)
  }

  side_embedding <- function(mask) {
    treated <- mask & W == 1L
    control <- mask & W == 0L
    if (!any(treated) || !any(control)) {
      return(NULL)
    }
    colMeans(Phi[treated, , drop = FALSE]) - colMeans(Phi[control, , drop = FALSE])
  }

  left_part <- side_embedding(left)
  right_part <- side_embedding(!left)
  if (is.null(left_part) || is.null(right_part)) {
    return(0.0)
  }

  gap <- left_part - right_part
  scale <- (n_left * n_right) / ((n_left + n_right)^2)
  # ncol(Phi) counts real and imaginary parts, so the 2 / ncol factor averages
  # over the S frequencies.
  as.numeric(scale * sum(gap^2) * 2 / ncol(Phi))
}

#' Evaluate the weighted-MMD split criterion directly from its definition.
#'
#' A transcription of eq. (C.1) with an exact kernel, used only to check the
#' compiled Fourier-approximated search against a hand case. It is deliberately
#' slow and literal.
#'
#' @param Y Numeric n x d response matrix restricted to the node.
#' @param W Binary treatment vector restricted to the node.
#' @param left Logical vector marking the left child.
#' @param sigma Gaussian bandwidth.
#' @return Scalar criterion value.
causal_drf_reference_criterion <- function(Y, W, left, sigma) {
  Y <- as.matrix(Y)
  W <- as.integer(W)
  left <- as.logical(left)
  n_left <- sum(left)
  n_right <- sum(!left)
  if (n_left == 0L || n_right == 0L) {
    return(0.0)
  }

  signed_weights <- function(mask) {
    treated <- mask & W == 1L
    control <- mask & W == 0L
    if (!any(treated) || !any(control)) {
      return(NULL)
    }
    weights <- numeric(length(W))
    weights[treated] <- 1 / sum(treated)
    weights[control] <- -1 / sum(control)
    weights
  }

  nu_left <- signed_weights(left)
  nu_right <- signed_weights(!left)
  if (is.null(nu_left) || is.null(nu_right)) {
    return(0.0)
  }

  contrast <- nu_left - nu_right
  gram <- gaussian_kernel_matrix(Y, Y, sigma)
  scale <- (n_left * n_right) / ((n_left + n_right)^2)
  as.numeric(scale * (t(contrast) %*% gram %*% contrast))
}
