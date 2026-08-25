#!/usr/bin/env Rscript
#
# Shared geometry and prediction-schema helpers for the WP2-C forest baselines.
#
# Both W-DRF-T (WP2-C1) and Causal-DRF (WP2-C2) consume the same rescaled
# quantile representation and must emit the same prediction record, so that a
# tournament cell can swap one for the other without touching downstream code.
# Nothing in this file is method specific.
#
# The representation is the one fixed in `research_phases/_phase_shared.md`:
# a distribution-valued outcome is carried by its quantile vector
# q = (Q_Y(u_1), ..., Q_Y(u_K)) on a fixed grid with positive quadrature
# weights w, and the rescaled coordinates z_k = sqrt(w_k) q_k satisfy
#
#     || z - z' ||_2^2 = sum_k w_k (q_k - q_k')^2 = W_{2,K}^2(q, q').
#
# Euclidean geometry on z is therefore exactly discretized 2-Wasserstein
# geometry on q. This is a representation map, not a methodological
# contribution; see F0.5 of the research plan.

BASELINE_SCHEMA_VERSION <- "WP2-C-PRED-SCHEMA-v1"

# ---------------------------------------------------------------------------
# Quantile-grid geometry
# ---------------------------------------------------------------------------

#' Validate a vector of positive finite quadrature weights.
#'
#' @param quad_weights Numeric vector of length K.
#' @param n_coordinates Optional expected length.
#' @return The validated numeric vector.
validate_quad_weights <- function(quad_weights, n_coordinates = NULL) {
  quad_weights <- as.numeric(quad_weights)
  if (length(quad_weights) == 0L) {
    stop("quad_weights must be a nonempty numeric vector")
  }
  if (!all(is.finite(quad_weights))) {
    stop("quad_weights must be finite")
  }
  if (any(quad_weights <= 0)) {
    stop("all quadrature weights must be strictly positive")
  }
  if (!is.null(n_coordinates) && length(quad_weights) != n_coordinates) {
    stop(sprintf(
      "quad_weights has length %d, expected %d",
      length(quad_weights), n_coordinates
    ))
  }
  quad_weights
}

#' Validate a matrix of monotone quantile vectors.
#'
#' @param Q Numeric matrix, one row per unit and one column per grid point.
#' @param check_monotone Whether to enforce nondecreasing rows.
#' @param tolerance Absolute slack allowed on the monotonicity check.
#' @return The validated matrix.
validate_quantile_matrix <- function(Q, check_monotone = TRUE, tolerance = 1e-12) {
  Q <- as.matrix(Q)
  if (!is.numeric(Q) || nrow(Q) == 0L || ncol(Q) == 0L) {
    stop("Q must be a nonempty numeric matrix")
  }
  if (!all(is.finite(Q))) {
    stop("Q must be finite")
  }
  if (check_monotone && ncol(Q) > 1L) {
    steps <- Q[, -1L, drop = FALSE] - Q[, -ncol(Q), drop = FALSE]
    if (any(steps < -tolerance)) {
      stop("every row of Q must be a nondecreasing quantile vector")
    }
  }
  Q
}

#' Map quantile vectors to rescaled Wasserstein coordinates z = diag(sqrt(w)) q.
#'
#' @param Q Numeric n x K matrix of quantile vectors.
#' @param quad_weights Numeric vector of length K.
#' @return Numeric n x K matrix of rescaled coordinates.
rescale_quantiles <- function(Q, quad_weights) {
  Q <- as.matrix(Q)
  quad_weights <- validate_quad_weights(quad_weights, ncol(Q))
  sweep(Q, 2L, sqrt(quad_weights), `*`)
}

#' Invert `rescale_quantiles`.
#'
#' @param Z Numeric n x K matrix of rescaled coordinates.
#' @param quad_weights Numeric vector of length K.
#' @return Numeric n x K matrix of quantile vectors.
unscale_quantiles <- function(Z, quad_weights) {
  Z <- as.matrix(Z)
  quad_weights <- validate_quad_weights(quad_weights, ncol(Z))
  sweep(Z, 2L, sqrt(quad_weights), `/`)
}

#' Discretized 2-Wasserstein distance between quantile vectors and one target.
#'
#' @param Q Numeric n x K matrix of quantile vectors.
#' @param target Numeric length-K quantile vector.
#' @param quad_weights Numeric vector of length K.
#' @return Numeric vector of length n.
w2_grid_distance <- function(Q, target, quad_weights) {
  Q <- as.matrix(Q)
  quad_weights <- validate_quad_weights(quad_weights, ncol(Q))
  target <- as.numeric(target)
  if (length(target) != ncol(Q)) {
    stop("target must have one entry per grid point")
  }
  gaps <- sweep(Q, 2L, target, `-`)
  sqrt(as.numeric((gaps^2) %*% quad_weights))
}

#' Maximum absolute error of the rescaled-geometry identity.
#'
#' Compares squared Euclidean distances on z against weighted squared
#' quantile distances on q for every pair of rows. Used as the E4 check
#' required by both WP2-C1 and WP2-C2.
#'
#' @param Q Numeric n x K matrix of quantile vectors.
#' @param quad_weights Numeric vector of length K.
#' @return List with the maximum squared-distance error and kernel error.
check_geometry_identity <- function(Q, quad_weights) {
  Q <- as.matrix(Q)
  quad_weights <- validate_quad_weights(quad_weights, ncol(Q))
  Z <- rescale_quantiles(Q, quad_weights)
  euclidean <- as.matrix(dist(Z))^2
  n <- nrow(Q)
  weighted <- matrix(0.0, n, n)
  for (i in seq_len(n)) {
    gaps <- sweep(Q, 2L, Q[i, ], `-`)
    weighted[i, ] <- as.numeric((gaps^2) %*% quad_weights)
  }
  sigma <- 1.0
  kernel_euclidean <- exp(-euclidean / (2 * sigma^2))
  kernel_weighted <- exp(-weighted / (2 * sigma^2))
  list(
    max_squared_distance_error = max(abs(euclidean - weighted)),
    max_kernel_error = max(abs(kernel_euclidean - kernel_weighted))
  )
}

#' Gaussian-kernel bandwidth by the median heuristic.
#'
#' Two conventions are available and they are not interchangeable.
#'
#' `"median_distance"` is the median pairwise Euclidean distance between
#' response rows, which is the median heuristic as stated by Gretton et al.
#' (2012) and as described in words by both the DRF and the Causal-DRF papers.
#' On the rescaled coordinates it is the median pairwise discretized W_2
#' distance, and it is equivariant: scaling the responses by c scales the
#' bandwidth by c.
#'
#' `"drf_package"` is the chain the `drf` package actually computes,
#' `sqrt(median(sqrt(dist(Y) / 2)))`. It is what reproduces the published
#' Causal-DRF numbers, so it is the convention the reproduction cell uses. It
#' is **not** equivariant: scaling the responses by c scales it by c^(1/4).
#' That is why `drf` standardizes the response by default, and it is why this
#' convention must not be used on the rescaled quantile representation, whose
#' whole purpose is to carry an exact and meaningful scale.
#'
#' @param Z Numeric n x d response matrix.
#' @param rule Either "median_distance" or "drf_package".
#' @param max_pairs Cap on the number of rows used for the median.
#' @param seed Seed used when subsampling rows.
#' @return Positive scalar bandwidth.
median_heuristic_bandwidth <- function(Z,
                                       rule = c("median_distance", "drf_package"),
                                       max_pairs = 1000L,
                                       seed = 1L) {
  rule <- match.arg(rule)
  Z <- as.matrix(Z)
  if (nrow(Z) > max_pairs) {
    state <- .Random.seed_snapshot()
    on.exit(.Random.seed_restore(state), add = TRUE)
    set.seed(seed)
    Z <- Z[sample.int(nrow(Z), max_pairs), , drop = FALSE]
  }
  distances <- as.numeric(dist(Z))
  distances <- distances[distances > 0]
  if (length(distances) == 0L) {
    return(1.0)
  }
  if (identical(rule, "drf_package")) {
    return(sqrt(as.numeric(median(sqrt(distances / 2)))))
  }
  as.numeric(median(distances))
}

#' Gaussian kernel matrix k(u, v) = exp(-||u - v||^2 / (2 sigma^2)).
#'
#' @param Z1 Numeric n1 x d matrix.
#' @param Z2 Numeric n2 x d matrix.
#' @param sigma Positive bandwidth.
#' @return Numeric n1 x n2 kernel matrix.
gaussian_kernel_matrix <- function(Z1, Z2, sigma) {
  Z1 <- as.matrix(Z1)
  Z2 <- as.matrix(Z2)
  if (ncol(Z1) != ncol(Z2)) {
    stop("Z1 and Z2 must have the same number of columns")
  }
  if (!is.finite(sigma) || sigma <= 0) {
    stop("sigma must be a positive finite bandwidth")
  }
  cross <- Z1 %*% t(Z2)
  norm1 <- rowSums(Z1^2)
  norm2 <- rowSums(Z2^2)
  squared <- outer(norm1, norm2, `+`) - 2 * cross
  squared[squared < 0] <- 0
  exp(-squared / (2 * sigma^2))
}

# The RNG snapshot helpers keep bandwidth selection from perturbing the
# caller's random stream, which would otherwise break seed reproducibility.
.Random.seed_snapshot <- function() {
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    get(".Random.seed", envir = globalenv())
  } else {
    NULL
  }
}

.Random.seed_restore <- function(state) {
  if (is.null(state)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
  } else {
    assign(".Random.seed", state, envir = globalenv())
  }
}

# ---------------------------------------------------------------------------
# Common prediction schema
# ---------------------------------------------------------------------------

#' Assemble the prediction record shared by every WP2-C forest baseline.
#'
#' Both baselines produce, for each test point, one normalized weight vector
#' over the training sample per treatment arm. Everything else in the record
#' is a deterministic function of those weights and the training targets, so a
#' downstream tournament cell can treat the two methods interchangeably.
#'
#' @param method_id Label recorded in every downstream schema.
#' @param weights_treated Numeric n_test x n_train matrix, rows summing to 1,
#'   supported on treated training units.
#' @param weights_control Numeric n_test x n_train matrix, rows summing to 1,
#'   supported on control training units.
#' @param Q_train Numeric n_train x K matrix of training quantile vectors.
#' @param quad_weights Numeric vector of length K.
#' @param reference_quantiles Optional length-K quantile vector of the frozen
#'   reference measure nu_star. When supplied, arm reference levels
#'   r_a(x) = E[W_2(Y^a, nu_star) | X = x] and the reference effect are added.
#' @param functionals Optional numeric n_train x J matrix whose column j holds
#'   T_j(Y_i). Column names become the functional names.
#' @param extra Named list of method-specific fields appended verbatim.
#' @return A `wp2c_prediction` list.
baseline_prediction <- function(method_id,
                                weights_treated,
                                weights_control,
                                Q_train,
                                quad_weights,
                                reference_quantiles = NULL,
                                functionals = NULL,
                                extra = list()) {
  weights_treated <- as.matrix(weights_treated)
  weights_control <- as.matrix(weights_control)
  Q_train <- validate_quantile_matrix(Q_train)
  quad_weights <- validate_quad_weights(quad_weights, ncol(Q_train))

  if (!identical(dim(weights_treated), dim(weights_control))) {
    stop("arm weight matrices must have identical shape")
  }
  if (ncol(weights_treated) != nrow(Q_train)) {
    stop("weight matrices must have one column per training unit")
  }

  record <- list(
    method_id = method_id,
    schema_version = BASELINE_SCHEMA_VERSION,
    n_test = nrow(weights_treated),
    n_train = nrow(Q_train),
    n_grid = ncol(Q_train),
    quad_weights = quad_weights,
    weights_treated = weights_treated,
    weights_control = weights_control,
    weights_signed = weights_treated - weights_control
  )

  # Convex combinations of nondecreasing quantile vectors stay nondecreasing,
  # so the conditional barycenter needs no isotonic projection.
  record$barycenter_treated <- weights_treated %*% Q_train
  record$barycenter_control <- weights_control %*% Q_train
  record$barycenter_contrast <- record$barycenter_treated - record$barycenter_control

  if (!is.null(reference_quantiles)) {
    distances <- w2_grid_distance(Q_train, reference_quantiles, quad_weights)
    record$reference_quantiles <- as.numeric(reference_quantiles)
    record$reference_treated <- as.numeric(weights_treated %*% distances)
    record$reference_control <- as.numeric(weights_control %*% distances)
    record$reference_effect <- record$reference_treated - record$reference_control
  }

  if (!is.null(functionals)) {
    functionals <- as.matrix(functionals)
    if (nrow(functionals) != nrow(Q_train)) {
      stop("functionals must have one row per training unit")
    }
    record$functional_names <- colnames(functionals)
    record$functional_treated <- weights_treated %*% functionals
    record$functional_control <- weights_control %*% functionals
    record$functional_effect <- record$functional_treated - record$functional_control
  }

  for (name in names(extra)) {
    record[[name]] <- extra[[name]]
  }
  class(record) <- c("wp2c_prediction", "list")
  record
}

#' Check that arm weight rows are normalized and respect arm separation.
#'
#' @param record A `wp2c_prediction` list.
#' @param treated_index Integer indices of treated training units.
#' @param tolerance Absolute tolerance on the row sums and on the leakage mass.
#' @return Invisible list of the observed maxima.
check_prediction_invariants <- function(record, treated_index, tolerance = 1e-10) {
  n_train <- record$n_train
  is_treated <- logical(n_train)
  is_treated[treated_index] <- TRUE

  row_error_treated <- max(abs(rowSums(record$weights_treated) - 1))
  row_error_control <- max(abs(rowSums(record$weights_control) - 1))
  leak_treated <- max(abs(record$weights_treated[, !is_treated, drop = FALSE]))
  leak_control <- max(abs(record$weights_control[, is_treated, drop = FALSE]))
  negative_mass <- min(c(record$weights_treated, record$weights_control))

  if (row_error_treated > tolerance || row_error_control > tolerance) {
    stop("arm weight rows are not normalized")
  }
  if (leak_treated > tolerance || leak_control > tolerance) {
    stop("arm weights place mass on the opposite treatment arm")
  }
  if (negative_mass < -tolerance) {
    stop("arm weights must be nonnegative before signing")
  }

  invisible(list(
    row_error_treated = row_error_treated,
    row_error_control = row_error_control,
    leak_treated = leak_treated,
    leak_control = leak_control,
    min_weight = negative_mass
  ))
}

#' Draw M equal-weight particles per test point from an arm weight matrix.
#'
#' Gives the fixed-M output shape used by C-WDB so the law-level metrics in
#' G3 can be computed on a common footing. The particles are training quantile
#' vectors resampled with replacement; they carry no cross-arm pairing.
#'
#' @param record A `wp2c_prediction` list.
#' @param Q_train Numeric n_train x K matrix of training quantile vectors.
#' @param arm Either 0 or 1.
#' @param n_particles Number of particles M.
#' @param seed Integer seed.
#' @return Numeric n_test x M x K array.
baseline_particles <- function(record, Q_train, arm, n_particles = 10L, seed = 1L) {
  Q_train <- as.matrix(Q_train)
  weights <- if (identical(as.integer(arm), 1L)) {
    record$weights_treated
  } else {
    record$weights_control
  }
  state <- .Random.seed_snapshot()
  on.exit(.Random.seed_restore(state), add = TRUE)
  set.seed(seed)
  out <- array(0.0, dim = c(nrow(weights), n_particles, ncol(Q_train)))
  for (i in seq_len(nrow(weights))) {
    row <- weights[i, ]
    support <- which(row > 0)
    drawn <- sample(support, n_particles, replace = TRUE, prob = row[support])
    out[i, , ] <- Q_train[drawn, , drop = FALSE]
  }
  out
}
