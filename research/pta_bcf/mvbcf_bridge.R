#!/usr/bin/env Rscript
#
# WP2-B2. Forced-shared multivariate BCF baseline (PTA-F).
#
# This bridge wraps the official MIT-licensed `mvbcf` package by McJames,
# Parnell, Goh and O'Shea behind a fixed target-matrix interface. The package
# is the licensed distribution of the `fast_bart` sampler used in the paper
# replication scripts; the replication repository itself carries no license,
# so it is deliberately not sourced here.
#
# Forced sharing means every target coordinate is fitted on one common tree
# partition with vector leaf parameters. It is a published baseline, not a
# contribution of this project.

MVBCF_BRIDGE_ID <- "PTA-F-MVBCF-BRIDGE-v1"
MVBCF_PACKAGE_SOURCE <- "github.com/Nathan-McJames/mvbcf"
MVBCF_PINNED_REVISION <- "fc3b89b0a78ce8a31ae75c43a6ec75f1945ca0c8"
MVBCF_PACKAGE_LICENSE <- "MIT"

suppressPackageStartupMessages({
  library(mvbcf)
})

# ---------------------------------------------------------------------------
# Fixed target-matrix interface
# ---------------------------------------------------------------------------

#' Fit forced-shared MVBCF and return posterior draws in the common schema.
#'
#' @param X_con Numeric matrix of prognostic covariates, including the
#'   estimated propensity column when one is used.
#' @param Y Numeric target matrix with one column per PTA coordinate.
#' @param Z Numeric binary treatment vector shared by every coordinate.
#' @param X_mod Numeric matrix of effect moderators.
#' @return List with tau_train, tau_test, mu_train, mu_test (n x D x S arrays),
#'   sigma (D x D x S), and the manifest fields describing the run.
mvbcf_fit_predict <- function(X_con,
                              Y,
                              Z,
                              X_mod,
                              X_con_test = X_con,
                              X_mod_test = X_mod,
                              n_iter = 1000,
                              n_burn = 500,
                              n_tree = 50,
                              n_tree_tau = 20,
                              min_nodesize = 1,
                              sigma_mu_scale = 1,
                              sigma_tau_scale = 0.375,
                              alpha = 0.95,
                              beta = 2,
                              alpha_tau = 0.25,
                              beta_tau = 3,
                              seed = 1L) {
  X_con <- as.matrix(X_con)
  X_mod <- as.matrix(X_mod)
  X_con_test <- as.matrix(X_con_test)
  X_mod_test <- as.matrix(X_mod_test)
  Y <- as.matrix(Y)
  Z <- as.numeric(Z)

  stopifnot("n_burn must be smaller than n_iter" = n_burn < n_iter)
  stopifnot("n_burn must be nonnegative" = n_burn >= 0)
  stopifnot("Z must be binary" = all(Z %in% c(0, 1)))
  stopifnot("Y must have at least two rows" = nrow(Y) >= 2)

  n_targets <- ncol(Y)
  set.seed(seed)
  started <- Sys.time()
  # The sampler streams a progress counter to stdout; the bridge is used
  # programmatically, so that stream is captured and discarded.
  invisible(utils::capture.output(
    fitted <- mvbcf::run_mvbcf(
      X_con = X_con,
      y = Y,
      Z = Z,
      X_mod = X_mod,
      X_con_test = X_con_test,
      X_mod_test = X_mod_test,
      alpha = alpha,
      beta = beta,
      alpha_tau = alpha_tau,
      beta_tau = beta_tau,
      sigma_mu = diag(sigma_mu_scale^2 / n_tree, n_targets),
      sigma_tau = diag(sigma_tau_scale^2 / n_tree_tau, n_targets),
      v_0 = n_targets + 2,
      sigma_0 = diag(1, n_targets),
      n_iter = n_iter,
      n_tree = n_tree,
      n_tree_tau = n_tree_tau,
      min_nodesize = min_nodesize
    ),
    type = "output"
  ))
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))

  keep <- seq.int(n_burn + 1L, n_iter)
  list(
    tau_train = fitted$predictions_tau[, , keep, drop = FALSE],
    tau_test = fitted$predictions_tau_test[, , keep, drop = FALSE],
    mu_train = fitted$predictions[, , keep, drop = FALSE],
    mu_test = fitted$predictions_test[, , keep, drop = FALSE],
    sigma = fitted$sigmas[, , keep, drop = FALSE],
    bridge_id = MVBCF_BRIDGE_ID,
    package_revision = MVBCF_PINNED_REVISION,
    n_iter = n_iter,
    n_burn = n_burn,
    n_draws = length(keep),
    n_targets = n_targets,
    seed = seed,
    elapsed_seconds = elapsed
  )
}

#' Posterior mean of the conditional target contrast, shape n x D.
mvbcf_contrast_mean <- function(fit, which = "test") {
  draws <- if (which == "test") fit$tau_test else fit$tau_train
  apply(draws, c(1, 2), mean)
}

#' Posterior mean of the control-arm target surface, shape n x D.
mvbcf_control_mean <- function(fit, which = "test") {
  draws <- if (which == "test") fit$mu_test else fit$mu_train
  apply(draws, c(1, 2), mean)
}

# ---------------------------------------------------------------------------
# Binary array exchange used by the Python driver
# ---------------------------------------------------------------------------

read_array_bin <- function(path, shape) {
  values <- readBin(path, what = "double", n = prod(shape), size = 8)
  stopifnot("truncated input array" = length(values) == prod(shape))
  array(values, dim = shape)
}

write_array_bin <- function(array_value, path) {
  connection <- file(path, "wb")
  on.exit(close(connection))
  writeBin(as.double(as.vector(array_value)), connection, size = 8)
  invisible(dim(array_value))
}

run_fit_command <- function(spec_path) {
  spec <- jsonlite::fromJSON(spec_path, simplifyVector = TRUE)
  arrays <- list()
  for (name in names(spec$arrays)) {
    entry <- spec$arrays[[name]]
    arrays[[name]] <- read_array_bin(entry$path, as.integer(entry$shape))
  }
  params <- spec$params
  fit <- mvbcf_fit_predict(
    X_con = arrays$X_con,
    Y = arrays$Y,
    Z = as.numeric(arrays$Z),
    X_mod = arrays$X_mod,
    X_con_test = arrays$X_con_test,
    X_mod_test = arrays$X_mod_test,
    n_iter = as.integer(params$n_iter),
    n_burn = as.integer(params$n_burn),
    n_tree = as.integer(params$n_tree),
    n_tree_tau = as.integer(params$n_tree_tau),
    min_nodesize = as.integer(params$min_nodesize),
    sigma_mu_scale = as.numeric(params$sigma_mu_scale),
    sigma_tau_scale = as.numeric(params$sigma_tau_scale),
    seed = as.integer(params$seed)
  )
  output_dir <- spec$output_dir
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  shapes <- list()
  for (name in c("tau_train", "tau_test", "mu_train", "mu_test", "sigma")) {
    shapes[[name]] <- write_array_bin(
      fit[[name]], file.path(output_dir, paste0(name, ".bin"))
    )
  }
  meta <- list(
    bridge_id = fit$bridge_id,
    package_revision = fit$package_revision,
    package_license = MVBCF_PACKAGE_LICENSE,
    n_draws = fit$n_draws,
    n_iter = fit$n_iter,
    n_burn = fit$n_burn,
    n_targets = fit$n_targets,
    seed = fit$seed,
    elapsed_seconds = fit$elapsed_seconds,
    shapes = shapes
  )
  writeLines(
    jsonlite::toJSON(meta, auto_unbox = TRUE, digits = 12),
    file.path(output_dir, "meta.json")
  )
  invisible(meta)
}

# ---------------------------------------------------------------------------
# Published-cell reproduction (McJames et al., homogeneous treatment effect)
# ---------------------------------------------------------------------------

#' Homogeneous-effect cell of the published simulation study.
#'
#' The prognostic surface is the first Friedman function, the noise variance
#' is drawn so the signal-to-noise ratio lies between 2:1 and 1:1, and the
#' homogeneous effects have magnitude below 0.3 sd(y).
#'
#' Two documented deviations from the paper are recorded in the report:
#' the paper writes P(Z = 1) proportional to mu without fixing the
#' proportionality map, which is resolved here as a linear rescaling onto
#' [0.05, 0.95]; and the licensed `mvbcf` package accepts a single shared
#' treatment vector, so the paper's outcome-specific assignments Z1 and Z2 are
#' replaced by one shared assignment.
friedman_homogeneous_cell <- function(n_train = 500,
                                      n_test = 500,
                                      n_targets = 2,
                                      seed = 1L) {
  set.seed(seed)
  n_total <- n_train + n_test
  X <- matrix(runif(n_total * 10), nrow = n_total, ncol = 10)
  mu <- 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 +
    10 * X[, 4] + 5 * X[, 5]

  signal_to_noise <- runif(1, 1, 2)
  sigma <- sqrt(var(mu) / signal_to_noise)
  noise <- matrix(rnorm(n_total * n_targets, 0, sigma), n_total, n_targets)

  outcome_sd <- sqrt(var(mu) + sigma^2)
  tau <- runif(n_targets, -0.3, 0.3) * outcome_sd

  propensity <- 0.05 + 0.9 * (mu - min(mu)) / (max(mu) - min(mu))
  Z <- rbinom(n_total, 1, propensity)

  Mu <- matrix(rep(mu, n_targets), ncol = n_targets)
  Y <- Mu + outer(Z, tau) + noise

  train <- seq_len(n_train)
  test <- seq.int(n_train + 1L, n_total)
  list(
    X_train = X[train, , drop = FALSE],
    X_test = X[test, , drop = FALSE],
    Y_train = Y[train, , drop = FALSE],
    Y_test = Y[test, , drop = FALSE],
    Z_train = Z[train],
    Z_test = Z[test],
    mu_train = Mu[train, , drop = FALSE],
    mu_test = Mu[test, , drop = FALSE],
    tau = tau,
    sigma = sigma,
    propensity_train = propensity[train],
    propensity_test = propensity[test]
  )
}

#' Estimated propensity used as a prognostic covariate.
#'
#' The published scripts estimate the propensity with a probit BART using
#' k = 3, so `method = "bart"` reproduces the paper. The logistic fallback
#' exists only for environments without dbarts and is recorded in the result
#' rows, because a misspecified propensity inflates the treatment-effect error
#' of every model in the comparison.
estimate_propensity <- function(cell, method = "bart", seed = 1L) {
  if (method == "bart" && requireNamespace("dbarts", quietly = TRUE)) {
    set.seed(seed)
    invisible(utils::capture.output(
      fitted <- dbarts::bart(
        x.train = cell$X_train,
        y.train = cell$Z_train,
        x.test = cell$X_test,
        k = 3,
        verbose = FALSE
      ),
      type = "output"
    ))
    return(list(
      train = as.numeric(colMeans(pnorm(fitted$yhat.train))),
      test = as.numeric(colMeans(pnorm(fitted$yhat.test))),
      method = "bart"
    ))
  }
  design_train <- as.data.frame(cell$X_train)
  names(design_train) <- paste0("V", seq_len(ncol(cell$X_train)))
  design_test <- as.data.frame(cell$X_test)
  names(design_test) <- names(design_train)
  fitted <- glm(
    Z ~ .,
    data = cbind(design_train, Z = cell$Z_train),
    family = binomial()
  )
  list(
    train = as.numeric(predict(fitted, type = "response")),
    test = as.numeric(predict(fitted, newdata = design_test, type = "response")),
    method = "logistic"
  )
}

reproduce_published_cell <- function(seed,
                                     n_train = 500,
                                     n_test = 500,
                                     n_iter = 1000,
                                     n_burn = 500,
                                     n_tree = 50,
                                     n_tree_tau = 20,
                                     with_univariate = TRUE,
                                     propensity_method = "bart") {
  cell <- friedman_homogeneous_cell(n_train, n_test, 2L, seed)

  # Propensity is estimated once and shared by every fitted model.
  scores <- estimate_propensity(cell, propensity_method, seed)
  pi_train <- scores$train
  pi_test <- scores$test

  X_con <- cbind(cell$X_train, pi_train)
  X_con_test <- cbind(cell$X_test, pi_test)

  fit <- mvbcf_fit_predict(
    X_con = X_con,
    Y = cell$Y_train,
    Z = cell$Z_train,
    X_mod = cell$X_train,
    X_con_test = X_con_test,
    X_mod_test = cell$X_test,
    n_iter = n_iter,
    n_burn = n_burn,
    n_tree = n_tree,
    n_tree_tau = n_tree_tau,
    seed = seed
  )
  tau_hat <- mvbcf_contrast_mean(fit, "test")
  mu_hat <- mvbcf_control_mean(fit, "test")
  y_hat <- mu_hat + cell$Z_test * tau_hat

  rows <- list()
  for (target in seq_len(2)) {
    rows[[length(rows) + 1L]] <- data.frame(
      method = "MVBCF",
      seed = seed,
      target = target,
      rmse_mu = sqrt(mean((mu_hat[, target] - cell$mu_test[, target])^2)),
      pehe_tau = sqrt(mean((tau_hat[, target] - cell$tau[target])^2)),
      rmse_y = sqrt(mean((y_hat[, target] - cell$Y_test[, target])^2)),
      runtime_seconds = fit$elapsed_seconds,
      propensity_method = scores$method,
      stringsAsFactors = FALSE
    )
  }

  if (with_univariate) {
    for (target in seq_len(2)) {
      started <- Sys.time()
      # Each fit needs its own tree directory: replicates run in parallel.
      tree_dir <- file.path(
        tempdir(), sprintf("bcf_seed%d_target%d", seed, target)
      )
      dir.create(tree_dir, showWarnings = FALSE, recursive = TRUE)
      invisible(utils::capture.output(
        univariate <- bcf::bcf(
          y = cell$Y_train[, target],
          z = cell$Z_train,
          x_control = X_con,
          x_moderate = cell$X_train,
          pihat = pi_train,
          nburn = n_burn,
          nsim = n_iter - n_burn,
          ntree_control = n_tree,
          ntree_moderate = n_tree_tau,
          n_chains = 1,
          n_threads = 1,
          include_pi = "control",
          random_seed = seed,
          verbose = FALSE,
          save_tree_directory = tree_dir,
          log_file = file.path(tree_dir, "bcf.log")
        ),
        type = "output"
      ))
      invisible(utils::capture.output(
        prediction <- predict(
          object = univariate,
          x_predict_control = X_con_test,
          x_predict_moderate = cell$X_test,
          pi_pred = pi_test,
          z_pred = cell$Z_test,
          n_cores = 1,
          save_tree_directory = tree_dir,
          log_file = file.path(tree_dir, "bcf_predict.log")
        ),
        type = "output"
      ))
      univariate_elapsed <- as.numeric(
        difftime(Sys.time(), started, units = "secs")
      )
      tau_uni <- colMeans(prediction$tau)
      mu_uni <- colMeans(prediction$mu)
      y_uni <- mu_uni + cell$Z_test * tau_uni
      rows[[length(rows) + 1L]] <- data.frame(
        method = "BCF-univariate",
        seed = seed,
        target = target,
        rmse_mu = sqrt(mean((mu_uni - cell$mu_test[, target])^2)),
        pehe_tau = sqrt(mean((tau_uni - cell$tau[target])^2)),
        rmse_y = sqrt(mean((y_uni - cell$Y_test[, target])^2)),
        runtime_seconds = univariate_elapsed,
        propensity_method = scores$method,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

run_reproduction_command <- function(output_path,
                                     replicates = 30L,
                                     workers = 4L,
                                     with_univariate = TRUE) {
  seeds <- seq_len(replicates)
  results <- parallel::mclapply(
    seeds,
    function(seed) {
      tryCatch(
        reproduce_published_cell(seed, with_univariate = with_univariate),
        error = function(e) {
          data.frame(
            method = "FAILED", seed = seed, target = NA_integer_,
            rmse_mu = NA_real_, pehe_tau = NA_real_, rmse_y = NA_real_,
            runtime_seconds = NA_real_, propensity_method = NA_character_,
            stringsAsFactors = FALSE
          )
        }
      )
    },
    mc.cores = workers
  )
  frame <- do.call(rbind, results)
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  write.csv(frame, output_path, row.names = FALSE)
  invisible(frame)
}

# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------

main <- function(arguments) {
  if (length(arguments) == 0L) {
    stop("usage: mvbcf_bridge.R fit <spec.json> | reproduce <out.csv> [n] [workers]")
  }
  command <- arguments[[1L]]
  if (command == "fit") {
    run_fit_command(arguments[[2L]])
  } else if (command == "reproduce") {
    replicates <- if (length(arguments) >= 3L) as.integer(arguments[[3L]]) else 30L
    workers <- if (length(arguments) >= 4L) as.integer(arguments[[4L]]) else 4L
    run_reproduction_command(arguments[[2L]], replicates, workers)
  } else {
    stop(sprintf("unknown command '%s'", command))
  }
  invisible(NULL)
}

if (sys.nframe() == 0L && !interactive()) {
  suppressPackageStartupMessages(library(jsonlite))
  main(commandArgs(trailingOnly = TRUE))
}
