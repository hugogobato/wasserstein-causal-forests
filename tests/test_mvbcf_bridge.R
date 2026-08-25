#!/usr/bin/env Rscript
#
# WP2-B2 verification for the forced-shared MVBCF bridge.
# Run from the repository root in a clean session:
#
#   Rscript tests/test_mvbcf_bridge.R

source("research/pta_bcf/mvbcf_bridge.R")
suppressPackageStartupMessages(library(jsonlite))

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

small_cell <- function(seed = 3L, n = 120L, n_targets = 3L) {
  set.seed(seed)
  X <- matrix(runif(n * 4), n, 4)
  propensity <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, propensity)
  base <- 2 * X[, 1] + sin(3 * X[, 2])
  effects <- seq_len(n_targets) * 0.5
  Y <- outer(base, rep(1, n_targets)) + outer(Z, effects) +
    matrix(rnorm(n * n_targets, 0, 0.3), n, n_targets)
  list(X = X, Z = Z, Y = Y, effects = effects, propensity = propensity)
}

# --- provenance -----------------------------------------------------------

check(
  "bridge records the pinned upstream revision",
  nchar(MVBCF_PINNED_REVISION) == 40L && MVBCF_PACKAGE_LICENSE == "MIT"
)
check(
  "pinned mvbcf package is installed",
  requireNamespace("mvbcf", quietly = TRUE)
)

# --- draw shapes and schema ----------------------------------------------

cell <- small_cell()
n_draws <- 60L
fit <- mvbcf_fit_predict(
  X_con = cell$X,
  Y = cell$Y,
  Z = cell$Z,
  X_mod = cell$X,
  n_iter = 120L,
  n_burn = 60L,
  n_tree = 20L,
  n_tree_tau = 10L,
  seed = 11L
)

check(
  "tau_train has shape n x D x S",
  identical(dim(fit$tau_train), c(nrow(cell$X), ncol(cell$Y), n_draws))
)
check(
  "mu_test has shape n x D x S",
  identical(dim(fit$mu_test), c(nrow(cell$X), ncol(cell$Y), n_draws))
)
check(
  "residual covariance has shape D x D x S",
  identical(dim(fit$sigma), c(ncol(cell$Y), ncol(cell$Y), n_draws))
)
check("burn-in draws are dropped", fit$n_draws == n_draws)
check("all draws are finite", all(is.finite(fit$tau_train)))
check(
  "posterior mean contrast has shape n x D",
  identical(dim(mvbcf_contrast_mean(fit, "train")), c(nrow(cell$X), ncol(cell$Y)))
)

# --- seed behaviour -------------------------------------------------------

repeat_fit <- mvbcf_fit_predict(
  X_con = cell$X, Y = cell$Y, Z = cell$Z, X_mod = cell$X,
  n_iter = 120L, n_burn = 60L, n_tree = 20L, n_tree_tau = 10L, seed = 11L
)
other_fit <- mvbcf_fit_predict(
  X_con = cell$X, Y = cell$Y, Z = cell$Z, X_mod = cell$X,
  n_iter = 120L, n_burn = 60L, n_tree = 20L, n_tree_tau = 10L, seed = 12L
)
check(
  "same seed reproduces the posterior draws exactly",
  identical(fit$tau_train, repeat_fit$tau_train)
)
check(
  "a different seed changes the posterior draws",
  !identical(fit$tau_train, other_fit$tau_train)
)

# --- forced sharing recovers a common effect ------------------------------

posterior_effects <- colMeans(mvbcf_contrast_mean(fit, "train"))
check(
  "ordered homogeneous effects are recovered in order",
  all(diff(posterior_effects) > 0)
)
check(
  "recovered effects are within 40% of the truth",
  all(abs(posterior_effects - cell$effects) < 0.4 * cell$effects)
)

# The null check needs a sample size where the residual confounded signal is
# smaller than the tolerance; at n = 120 the sampling noise alone exceeds it.
null_cell <- small_cell(seed = 5L, n = 400L)
null_cell$Y <- null_cell$Y - outer(null_cell$Z, null_cell$effects)
null_fit <- mvbcf_fit_predict(
  X_con = null_cell$X, Y = null_cell$Y, Z = null_cell$Z, X_mod = null_cell$X,
  n_iter = 400L, n_burn = 200L, n_tree = 20L, n_tree_tau = 10L, seed = 13L
)
null_effects <- colMeans(mvbcf_contrast_mean(null_fit, "train"))
check(
  "a null treatment produces shrunken contrasts",
  all(abs(null_effects) < 0.15 * apply(null_cell$Y, 2, sd))
)

# --- input contract -------------------------------------------------------

check(
  "non-binary treatment is rejected",
  inherits(
    try(
      mvbcf_fit_predict(
        X_con = cell$X, Y = cell$Y, Z = cell$Z + 0.5, X_mod = cell$X,
        n_iter = 20L, n_burn = 10L, seed = 1L
      ),
      silent = TRUE
    ),
    "try-error"
  )
)
check(
  "burn-in beyond the chain length is rejected",
  inherits(
    try(
      mvbcf_fit_predict(
        X_con = cell$X, Y = cell$Y, Z = cell$Z, X_mod = cell$X,
        n_iter = 20L, n_burn = 20L, seed = 1L
      ),
      silent = TRUE
    ),
    "try-error"
  )
)

# --- binary exchange round trip used by the Python driver -----------------

workspace <- file.path(tempdir(), "mvbcf_bridge_test")
dir.create(workspace, showWarnings = FALSE, recursive = TRUE)
write_array_bin(cell$X, file.path(workspace, "X_con.bin"))
write_array_bin(cell$Y, file.path(workspace, "Y.bin"))
write_array_bin(matrix(cell$Z, ncol = 1), file.path(workspace, "Z.bin"))
write_array_bin(cell$X, file.path(workspace, "X_mod.bin"))
write_array_bin(cell$X, file.path(workspace, "X_con_test.bin"))
write_array_bin(cell$X, file.path(workspace, "X_mod_test.bin"))
spec <- list(
  arrays = list(
    X_con = list(path = file.path(workspace, "X_con.bin"), shape = dim(cell$X)),
    Y = list(path = file.path(workspace, "Y.bin"), shape = dim(cell$Y)),
    Z = list(path = file.path(workspace, "Z.bin"), shape = c(length(cell$Z), 1L)),
    X_mod = list(path = file.path(workspace, "X_mod.bin"), shape = dim(cell$X)),
    X_con_test = list(
      path = file.path(workspace, "X_con_test.bin"), shape = dim(cell$X)
    ),
    X_mod_test = list(
      path = file.path(workspace, "X_mod_test.bin"), shape = dim(cell$X)
    )
  ),
  params = list(
    n_iter = 60L, n_burn = 30L, n_tree = 10L, n_tree_tau = 5L,
    min_nodesize = 1L, sigma_mu_scale = 1, sigma_tau_scale = 0.375, seed = 21L
  ),
  output_dir = file.path(workspace, "out")
)
spec_path <- file.path(workspace, "spec.json")
writeLines(toJSON(spec, auto_unbox = TRUE, digits = 12), spec_path)
meta <- run_fit_command(spec_path)
check(
  "fit command writes the declared arrays",
  all(file.exists(file.path(
    workspace, "out",
    c("tau_train.bin", "tau_test.bin", "mu_train.bin", "mu_test.bin", "sigma.bin")
  )))
)
check("fit command reports the pinned revision", meta$package_revision == MVBCF_PINNED_REVISION)
check("fit command reports the retained draw count", meta$n_draws == 30L)
round_trip <- read_array_bin(
  file.path(workspace, "out", "tau_train.bin"),
  as.integer(meta$shapes$tau_train)
)
check(
  "written draws read back with the declared shape",
  identical(dim(round_trip), c(nrow(cell$X), ncol(cell$Y), 30L))
)

# --- published data generating process ------------------------------------

published <- friedman_homogeneous_cell(300L, 200L, 2L, 7L)
check("published cell splits train and test rows", nrow(published$X_train) == 300L &&
  nrow(published$X_test) == 200L)
check(
  "published cell keeps the signal-to-noise ratio between 1 and 2",
  {
    ratio <- var(published$mu_train[, 1]) / published$sigma^2
    ratio >= 1 && ratio <= 2
  }
)
check(
  "published cell keeps homogeneous effects below 0.3 sd(y)",
  all(abs(published$tau) <= 0.3 * sqrt(var(published$mu_train[, 1]) + published$sigma^2) + 1e-8)
)
check(
  "published cell is deterministic in its seed",
  identical(published$Y_train, friedman_homogeneous_cell(300L, 200L, 2L, 7L)$Y_train)
)

cat(sprintf("\n%d checks, %d failures\n", checks, failures))
if (failures > 0L) {
  quit(status = 1L)
}
