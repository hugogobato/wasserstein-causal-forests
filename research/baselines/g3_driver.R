#!/usr/bin/env Rscript
# Phase G3 driver for the two R forest baselines.
#
# Invoked once per tournament cell by `wasserstein_causal_forests.g3.r_bridge`.
# Reads a workspace of flat column-major double arrays, fits one baseline, and
# writes back the two arm weight matrices over the shared training bank plus a
# JSON record of timing and hyperparameters. The Python side turns those
# weights into a `LawPrediction`, so both baselines and C-WDB are scored by one
# metric implementation.
#
# Usage:
#   Rscript research/baselines/g3_driver.R <workspace> <method>
#
# `method` is `wdrft` or `causal_drf`. The workspace must contain X_train.bin,
# A_train.bin, Q_train.bin, X_test.bin, quad_weights.bin, reference.bin, and
# spec.json.

suppressWarnings(suppressMessages({
  library(jsonlite)
}))

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 2L) {
  stop("usage: g3_driver.R <workspace> <method>")
}
workspace <- normalizePath(arguments[[1]], mustWork = TRUE)
method <- arguments[[2]]

script_path <- (function() {
  called <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(called)) {
    return(dirname(normalizePath(sub("^--file=", "", called[[1]]))))
  }
  getwd()
})()

source(file.path(script_path, "baseline_common.R"), chdir = TRUE)
if (method == "wdrft") {
  source(file.path(script_path, "drf_tlearner.R"), chdir = TRUE)
} else if (method == "causal_drf") {
  source(file.path(script_path, "causal_drf_r", "causal_drf.R"), chdir = TRUE)
} else {
  stop(sprintf("unknown method '%s'", method))
}

read_array <- function(name, shape) {
  values <- readBin(
    file.path(workspace, name), what = "double",
    n = prod(shape), size = 8L, endian = .Platform$endian
  )
  if (length(values) != prod(shape)) {
    stop(sprintf("%s holds %d values, expected %d", name, length(values), prod(shape)))
  }
  array(values, dim = shape)
}

write_array <- function(values, name) {
  # Column-major, matching numpy's order="F" convention on the Python side.
  writeBin(as.double(as.vector(values)), file.path(workspace, name), size = 8L,
           endian = .Platform$endian)
}

spec <- fromJSON(file.path(workspace, "spec.json"), simplifyVector = TRUE)

X_train <- read_array("X_train.bin", c(spec$n_train, spec$n_features))
A_train <- as.integer(round(as.vector(read_array("A_train.bin", c(spec$n_train, 1L)))))
Q_train <- read_array("Q_train.bin", c(spec$n_train, spec$n_grid))
X_test <- read_array("X_test.bin", c(spec$n_test, spec$n_features))
quad_weights <- as.vector(read_array("quad_weights.bin", c(spec$n_grid, 1L)))
reference <- as.vector(read_array("reference.bin", c(spec$n_grid, 1L)))
# Grid functionals are not requested here. `baseline_prediction` would expect a
# matrix of h_j values per training unit, and the Python side already computes
# those expectations from the returned weights, so asking R for them would give
# two implementations of one definition.

started <- proc.time()[["elapsed"]]
if (method == "wdrft") {
  fit <- wdrft_fit(
    X = X_train, A = A_train, Q = Q_train, quad_weights = quad_weights,
    num_trees = as.integer(spec$num_trees),
    min_node_size = as.integer(spec$min_node_size),
    seed = as.integer(spec$seed)
  )
  fit_seconds <- proc.time()[["elapsed"]] - started
  record <- wdrft_predict(fit, X_test, reference_quantiles = reference)
} else {
  fit <- causal_drf_fit(
    X = X_train, A = A_train, Y = Q_train,
    num_trees = as.integer(spec$num_trees),
    min_arm_leaf = as.integer(spec$min_arm_leaf),
    # Frozen by the Phase G3 computational notes: the drf-package rule is not
    # scale equivariant and would tie the kernel to arbitrary outcome units.
    bandwidth_rule = "median_distance",
    seed = as.integer(spec$seed)
  )
  fit_seconds <- proc.time()[["elapsed"]] - started
  record <- causal_drf_predict(fit, X_test, Q_train = Q_train,
                               quad_weights = quad_weights,
                               reference_quantiles = reference)
}
total_seconds <- proc.time()[["elapsed"]] - started

check_prediction_invariants(record, which(A_train == 1L))
write_array(record$weights_control, "weights_control.bin")
write_array(record$weights_treated, "weights_treated.bin")

result <- list(
  method = method,
  schema_version = BASELINE_SCHEMA_VERSION,
  n_train = spec$n_train,
  n_test = spec$n_test,
  n_grid = spec$n_grid,
  fit_seconds = unname(fit_seconds),
  total_seconds = unname(total_seconds),
  # gc() returns max-used in cells on columns 5 and in megabytes on column 6.
  # Summing the cell counts and scaling them by hand gives nonsense, because
  # Ncells and Vcells are different sizes; take the megabyte columns directly.
  peak_ram_mb = unname(sum(gc()[, 6L])),
  status = "ok"
)
write(toJSON(result, auto_unbox = TRUE, digits = 12),
      file.path(workspace, "result.json"))
