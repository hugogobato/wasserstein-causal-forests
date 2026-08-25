#!/usr/bin/env Rscript
#
# Section 11 DRF comparator from the Causal-DRF paper.
#
# This is the paper's two-separate-DRF benchmark, not the ordinary one-fit
# W-DRF-T adapter used by the original G3 tournament.  The supplied
# drfinference code fits B half-sample forests per arm and averages their
# prediction weights.  With 2500 total trees and 50 trees per group, B=50.
# The returned weights are expanded to the shared training bank so the Python
# evaluator scores this method with the same law and treatment-effect metrics
# as every other law-producing method.

suppressWarnings(suppressMessages({
  library(jsonlite)
  library(Matrix)
  library(drf)
}))

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1L) {
  stop("usage: g3_drf_original_driver.R <workspace>")
}
workspace <- normalizePath(arguments[[1]], mustWork = TRUE)

script_path <- (function() {
  called <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(called)) {
    return(dirname(normalizePath(sub("^--file=", "", called[[1]]))))
  }
  getwd()
})()
repository_root <- normalizePath(file.path(script_path, "..", ".."), mustWork = TRUE)

# This is the supplied implementation used by the Causal-DRF paper's
# separate-forest benchmark.  It includes drfCI(), predictdrf(), and drfown().
source(file.path(repository_root, "code", "drfinference-main", "drf-foo.R"),
       chdir = TRUE)
source(file.path(repository_root, "research", "baselines", "baseline_common.R"),
       chdir = TRUE)

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
  writeBin(as.double(as.vector(values)), file.path(workspace, name), size = 8L,
           endian = .Platform$endian)
}

write_full_arm_weights <- function(prediction, index, n_train) {
  local <- as.matrix(prediction$weights)
  full <- matrix(0.0, nrow = nrow(local), ncol = n_train)
  full[, index] <- local
  full
}

check_weights <- function(weights, arm_index, label) {
  values <- as.matrix(weights)
  if (any(!is.finite(values)) || any(values < -1e-10)) {
    stop(sprintf("%s weights are non-finite or negative", label))
  }
  if (max(abs(rowSums(values) - 1.0)) > 1e-8) {
    stop(sprintf("%s weight rows do not sum to one", label))
  }
  outside <- setdiff(seq_len(ncol(values)), arm_index)
  if (length(outside) && max(abs(values[, outside, drop = FALSE])) > 1e-12) {
    stop(sprintf("%s weights leak across treatment arms", label))
  }
}

spec <- fromJSON(file.path(workspace, "spec.json"), simplifyVector = TRUE)
X_train <- read_array("X_train.bin", c(spec$n_train, spec$n_features))
A_train <- as.integer(round(as.vector(read_array("A_train.bin", c(spec$n_train, 1L)))))
Q_train <- read_array("Q_train.bin", c(spec$n_train, spec$n_grid))
X_test <- read_array("X_test.bin", c(spec$n_test, spec$n_features))

if (!all(A_train %in% c(0L, 1L))) {
  stop("A_train must be binary")
}
control_index <- which(A_train == 0L)
treated_index <- which(A_train == 1L)

num_trees <- as.integer(spec$num_trees)
ci_group_size <- as.integer(spec$ci_group_size)
num_groups <- as.integer(num_trees / ci_group_size)
if (num_groups * ci_group_size != num_trees) {
  stop("num_trees must be divisible by ci_group_size")
}

# drfCI() owns both the half-sample draws and the per-group DRF seeds.  The
# original code relies on the global R stream, so seed it once before fitting
# the control and treated ensembles in their original order.
set.seed(as.integer(spec$seed))
started <- proc.time()[["elapsed"]]
control_fit <- drfCI(
  X = X_train[control_index, , drop = FALSE],
  Y = Q_train[control_index, , drop = FALSE],
  B = num_groups,
  num.trees = ci_group_size,
  num.threads = 1L
)
treated_fit <- drfCI(
  X = X_train[treated_index, , drop = FALSE],
  Y = Q_train[treated_index, , drop = FALSE],
  B = num_groups,
  num.trees = ci_group_size,
  num.threads = 1L
)
fit_seconds <- proc.time()[["elapsed"]] - started

control_prediction <- predictdrf(control_fit, x = X_test)
treated_prediction <- predictdrf(treated_fit, x = X_test)
weights_control <- write_full_arm_weights(
  control_prediction, control_index, spec$n_train
)
weights_treated <- write_full_arm_weights(
  treated_prediction, treated_index, spec$n_train
)
check_weights(weights_control, control_index, "control")
check_weights(weights_treated, treated_index, "treated")

total_seconds <- proc.time()[["elapsed"]] - started
write_array(weights_control, "weights_control.bin")
write_array(weights_treated, "weights_treated.bin")

record <- list(
  method = "drf",
  implementation = "code/drfinference-main/drf-foo.R",
  n_train = spec$n_train,
  n_test = spec$n_test,
  n_grid = spec$n_grid,
  num_trees = num_trees,
  ci_group_size = ci_group_size,
  num_groups = num_groups,
  response_scaling = TRUE,
  fit_seconds = unname(fit_seconds),
  total_seconds = unname(total_seconds),
  peak_ram_mb = unname(sum(gc()[, 6L])),
  status = "ok"
)
write(toJSON(record, auto_unbox = TRUE, digits = 12),
      file.path(workspace, "result.json"))
