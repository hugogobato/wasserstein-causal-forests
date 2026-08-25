#!/usr/bin/env Rscript
#
# G3 driver for the authors' Causal-DRF implementation.
#
# The model fit call below follows the supplied
# code/causal_drf_paper-main/R/simulation_study_setup.R exactly: the causal-clean
# drf package is called with W, response.scaling = FALSE, 2500 trees, and 50
# confidence groups. The package's prediction weights are exported instead of
# calling the paper script's witness-only wrapper, because the common G3 scorer
# needs the arm laws on the shared evaluation bank.

suppressWarnings(suppressMessages({
  library(jsonlite)
}))

causal_drf_library <- Sys.getenv("WCF_CAUSAL_DRF_R_LIB", unset = "")
if (nzchar(causal_drf_library)) {
  .libPaths(c(causal_drf_library, .libPaths()))
}
suppressPackageStartupMessages(library(drf))

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1L) {
  stop("usage: g3_causal_drf_original_driver.R <workspace>")
}
workspace <- normalizePath(arguments[[1]], mustWork = TRUE)

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
  writeBin(
    as.double(as.vector(values)), file.path(workspace, name), size = 8L,
    endian = .Platform$endian
  )
}

spec <- fromJSON(file.path(workspace, "spec.json"), simplifyVector = TRUE)
X_train <- read_array("X_train.bin", c(spec$n_train, spec$n_features))
A_train <- as.integer(round(as.vector(
  read_array("A_train.bin", c(spec$n_train, 1L))
)))
Q_train <- read_array("Q_train.bin", c(spec$n_train, spec$n_grid))
X_test <- read_array("X_test.bin", c(spec$n_test, spec$n_features))

num_trees <- as.integer(spec$num_trees)
ci_group_size <- as.integer(spec$ci_group_size)
if (num_trees < 1L || ci_group_size < 2L || ci_group_size > num_trees) {
  stop("invalid original Causal-DRF tree or confidence-group budget")
}

started <- proc.time()[["elapsed"]]
fit <- drf(
  X = X_train,
  Y = Q_train,
  W = matrix(A_train, ncol = 1L),
  num.trees = num_trees,
  ci.group.size = ci_group_size,
  num.threads = 1L,
  response.scaling = FALSE,
  seed = as.integer(spec$seed)
)
fit_seconds <- proc.time()[["elapsed"]] - started

predict_arm <- function(arm) {
  prediction <- predict(
    fit,
    newdata = X_test,
    newtreatment = matrix(rep(arm, spec$n_test), ncol = 1L),
    num.threads = 1L,
    bootstrap = FALSE
  )
  weights <- as.matrix(prediction$weights)
  if (!all(dim(weights) == c(spec$n_test, spec$n_train))) {
    stop(sprintf("arm %d weights have unexpected dimensions", arm))
  }
  if (any(!is.finite(weights)) || any(weights < -1e-12)) {
    stop(sprintf("arm %d weights are not finite and nonnegative", arm))
  }
  if (max(abs(rowSums(weights) - 1)) > 1e-8) {
    stop(sprintf("arm %d weights are not normalized", arm))
  }
  if (any(abs(weights[, A_train != arm, drop = FALSE]) > 1e-10)) {
    stop(sprintf("arm %d weights leak across treatment arms", arm))
  }
  weights
}

weights_control <- predict_arm(0L)
weights_treated <- predict_arm(1L)
write_array(weights_control, "weights_control.bin")
write_array(weights_treated, "weights_treated.bin")

total_seconds <- proc.time()[["elapsed"]] - started
peak_ram_mb <- 0.0
status_lines <- tryCatch(
  readLines("/proc/self/status"),
  error = function(error) character()
)
hwm <- grep("^VmHWM:", status_lines, value = TRUE)
if (length(hwm) == 1L) {
  peak_ram_mb <- as.numeric(sub("^VmHWM:\\s*([0-9]+) kB$", "\\1", hwm)) / 1024
}

write_json(
  list(
    fit_seconds = unname(fit_seconds),
    total_seconds = unname(total_seconds),
    peak_ram_mb = unname(peak_ram_mb),
    implementation = "code/causal_drf_paper-main/R/simulation_study_setup.R",
    drf_branch = "herbps10/drf causal-clean",
    drf_commit = "0a1a508444176b5b1553f13e832be93a374b0af2",
    num_trees = num_trees,
    ci_group_size = ci_group_size
  ),
  file.path(workspace, "result.json"), auto_unbox = TRUE, pretty = TRUE
)
