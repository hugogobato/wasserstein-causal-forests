`.sourceCpp_1_DLLInfo` <- dyn.load('/home/hugo_souto/Stuff/Research/Wasserstein_Causal_Forests/results/rcpp_cache/sourceCpp-x86_64-pc-linux-gnu-1.1.1/sourcecpp_32d26624cb82e/sourceCpp_2.so')

causal_drf_grow_tree <- Rcpp:::sourceCppFunction(function(X, W, Phi, build_rows, pop_rows, mtry, min_arm_build, min_arm_pop, alpha, max_depth) {}, FALSE, `.sourceCpp_1_DLLInfo`, 'sourceCpp_1_causal_drf_grow_tree')
causal_drf_predict_weights <- Rcpp:::sourceCppFunction(function(trees, tree_group, n_groups, X_test, W, n_train, with_groups) {}, FALSE, `.sourceCpp_1_DLLInfo`, 'sourceCpp_1_causal_drf_predict_weights')

rm(`.sourceCpp_1_DLLInfo`)
