// WP2-C2. Tree builder for the faithful shared Causal-DRF reimplementation.
//
// Näf, Park, Susmann (2026), "Causal-DRF: Conditional Kernel Treatment Effect
// Estimation using Distributional Random Forest", AISTATS 2026 (PMLR v300),
// arXiv:2411.08778v2. The authors published no code; this file is written from
// the paper and shares no source with the GPL-3.0 `drf` package.
//
// What lives here is exactly the part that must be fast: the recursive search
// over candidate splits under the treatment-aware weighted-MMD criterion of
// eq. (C.1), together with the honest routing of the leaf-populating sample.
// Forest assembly, prediction weights, and inference are in `causal_drf.R`.
//
// Criterion, paper eq. (C.1) with the random Fourier approximation eq. (C.2):
//
//   (|I_L| |I_R| / (|I_L| + |I_R|)^2) * (1/S) *
//     sum_b | sum_{i in I_L} nu_{i,L} phi_b(Y_i)
//            - sum_{i in I_R} nu_{i,R} phi_b(Y_i) |^2
//
// with the treatment-aware signed weights
//
//   nu_{i,L} = W_i / |{i in I_L : W_i = 1}| - (1 - W_i) / |{i in I_L : W_i = 0}|
//
// and phi_b(y) = exp(i omega_b' y), omega_b ~ N_d(0, sigma^-2 I_d). The real
// and imaginary parts arrive pre-evaluated in `Phi` so the criterion never
// touches the response dimension d; this is what keeps the cost independent of
// the quantile-grid size K.
//
// Every arm normalization above is undefined when a child holds no unit of one
// arm. Enforcing both-arm minimums is therefore a correctness requirement, not
// a regularization choice.

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace Rcpp;

namespace {

// Order a set of row indices by one covariate column.
void order_by_column(const NumericMatrix& X,
                     int column,
                     std::vector<int>& index) {
  const double* values = &X(0, column);
  std::sort(index.begin(), index.end(),
            [&](int a, int b) { return values[a] < values[b]; });
}

struct SplitCandidate {
  bool found = false;
  int variable = -1;
  double value = 0.0;
  double criterion = -1.0;
};

// Search one covariate for the best admissible split of the current node.
//
// `build_order` and `pop_order` hold the node's build and populate rows sorted
// by covariate `variable`. The build rows drive the criterion; the populate
// rows are walked in parallel so that the paper's per-arm leaf minimums and
// alpha-regularity, which (F4) states for the leaf-populating sample, can be
// checked on the sample they are stated for.
void search_variable(const NumericMatrix& X,
                     const IntegerVector& W,
                     const NumericMatrix& Phi,
                     int variable,
                     const std::vector<int>& build_order,
                     const std::vector<int>& pop_order,
                     int min_arm_build,
                     int min_arm_pop,
                     double alpha,
                     SplitCandidate& best) {
  const int n_build = static_cast<int>(build_order.size());
  const int n_pop = static_cast<int>(pop_order.size());
  const int n_features = Phi.ncol();  // 2 * S: real parts then imaginary parts

  int total_treated = 0;
  std::vector<double> total_treated_sum(n_features, 0.0);
  std::vector<double> total_control_sum(n_features, 0.0);
  for (int position = 0; position < n_build; ++position) {
    const int row = build_order[position];
    if (W[row] == 1) {
      ++total_treated;
      for (int f = 0; f < n_features; ++f) total_treated_sum[f] += Phi(row, f);
    } else {
      for (int f = 0; f < n_features; ++f) total_control_sum[f] += Phi(row, f);
    }
  }
  const int total_control = n_build - total_treated;
  if (total_treated < 2 * min_arm_build || total_control < 2 * min_arm_build) return;

  int total_treated_pop = 0;
  for (int position = 0; position < n_pop; ++position) {
    if (W[pop_order[position]] == 1) ++total_treated_pop;
  }
  const int total_control_pop = n_pop - total_treated_pop;
  if (total_treated_pop < 2 * min_arm_pop || total_control_pop < 2 * min_arm_pop) return;

  const int min_side_build =
      std::max(1, static_cast<int>(std::ceil(alpha * n_build)));
  const int min_side_pop =
      std::max(1, static_cast<int>(std::ceil(alpha * n_pop)));

  std::vector<double> left_treated_sum(n_features, 0.0);
  std::vector<double> left_control_sum(n_features, 0.0);
  int left_treated = 0;
  int left_count = 0;
  int left_treated_pop = 0;
  int left_pop_count = 0;
  int pop_cursor = 0;

  for (int position = 0; position < n_build - 1; ++position) {
    const int row = build_order[position];
    if (W[row] == 1) {
      ++left_treated;
      for (int f = 0; f < n_features; ++f) left_treated_sum[f] += Phi(row, f);
    } else {
      for (int f = 0; f < n_features; ++f) left_control_sum[f] += Phi(row, f);
    }
    ++left_count;

    const double value = X(row, variable);
    // Only realized values are candidate split points, and ties must stay on
    // the same side, so skip positions that do not close a run of equal values.
    if (X(build_order[position + 1], variable) <= value) continue;

    while (pop_cursor < n_pop && X(pop_order[pop_cursor], variable) <= value) {
      if (W[pop_order[pop_cursor]] == 1) ++left_treated_pop;
      ++left_pop_count;
      ++pop_cursor;
    }

    const int right_count = n_build - left_count;
    const int left_control = left_count - left_treated;
    const int right_treated = total_treated - left_treated;
    const int right_control = total_control - left_control;
    if (left_treated < min_arm_build || left_control < min_arm_build) continue;
    if (right_treated < min_arm_build || right_control < min_arm_build) continue;
    if (left_count < min_side_build || right_count < min_side_build) continue;

    const int left_control_pop = left_pop_count - left_treated_pop;
    const int right_pop_count = n_pop - left_pop_count;
    const int right_treated_pop = total_treated_pop - left_treated_pop;
    const int right_control_pop = total_control_pop - left_control_pop;
    if (left_treated_pop < min_arm_pop || left_control_pop < min_arm_pop) continue;
    if (right_treated_pop < min_arm_pop || right_control_pop < min_arm_pop) continue;
    if (left_pop_count < min_side_pop || right_pop_count < min_side_pop) continue;

    double accumulated = 0.0;
    for (int f = 0; f < n_features; ++f) {
      const double left_part = left_treated_sum[f] / left_treated -
                               left_control_sum[f] / left_control;
      const double right_part =
          (total_treated_sum[f] - left_treated_sum[f]) / right_treated -
          (total_control_sum[f] - left_control_sum[f]) / right_control;
      const double gap = left_part - right_part;
      accumulated += gap * gap;
    }
    // n_features counts real and imaginary parts, so dividing by it averages
    // over the S frequencies as eq. (C.2) requires (each contributes |.|^2).
    const double weight = static_cast<double>(left_count) *
                          static_cast<double>(right_count) /
                          (static_cast<double>(n_build) * static_cast<double>(n_build));
    const double criterion = weight * accumulated * 2.0 / n_features;

    if (criterion > best.criterion) {
      best.found = true;
      best.variable = variable;
      best.value = value;
      best.criterion = criterion;
    }
  }
}

struct TreeBuffers {
  std::vector<int> split_var;
  std::vector<double> split_value;
  std::vector<int> left_child;
  std::vector<int> right_child;
  std::vector<std::vector<int> > leaf_pop;
};

int add_node(TreeBuffers& tree) {
  tree.split_var.push_back(-1);
  tree.split_value.push_back(0.0);
  tree.left_child.push_back(-1);
  tree.right_child.push_back(-1);
  tree.leaf_pop.push_back(std::vector<int>());
  return static_cast<int>(tree.split_var.size()) - 1;
}

// Node membership, held once per covariate in that covariate's sort order.
// Carrying the orders down the recursion and partitioning them stably at each
// split costs O(p * m) per node, instead of re-sorting every candidate
// covariate at every node. The split search then reads a ready-sorted vector.
typedef std::vector<std::vector<int> > SortedRows;

void partition_sorted(const NumericMatrix& X,
                      int variable,
                      double value,
                      const SortedRows& parent,
                      SortedRows& left,
                      SortedRows& right) {
  const int n_covariates = static_cast<int>(parent.size());
  left.assign(n_covariates, std::vector<int>());
  right.assign(n_covariates, std::vector<int>());
  for (int j = 0; j < n_covariates; ++j) {
    left[j].reserve(parent[j].size());
    right[j].reserve(parent[j].size());
    for (std::size_t k = 0; k < parent[j].size(); ++k) {
      const int row = parent[j][k];
      if (X(row, variable) <= value) {
        left[j].push_back(row);
      } else {
        right[j].push_back(row);
      }
    }
  }
}

void build_node(const NumericMatrix& X,
                const IntegerVector& W,
                const NumericMatrix& Phi,
                TreeBuffers& tree,
                int node,
                const SortedRows& build_sorted,
                const SortedRows& pop_sorted,
                int depth,
                int max_depth,
                int mtry,
                int min_arm_build,
                int min_arm_pop,
                double alpha) {
  const int n_covariates = X.ncol();

  if (depth >= max_depth) {
    tree.leaf_pop[node] = pop_sorted[0];
    return;
  }

  // (F2) random-split: a fresh uniform draw of `mtry` candidate covariates at
  // every node, so each covariate is a candidate with probability mtry / p.
  const int n_candidates = std::min(mtry, n_covariates);
  std::vector<int> candidates(n_covariates);
  for (int j = 0; j < n_covariates; ++j) candidates[j] = j;
  for (int j = 0; j < n_candidates; ++j) {
    const int pick = j + static_cast<int>(R::unif_rand() * (n_covariates - j));
    std::swap(candidates[j], candidates[std::min(pick, n_covariates - 1)]);
  }
  candidates.resize(n_candidates);

  SplitCandidate best;
  for (std::size_t c = 0; c < candidates.size(); ++c) {
    const int variable = candidates[c];
    search_variable(X, W, Phi, variable, build_sorted[variable],
                    pop_sorted[variable], min_arm_build, min_arm_pop, alpha,
                    best);
  }

  if (!best.found) {
    tree.leaf_pop[node] = pop_sorted[0];
    return;
  }

  SortedRows build_left, build_right, pop_left, pop_right;
  partition_sorted(X, best.variable, best.value, build_sorted, build_left,
                   build_right);
  partition_sorted(X, best.variable, best.value, pop_sorted, pop_left,
                   pop_right);

  tree.split_var[node] = best.variable;
  tree.split_value[node] = best.value;
  const int left = add_node(tree);
  const int right = add_node(tree);
  tree.left_child[node] = left;
  tree.right_child[node] = right;

  build_node(X, W, Phi, tree, left, build_left, pop_left, depth + 1, max_depth,
             mtry, min_arm_build, min_arm_pop, alpha);
  build_node(X, W, Phi, tree, right, build_right, pop_right, depth + 1,
             max_depth, mtry, min_arm_build, min_arm_pop, alpha);
}

}  // namespace

//' Grow one Causal-DRF tree.
//'
//' @param X Numeric n x p covariate matrix (full training sample).
//' @param W Integer treatment vector of length n.
//' @param Phi Numeric n x 2S matrix of random Fourier features of the response.
//' @param build_rows Zero-based rows used to determine the splits.
//' @param pop_rows Zero-based rows used to populate the leaves.
//' @param mtry Candidate covariates per node.
//' @param min_arm_build Minimum units of each arm per child in the build sample.
//' @param min_arm_pop Minimum units of each arm per leaf in the populate sample.
//' @param alpha Minimum fraction of a parent's units retained by each child.
//' @param max_depth Depth cap.
//' @return List describing the tree topology and its leaf membership.
// [[Rcpp::export]]
List causal_drf_grow_tree(NumericMatrix X,
                          IntegerVector W,
                          NumericMatrix Phi,
                          IntegerVector build_rows,
                          IntegerVector pop_rows,
                          int mtry,
                          int min_arm_build,
                          int min_arm_pop,
                          double alpha,
                          int max_depth) {
  TreeBuffers tree;
  const int root = add_node(tree);
  const int n_covariates = X.ncol();
  SortedRows build_sorted(n_covariates,
                          std::vector<int>(build_rows.begin(), build_rows.end()));
  SortedRows pop_sorted(n_covariates,
                        std::vector<int>(pop_rows.begin(), pop_rows.end()));
  for (int j = 0; j < n_covariates; ++j) {
    order_by_column(X, j, build_sorted[j]);
    order_by_column(X, j, pop_sorted[j]);
  }
  build_node(X, W, Phi, tree, root, build_sorted, pop_sorted, 0, max_depth,
             mtry, min_arm_build, min_arm_pop, alpha);

  const int n_nodes = static_cast<int>(tree.split_var.size());
  IntegerVector leaf_start(n_nodes, 0);
  IntegerVector leaf_size(n_nodes, 0);
  int total = 0;
  for (int node = 0; node < n_nodes; ++node) {
    leaf_start[node] = total;
    leaf_size[node] = static_cast<int>(tree.leaf_pop[node].size());
    total += leaf_size[node];
  }
  IntegerVector leaf_members(total);
  int cursor = 0;
  for (int node = 0; node < n_nodes; ++node) {
    for (std::size_t k = 0; k < tree.leaf_pop[node].size(); ++k) {
      leaf_members[cursor++] = tree.leaf_pop[node][k];
    }
  }

  return List::create(
      _["split_var"] = wrap(tree.split_var),
      _["split_value"] = wrap(tree.split_value),
      _["left_child"] = wrap(tree.left_child),
      _["right_child"] = wrap(tree.right_child),
      _["leaf_start"] = leaf_start,
      _["leaf_size"] = leaf_size,
      _["leaf_members"] = leaf_members,
      _["n_nodes"] = n_nodes);
}

//' Route test points to leaves and accumulate the signed prediction weights.
//'
//' Implements eq. (3) of the paper: each tree contributes, for the leaf a test
//' point falls into, mass 1 / |{j in leaf : W_j = 1}| to each treated member and
//' 1 / |{j in leaf : W_j = 0}| to each control member; the tree contributions are
//' averaged. The two arm matrices are returned separately, so their difference
//' is the signed weight vector and each of them is a normalized arm law.
//'
//' @param trees List of trees from `causal_drf_grow_tree`.
//' @param tree_group Zero-based subforest index of each tree.
//' @param n_groups Number of subforests.
//' @param X_test Numeric n_test x p matrix.
//' @param W Integer treatment vector of length n_train.
//' @param n_train Training sample size.
//' @param with_groups Whether to also return per-subforest weights.
//' @return List with the pooled arm weights and, optionally, the grouped ones.
// [[Rcpp::export]]
List causal_drf_predict_weights(List trees,
                                IntegerVector tree_group,
                                int n_groups,
                                NumericMatrix X_test,
                                IntegerVector W,
                                int n_train,
                                bool with_groups) {
  const int n_trees = trees.size();
  const int n_test = X_test.nrow();

  NumericMatrix treated(n_test, n_train);
  NumericMatrix control(n_test, n_train);
  // A tree can only contribute to a test point if the leaf it lands in holds
  // both arms. The both-arm leaf minimums make that the normal case, but the
  // contributing trees are counted rather than assumed so that the arm weights
  // stay exactly normalized even in a degenerate configuration.
  std::vector<int> contributing(n_test, 0);
  std::vector<int> grouped_contributing(
      with_groups ? static_cast<std::size_t>(n_groups) * n_test : 0, 0);

  // Grouped weights are only materialized when the resampling inference of
  // eqs. (4)-(6) is requested; they cost n_groups x n_test x n_train doubles.
  NumericVector grouped_treated;
  NumericVector grouped_control;
  if (with_groups) {
    grouped_treated = NumericVector(
        Dimension(n_groups, n_test, n_train));
    grouped_control = NumericVector(
        Dimension(n_groups, n_test, n_train));
  }

  for (int t = 0; t < n_trees; ++t) {
    List tree = trees[t];
    IntegerVector split_var = tree["split_var"];
    NumericVector split_value = tree["split_value"];
    IntegerVector left_child = tree["left_child"];
    IntegerVector right_child = tree["right_child"];
    IntegerVector leaf_start = tree["leaf_start"];
    IntegerVector leaf_size = tree["leaf_size"];
    IntegerVector leaf_members = tree["leaf_members"];
    const int group = tree_group[t];

    for (int i = 0; i < n_test; ++i) {
      int node = 0;
      while (split_var[node] >= 0) {
        node = (X_test(i, split_var[node]) <= split_value[node])
                   ? left_child[node]
                   : right_child[node];
      }
      const int start = leaf_start[node];
      const int size = leaf_size[node];
      int n_treated = 0;
      for (int k = 0; k < size; ++k) {
        if (W[leaf_members[start + k]] == 1) ++n_treated;
      }
      const int n_control = size - n_treated;
      if (n_treated == 0 || n_control == 0) continue;

      ++contributing[i];
      if (with_groups) ++grouped_contributing[group + n_groups * i];

      const double treated_share = 1.0 / n_treated;
      const double control_share = 1.0 / n_control;

      for (int k = 0; k < size; ++k) {
        const int row = leaf_members[start + k];
        if (W[row] == 1) {
          treated(i, row) += treated_share;
          if (with_groups) {
            grouped_treated[group + n_groups * (i + n_test * row)] +=
                treated_share;
          }
        } else {
          control(i, row) += control_share;
          if (with_groups) {
            grouped_control[group + n_groups * (i + n_test * row)] +=
                control_share;
          }
        }
      }
    }
  }

  for (int i = 0; i < n_test; ++i) {
    if (contributing[i] == 0) continue;
    const double scale = 1.0 / contributing[i];
    for (int row = 0; row < n_train; ++row) {
      treated(i, row) *= scale;
      control(i, row) *= scale;
    }
  }
  if (with_groups) {
    for (int g = 0; g < n_groups; ++g) {
      for (int i = 0; i < n_test; ++i) {
        const int count = grouped_contributing[g + n_groups * i];
        if (count == 0) continue;
        const double scale = 1.0 / count;
        for (int row = 0; row < n_train; ++row) {
          grouped_treated[g + n_groups * (i + n_test * row)] *= scale;
          grouped_control[g + n_groups * (i + n_test * row)] *= scale;
        }
      }
    }
  }

  List out = List::create(
      _["treated"] = treated,
      _["control"] = control,
      _["contributing_trees"] = wrap(contributing));
  if (with_groups) {
    out["grouped_treated"] = grouped_treated;
    out["grouped_control"] = grouped_control;
  }
  return out;
}
