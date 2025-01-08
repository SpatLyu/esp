#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <Rcpp.h>

/**
 * Computes the spatial PageRank for nodes based on node values `vec` and the spatial weight matrix `wt`.
 *
 * @param vec The node values (used as initial importance scores).
 * @param wt The spatial weight matrix (normalized to represent transition probabilities).
 * @param dampingFactor The damping factor for PageRank (usually 0.85).
 * @param maxIterations Maximum number of iterations for convergence.
 * @param tolerance Convergence tolerance.
 * @return A vector of node ranks.
 */
std::vector<double> spatialPageRank(
    const std::vector<double>& vec,
    const std::vector<std::vector<double>>& wt,
    double dampingFactor = 0.85,
    size_t maxIterations = 100,
    double tolerance = 1e-6
) {
  size_t n = vec.size(); // Number of nodes
  std::vector<double> ranks(n, 1.0 / n); // Initialize ranks uniformly

  // Normalize the weight matrix to represent transition probabilities
  std::vector<std::vector<double>> transitionMatrix(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; ++i) {
    double rowSum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      rowSum += wt[i][j];
    }
    if (rowSum > 0) {
      for (size_t j = 0; j < n; ++j) {
        transitionMatrix[i][j] = wt[i][j] / rowSum;
      }
    }
  }

  // Perform PageRank iterations
  for (size_t iter = 0; iter < maxIterations; ++iter) {
    std::vector<double> newRanks(n, 0.0);

    // Update ranks based on transition matrix and damping factor
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        newRanks[i] += transitionMatrix[j][i] * ranks[j];
      }
      newRanks[i] = dampingFactor * newRanks[i] + (1 - dampingFactor) * vec[i];
    }

    // Check for convergence
    double diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
      diff += std::abs(newRanks[i] - ranks[i]);
    }
    if (diff < tolerance) {
      break;
    }

    ranks = newRanks;
  }

  return ranks;
}

/**
 * Generates a rank order sequence for nodes based on their spatial PageRank scores.
 *
 * @param wt  The spatial weight matrix.
 * @param vec The node values.
 * @return A vector of node indices sorted by their rank in descending order.
 */
std::vector<size_t> genNodeRank(const std::vector<double>& vec,
                                const std::vector<std::vector<double>>& wt) {
  // Compute spatial PageRank scores
  std::vector<double> ranks = spatialPageRank(vec, wt);

  // Create a vector of pairs: (node index, rank)
  std::vector<std::pair<size_t, double>> nodeRankPairs;
  for (size_t i = 0; i < ranks.size(); ++i) {
    nodeRankPairs.emplace_back(i, ranks[i]);
  }

  // Sort the nodes by their ranks in descending order
  std::sort(nodeRankPairs.begin(), nodeRankPairs.end(),
            [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
              return b.second < a.second; // Sort by rank in descending order
            });

  // Extract the sorted node indices
  std::vector<size_t> rankOrder;
  for (const auto& pair : nodeRankPairs) {
    rankOrder.push_back(pair.first);
  }

  return rankOrder;
}

// Rcpp wrapper function
// [[Rcpp::export]]
Rcpp::IntegerVector RcppGenNodeRank(Rcpp::NumericVector vec, Rcpp::NumericMatrix wt) {
  // Convert R NumericVector to std::vector<double>
  std::vector<double> vec_cpp(vec.begin(), vec.end());

  // Convert R NumericMatrix to std::vector<std::vector<double>>
  std::vector<std::vector<double>> wt_cpp(wt.nrow(), std::vector<double>(wt.ncol()));
  for (int i = 0; i < wt.nrow(); ++i) {
    for (int j = 0; j < wt.ncol(); ++j) {
      wt_cpp[i][j] = wt(i, j);
    }
  }

  // Call the genNodeRank function
  std::vector<size_t> rankOrder = genNodeRank(vec_cpp, wt_cpp);

  // Convert the result to R IntegerVector
  Rcpp::IntegerVector rankOrder_r(rankOrder.begin(), rankOrder.end());

  return rankOrder_r;
}
