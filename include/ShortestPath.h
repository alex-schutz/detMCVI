/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace MCVI {

class InfMap {
 private:
  std::unordered_map<int64_t, double> _map;

 public:
  InfMap() = default;
  const std::unordered_map<int64_t, double>& map() const { return _map; }
  double& operator[](int64_t key) {
    auto it = _map.find(key);
    if (it != _map.end()) return it->second;
    return _map.insert({key, std::numeric_limits<double>::infinity()})
        .first->second;
  }
  auto begin() { return _map.begin(); }
  auto end() { return _map.end(); }
  auto size() const { return _map.size(); }
  auto empty() const { return _map.empty(); }
  void clear() { _map.clear(); }
};

/**
 * @brief Implements the Shortest Path Faster Algorithm (SPFA)
 *
 * Inherit from this class to calculate the shortest path from source to all
 * reachable nodes, according to edges given by `getEdges`.
 */
class ShortestPathFasterAlgorithm {
 private:
  mutable InfMap d;
  mutable std::unordered_map<int64_t, bool> inQueue;
  mutable std::unordered_map<int64_t, int64_t> depth;
  mutable std::unordered_map<int64_t, std::pair<int64_t, int64_t>> predecessor;

  void initParams() const;

 public:
  ShortestPathFasterAlgorithm() = default;

  /**
   * @brief Get the edges and weights out of `node`.
   *
   * Returns a list of destination nodes, associated edge costs and edge number.
   *
   * The edge number is only used to label which edge is taken when
   * reconstructing the path, and can be set arbitrarily.
   */
  virtual std::vector<std::tuple<int64_t, double, int64_t>> getEdges(
      int64_t node) const = 0;

  /**
   * @brief Calculate the shortest path between source and each node up to
   * maximum depth N
   *
   * Returns a map of <destination, cost> pairs and a map of <node,
   * <predecessor_node, edge>> pairs. Use the latter with `reconstructPath`.
   */
  std::tuple<std::unordered_map<int64_t, double>,
             std::unordered_map<int64_t, std::pair<int64_t, int64_t>>>
  calculate(int64_t source, int64_t N) const;

  /// @brief Reconstruct path of <node, next_edge> to target
  std::vector<std::pair<int64_t, int64_t>> reconstructPath(
      int64_t target,
      const std::unordered_map<int64_t, std::pair<int64_t, int64_t>>& paths)
      const;
};

}  // namespace MCVI
