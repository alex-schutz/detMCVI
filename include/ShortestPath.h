/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

#include "Cache.h"
#include "StateVector.h"

namespace MCVI {

class InfMap {
 private:
  StateMap<double> _map;

 public:
  InfMap() = default;
  InfMap(const StateMap<double>& v) : _map(v) {}
  const StateMap<double>& map() const { return _map; }
  double& operator[](std::vector<int64_t> key) {
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
  mutable StateMap<bool> inQueue;
  mutable StateMap<int64_t> depth;
  mutable StateMap<std::pair<State, int64_t>> predecessor;

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
  virtual std::vector<std::tuple<State, double, int64_t>> getEdges(
      const State& node) const = 0;

  /**
   * @brief Calculate the shortest path between source and each node up to
   * maximum depth N
   *
   * Returns a map of <destination, cost> pairs and a map of <node,
   * <predecessor_node, edge>> pairs. Use the latter with `reconstructPath`.
   */
  std::tuple<StateMap<double>, StateMap<std::pair<State, int64_t>>> calculate(
      const State& source, int64_t N) const;

  /// @brief Reconstruct path of <node, next_edge> to target
  std::vector<std::pair<State, int64_t>> reconstructPath(
      const State& target,
      const StateMap<std::pair<State, int64_t>>& pred) const;
};

class MaximiseReward {
 private:
  double discount_factor;
  mutable LRUCache<
      State, std::unordered_map<int64_t, std::tuple<int64_t, State, double>>,
      StateHash, StateEqual>
      cache;

  std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
  GetCachedOrSearch(const State& state, int64_t depth_to_go) const;

  std::pair<double, std::vector<std::tuple<int64_t, State, double>>> Search(
      const State& state, int64_t depth_to_go) const;

 public:
  MaximiseReward(double discount_factor, size_t cache_capacity = 25000)
      : discount_factor(discount_factor), cache(cache_capacity) {}

  // Available successors <action, state, reward> tuples
  virtual std::vector<std::tuple<int64_t, State, double, bool>> getSuccessors(
      const State& state) const = 0;

  // Return the maximum reward that can be obtained starting in `state`
  // up to `max_depth`, alongside the path of <action, next state,
  // immediate_reward> pairs
  std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
  getMaxReward(const State& state, int64_t max_depth) const;
};

}  // namespace MCVI
