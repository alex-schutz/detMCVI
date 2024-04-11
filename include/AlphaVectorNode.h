/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "BeliefTree.h"

namespace MCVI {

class AlphaVectorNode {
  using ValueMap = std::unordered_map<
      int64_t,
      std::unordered_map<int64_t, std::unordered_map<int64_t, double>>>;

 private:
  mutable std::unordered_map<int64_t, double> _Q_action;
  mutable std::unordered_map<int64_t, double>
      _R_action;  // expected instant reward
  mutable ValueMap _V_a_o_n;
  double _V_node;  // a lower bound value
  int64_t _best_action;

 public:
  AlphaVectorNode(int64_t init_best_action);

  /// @brief Return the best action
  int64_t GetBestAction() const { return _best_action; }

  /// @brief Return the best value for this node
  double V_node() const { return _V_node; }

  /// @brief Return the R value associated with `action`
  double GetR(int64_t action) const { return _R_action[action]; }

  /// @brief Return the Q value associated with `action`
  double GetQ(int64_t action) const { return _Q_action[action]; }

  /// @brief Add `reward` to the R value associated with `action`
  void AddR(int64_t action, double reward) { _R_action[action] += reward; }

  /// @brief Add `q` to the Q value associated with `action`
  void AddQ(int64_t action, double q) { _Q_action[action] += q; }

  /// @brief Add `val` to the value associated with `action`, `observation` and
  /// node `nI`
  void AddValue(int64_t action, int64_t observation, int64_t nI, double val);

  /// @brief Update the best action
  void UpdateBestValue(int64_t action, std::shared_ptr<BeliefTreeNode> tr);

  // Return a map of <observation, best node> for the given action, and the sum
  // of the values over all observations
  std::tuple<std::unordered_map<int64_t, int64_t>, double> BestNodePerObs(
      int64_t action);
};

}  // namespace MCVI
