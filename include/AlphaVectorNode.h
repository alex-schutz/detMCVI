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

#include "BeliefParticles.h"
#include "BeliefTree.h"

namespace MCVI {

class AlphaVectorNode {
  using ValueMap = std::unordered_map<
      int64_t,
      std::unordered_map<int64_t, std::unordered_map<int64_t, double>>>;

 private:
  std::unordered_map<int64_t, double> _Q_action;
  std::unordered_map<int64_t, double> _R_action;  // expected instant reward
  ValueMap _V_a_o_n;
  double _V_node;  // a lower bound value
  int64_t _best_action;

 public:
  AlphaVectorNode(const std::vector<int64_t>& action_space,
                  const std::vector<int64_t>& observation_space);

  /// @brief Return the best action
  int64_t GetBestAction() const { return _best_action; }

  /// @brief Return the stored value associated with action a and observation o
  const std::unordered_map<int64_t, double>& GetActionObservationValues(
      int64_t a, int64_t o) const;

  /// @brief Return the best value for this node
  double V_node() const { return _V_node; }

  /// @brief Return the R value associated with `action`
  double GetR(int64_t action) const { return _R_action.at(action); }

  /// @brief Return the Q value associated with `action`
  double GetQ(int64_t action) const { return _Q_action.at(action); }

  /// @brief Add `reward` to the R value associated with `action`
  void AddR(int64_t action, double reward) { _R_action[action] += reward; }

  /// @brief Add `q` to the Q value associated with `action`
  void AddQ(int64_t action, double q) { _Q_action[action] += q; }

  /// @brief Divide the Q value at `action` by `N`
  void NormaliseQ(int64_t action, int64_t N) { _Q_action[action] /= N; }

  /// @brief Set the value associated with `action`, `observation` and node `nI`
  /// to `val`
  void UpdateValue(int64_t action, int64_t observation, int64_t nI, double val);

  /// @brief Recalculate _V_node by finding the best action
  void UpdateBestValue(std::shared_ptr<BeliefTreeNode> tr);

 private:
  std::unordered_map<int64_t, double> InitDoubleKeys(
      const std::vector<int64_t>& action_space) const;

  ValueMap InitValueMap(const std::vector<int64_t>& action_space,
                        const std::vector<int64_t>& observation_space) const;

  /// @brief Calculate the best action according to the current Q values
  int64_t CalculateBestAction() const;
};

}  // namespace MCVI
