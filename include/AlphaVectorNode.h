/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include "Cache.h"
#include "StateVector.h"

namespace MCVI {

class AlphaVectorNode {
 private:
  int64_t _best_action;
  LRUCache<State, double, StateHash, StateEqual>
      _alpha;  // expected total reward of executing policy from
               // this node with initial state

 public:
  AlphaVectorNode(int64_t init_best_action);

  /// @brief Return the best action
  int64_t GetBestAction() const { return _best_action; }

  /// @brief Return the best value for this node
  double V_node() const;

  void SetAlpha(const State& state, double value) { _alpha[state] = value; }
  std::optional<double> GetAlpha(const State& state) const;
};

}  // namespace MCVI
