/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _ALPHAVECTORNODE_H_
#define _ALPHAVECTORNODE_H_

#include <random>
#include <unordered_map>
#include <vector>

#include "BeliefParticles.h"

class AlphaVectorNode {
  using ValueMap = std::unordered_map<
      int64_t,
      std::unordered_map<int64_t, std::unordered_map<int64_t, double>>>;

 private:
  BeliefParticles _state_particles;
  std::unordered_map<int64_t, double> _Q_action;
  std::unordered_map<int64_t, double> _R_action;  // expected instant reward
  ValueMap _V_a_o_n;
  std::unordered_map<int64_t, double> _V_node_s;
  std::unordered_map<int64_t, int64_t> _V_node_s_count;
  double _V_node;
  std::unordered_map<int64_t, bool> _best_action_update;

 public:
  AlphaVectorNode(const BeliefParticles& state_particles,
                  const std::vector<int64_t>& action_space,
                  const std::vector<int64_t>& observation_space);

  /// @brief Return the best action according to the current Q values
  int64_t GetBestAction() const;

  /// @brief Return the stored value associated with action a and observation o
  const std::unordered_map<int64_t, double>& GetActionObservationValues(
      int64_t a, int64_t o) const;

  /// @brief Reinitialise internal Q, R and V tables with the given key set
  void ReInit(const std::vector<int64_t>& action_space,
              const std::vector<int64_t>& observation_space);

  /// @brief Return the best value for this node
  double V_node() const { return _V_node; }

  /// @brief Return a sampled state particle
  int64_t SampleParticle() const;

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

  /// @brief  Recalculate _V_node by finding the best action
  void UpdateBestValue();

 private:
  std::unordered_map<int64_t, double> InitDoubleKeys(
      const std::vector<int64_t>& action_space) const;

  std::unordered_map<int64_t, bool> InitBoolKeys(
      const std::vector<int64_t>& action_space) const;

  ValueMap InitValueMap(const std::vector<int64_t>& action_space,
                        const std::vector<int64_t>& observation_space) const;
};

#endif /* !_ALPHAVECTORNODE_H_ */
