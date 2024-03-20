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

class AlphaVectorNode {
  using ValueMap = std::unordered_map<
      int64_t,
      std::unordered_map<int64_t, std::unordered_map<int64_t, double>>>;

 private:
  std::vector<int64_t> _state_particles;
  std::unordered_map<int64_t, double> _Q_action;
  std::unordered_map<int64_t, double> _R_action;  // expected instant reward
  ValueMap _V_a_o_n;
  std::unordered_map<int64_t, double> _V_node_s;
  std::unordered_map<int64_t, int64_t> _V_node_s_count;
  double _V_node;
  std::unordered_map<int64_t, bool> _best_action_update;
  mutable std::mt19937_64 _rng;

 public:
  AlphaVectorNode(const std::vector<int64_t>& state_particles,
                  const std::vector<int64_t>& action_space,
                  const std::vector<int64_t>& observation_space,
                  uint64_t seed = std::random_device{}());
  AlphaVectorNode(const std::vector<int64_t>& action_space,
                  const std::vector<int64_t>& observation_space,
                  uint64_t seed = std::random_device{}());

  ~AlphaVectorNode() = default;

  int64_t GetBestAction() const;

  const std::unordered_map<int64_t, double>& GetActionObservationValues(
      int64_t a, int64_t o) const {
    return _V_a_o_n.at(a).at(o);
  }

  void InitR(const std::vector<int64_t>& action_space) {
    _R_action = InitDoubleKeys(action_space);
  }
  void InitQ(const std::vector<int64_t>& action_space) {
    _Q_action = InitDoubleKeys(action_space);
  }
  void InitVals(const std::vector<int64_t>& action_space,
                const std::vector<int64_t>& observation_space) {
    _V_a_o_n = InitValueMap(action_space, observation_space);
  }
  double V_node() const { return _V_node; }

  int64_t SampleParticle() const;

  double GetR(int64_t action) const { return _R_action.at(action); }
  double GetQ(int64_t action) const { return _Q_action.at(action); }
  void AddR(int64_t action, double reward) { _R_action[action] += reward; }
  void AddQ(int64_t action, double q) { _Q_action[action] += q; }
  void NormaliseQ(int64_t action, int64_t N) { _Q_action[action] /= N; }
  void UpdateValue(int64_t action, int64_t observation, int64_t nI,
                   double val) {
    _V_a_o_n[action][observation][nI] += val;
  }

  /// @brief  Recalculate _V_node by finding the best action
  void UpdateBestValue();

 private:
  std::unordered_map<int64_t, double> AlphaVectorNode::InitDoubleKeys(
      const std::vector<int64_t>& action_space) const;
  std::unordered_map<int64_t, bool> AlphaVectorNode::InitBoolKeys(
      const std::vector<int64_t>& action_space) const;
  ValueMap AlphaVectorNode::InitValueMap(
      const std::vector<int64_t>& action_space,
      const std::vector<int64_t>& observation_space) const;
};

#endif /* !_ALPHAVECTORNODE_H_ */
