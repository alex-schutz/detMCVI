#include "AlphaVectorNode.h"

#include <algorithm>
#include <limits>

namespace MCVI {

std::unordered_map<int64_t, double> AlphaVectorNode::InitDoubleKeys(
    const std::vector<int64_t>& action_space) const {
  std::unordered_map<int64_t, double> v;
  for (const auto& a : action_space) v[a] = 0.0;
  return v;
}

AlphaVectorNode::ValueMap AlphaVectorNode::InitValueMap(
    const std::vector<int64_t>& action_space,
    const std::vector<int64_t>& observation_space) const {
  ValueMap v;
  for (const auto& a : action_space) {
    std::unordered_map<int64_t, std::unordered_map<int64_t, double>> v_a;
    for (const auto& o : observation_space) v_a[o] = {};
    v[a] = v_a;
  }
  return v;
}

static int64_t RandomAction(const std::vector<int64_t>& action_space) {
  std::mt19937_64 rng;
  std::uniform_int_distribution<> action_dist(0, action_space.size() - 1);
  return action_space[action_dist(rng)];
}

AlphaVectorNode::AlphaVectorNode(const std::vector<int64_t>& action_space,
                                 const std::vector<int64_t>& observation_space)
    : _Q_action(InitDoubleKeys(action_space)),
      _R_action(InitDoubleKeys(action_space)),
      _V_a_o_n(InitValueMap(action_space, observation_space)),
      _V_node(0.0),
      _best_action(RandomAction(action_space)) {}

int64_t AlphaVectorNode::CalculateBestAction() const {
  const auto best_action =
      std::max_element(std::begin(_Q_action), std::end(_Q_action),
                       [](const std::pair<int64_t, double>& p1,
                          const std::pair<int64_t, double>& p2) {
                         return p1.second < p2.second;
                       });
  return best_action->first;
}

void AlphaVectorNode::UpdateBestValue(std::shared_ptr<BeliefTreeNode> tr) {
  _best_action = CalculateBestAction();
  _V_node = _Q_action.at(_best_action);
  tr->SetBestAction(_best_action, _V_node);
}

const std::unordered_map<int64_t, double>&
AlphaVectorNode::GetActionObservationValues(int64_t a, int64_t o) const {
  return _V_a_o_n.at(a).at(o);
}

void AlphaVectorNode::UpdateValue(int64_t action, int64_t observation,
                                  int64_t nI, double val) {
  _V_a_o_n[action][observation][nI] += val;
}

}  // namespace MCVI
