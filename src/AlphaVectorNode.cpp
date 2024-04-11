#include "AlphaVectorNode.h"

#include <algorithm>
#include <limits>

namespace MCVI {

AlphaVectorNode::AlphaVectorNode(int64_t init_best_action)
    : _Q_action(),
      _R_action(),
      _V_a_o_n(),
      _V_node(0.0),
      _best_action(init_best_action) {}

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
  auto& o_n = _V_a_o_n[a];
  return o_n[o];
}

void AlphaVectorNode::AddValue(int64_t action, int64_t observation, int64_t nI,
                               double val) {
  auto& o_n = _V_a_o_n[action];
  auto& _n = o_n[observation];
  _n[nI] += val;
}

}  // namespace MCVI
