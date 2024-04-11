#include "AlphaVectorNode.h"

#include <algorithm>
#include <limits>

namespace MCVI {

static bool CmpPair(const std::pair<int64_t, double>& p1,
                    const std::pair<int64_t, double>& p2) {
  return p1.second < p2.second;
}

AlphaVectorNode::AlphaVectorNode(int64_t init_best_action)
    : _Q_action(),
      _R_action(),
      _V_a_o_n(),
      _V_node(0.0),
      _best_action(init_best_action) {}

void AlphaVectorNode::UpdateBestValue(int64_t action,
                                      std::shared_ptr<BeliefTreeNode> tr) {
  _best_action = action;
  _V_node = _Q_action.at(_best_action);
  tr->SetBestAction(_best_action, _V_node);
}

void AlphaVectorNode::AddValue(int64_t action, int64_t observation, int64_t nI,
                               double val) {
  auto& o_n = _V_a_o_n[action];
  auto& _n = o_n[observation];
  _n[nI] += val;
}

std::tuple<std::unordered_map<int64_t, int64_t>, double>
AlphaVectorNode::BestNodePerObs(int64_t action) {
  double sum_v = 0.0;
  std::unordered_map<int64_t, int64_t> edges;
  auto& o_n = _V_a_o_n[action];
  for (const auto& [obs, n_v] : o_n) {
    const auto it = std::max_element(std::begin(n_v), std::end(n_v), CmpPair);
    const int64_t nI = it->first;
    const double V = it->second;
    edges[obs] = nI;
    sum_v += V;
  }
  return {edges, sum_v};
}

}  // namespace MCVI
