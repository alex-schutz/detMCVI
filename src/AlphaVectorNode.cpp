#include "AlphaVectorNode.h"

#include <algorithm>
#include <limits>

namespace MCVI {

static bool CmpPair(
    const std::pair<State, std::pair<double, std::list<State>::iterator>>& p1,
    const std::pair<State, std::pair<double, std::list<State>::iterator>>& p2) {
  return p1.second.first < p2.second.first;
}

AlphaVectorNode::AlphaVectorNode(int64_t init_best_action)
    : _best_action(init_best_action), _alpha(2000000) {}

std::optional<double> AlphaVectorNode::GetAlpha(const State& state) const {
  const auto it = _alpha.find(state);
  if (it == _alpha.cend()) return std::nullopt;
  return it->second.first;
}

double AlphaVectorNode::V_node() const {
  auto v = std::max_element(_alpha.cbegin(), _alpha.cend(), CmpPair);
  if (v == _alpha.cend()) return 0;
  return v->second.first;
}

}  // namespace MCVI
