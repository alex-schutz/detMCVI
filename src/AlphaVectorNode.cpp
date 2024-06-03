#include "AlphaVectorNode.h"

#include <algorithm>
#include <limits>

namespace MCVI {

static bool CmpPair(const std::pair<State, double>& p1,
                    const std::pair<State, double>& p2) {
  return p1.second < p2.second;
}

AlphaVectorNode::AlphaVectorNode(int64_t init_best_action)
    : _best_action(init_best_action), _alpha({}) {}

std::optional<double> AlphaVectorNode::GetAlpha(const State& state) const {
  const auto it = _alpha.find(state);
  if (it == _alpha.cend()) return std::nullopt;
  return it->second;
}

double AlphaVectorNode::V_node() const {
  auto v = std::max_element(_alpha.begin(), _alpha.end(), CmpPair);
  if (v == _alpha.cend()) return 0;
  return v->second;
}

}  // namespace MCVI
