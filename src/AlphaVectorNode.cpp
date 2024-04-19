#include "AlphaVectorNode.h"

#include <algorithm>
#include <limits>

namespace MCVI {

AlphaVectorNode::AlphaVectorNode(int64_t init_best_action)
    : _V_node(0.0), _best_action(init_best_action), _alpha() {}

std::optional<double> AlphaVectorNode::GetAlpha(int64_t state) const {
  const auto it = _alpha.find(state);
  if (it == _alpha.cend()) return std::nullopt;
  return it->second;
}

}  // namespace MCVI
