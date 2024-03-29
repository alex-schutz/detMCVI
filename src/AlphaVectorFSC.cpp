#include "AlphaVectorFSC.h"

namespace MCVI {

int64_t AlphaVectorFSC::GetEdgeValue(int64_t nI, int64_t action,
                                     int64_t observation) const {
  const EdgeMap& m = _edges[nI];
  const auto it = m.find({action, observation});
  if (it != m.cend()) return it->second;
  return -1;
}

int64_t AlphaVectorFSC::AddNode(const AlphaVectorNode& node) {
  _nodes.emplace_back(node);
  return _nodes.size() - 1;
}

void AlphaVectorFSC::UpdateEdge(int64_t nI, int64_t a, int64_t o,
                                int64_t nI_new) {
  _edges[nI][{a, o}] = nI_new;
}

void AlphaVectorFSC::UpdateEdge(int64_t nI,
                                const AlphaVectorFSC::EdgeMap& edges) {
  _edges[nI] = edges;
}

}  // namespace MCVI
