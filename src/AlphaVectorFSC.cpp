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

void AlphaVectorFSC::GenerateGraphviz(
    std::ostream& ofs, const std::vector<std::string>& actions,
    const std::vector<std::string>& observations) const {
  if (!ofs) throw std::logic_error("Invalid output file");

  ofs << "digraph AlphaVectorFSC {" << std::endl;

  ofs << "node [shape=circle];" << std::endl;

  // Loop through each node
  for (int64_t i = 0; i < NumNodes(); ++i) {
    const AlphaVectorNode& node = GetNode(i);

    std::string action = actions.empty() ? std::to_string(node.GetBestAction())
                                         : actions[node.GetBestAction()];
    ofs << "  " << i << " [label=\"" << i << ",\\n" << action;
    if (i == GetStartNodeIndex()) {  // highlight start node with double outline
      ofs << "\", penwidth=2];" << std::endl;
    } else {
      ofs << "\"];" << std::endl;
    }

    // Loop through edges from this node
    for (const auto& edge : _edges[i]) {
      if (edge.first.first != node.GetBestAction()) continue;
      std::string observation = observations.empty()
                                    ? std::to_string(edge.first.second)
                                    : observations[edge.first.second];
      int64_t target_node = edge.second;
      ofs << "  " << i << " -> " << target_node << " [label=\"" << observation
          << "\"];";
    }
    ofs << std::endl;
  }

  ofs << "}" << std::endl;
}

}  // namespace MCVI
