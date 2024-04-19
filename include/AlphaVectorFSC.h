/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include "AlphaVectorNode.h"
#include "SimInterface.h"

namespace MCVI {

class AlphaVectorFSC {
 private:
  std::vector<std::unordered_map<int64_t, int64_t>> _edges;
  std::vector<AlphaVectorNode> _nodes;
  int64_t _start_node_index;

 public:
  AlphaVectorFSC(int64_t max_node_size)
      : _edges(max_node_size, std::unordered_map<int64_t, int64_t>()),
        _nodes(),
        _start_node_index(-1) {}

  /// @brief Return a reference to node number nI
  const AlphaVectorNode& GetNode(int64_t nI) const { return _nodes.at(nI); }
  AlphaVectorNode& GetNode(int64_t nI) { return _nodes[nI]; }

  /// @brief Return the number of nodes in the FSC
  int64_t NumNodes() const { return _nodes.size(); }

  /// @brief Add a node to the FSC
  int64_t AddNode(const AlphaVectorNode& node);

  /// @brief Return the edges associated with node nI
  const std::unordered_map<int64_t, int64_t>& GetEdges(int64_t nI) const;

  /// @brief Return the node index assosciated with node nI and
  /// observation o. Returns -1 if it does not exist.
  int64_t GetEdgeValue(int64_t nI, int64_t o) const;

  /// @brief Set the node index assosciated with node nI and
  /// observation o to nI_new.
  void UpdateEdge(int64_t nI, int64_t o, int64_t nI_new);
  void UpdateEdge(int64_t nI,
                  const std::unordered_map<int64_t, int64_t>& edges);

  int64_t GetStartNodeIndex() const { return _start_node_index; }
  void SetStartNodeIndex(int64_t idx) { _start_node_index = idx; }

  void GenerateGraphviz(
      std::ostream& ofs, const std::vector<std::string>& actions = {},
      const std::vector<std::string>& observations = {}) const;

  double SimulateTrajectory(int64_t nI, int64_t state, int64_t max_depth,
                            double R_lower, SimInterface* pomdp);

  double GetNodeAlpha(int64_t state, int64_t nI, double R_lower,
                      int64_t max_depth_sim, SimInterface* pomdp);
};

}  // namespace MCVI
