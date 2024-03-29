/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include "AlphaVectorNode.h"

namespace MCVI {

class AlphaVectorFSC {
 public:
  struct PairHash {
    std::size_t operator()(const std::pair<int64_t, int64_t>& p) const {
      size_t hash = 0x9e3779b97f4a7c15;
      hash ^= std::hash<int64_t>{}(p.first) + 0x9e3779b9;
      hash ^= std::hash<int64_t>{}(p.second) + 0x9e3779b9 + (hash << 6) +
              (hash >> 2);
      return hash;
    }
  };

  using EdgeMap =
      std::unordered_map<std::pair<int64_t, int64_t>, int64_t, PairHash>;

 private:
  std::vector<EdgeMap> _edges;
  std::vector<AlphaVectorNode> _nodes;
  std::vector<int64_t> _action_space;
  std::vector<int64_t> _observation_space;
  int64_t _start_node_index;

 public:
  AlphaVectorFSC(int64_t max_node_size,
                 const std::vector<int64_t>& action_space,
                 const std::vector<int64_t>& observation_space)
      : _edges(max_node_size, EdgeMap()),
        _nodes(),
        _action_space(action_space),
        _observation_space(observation_space),
        _start_node_index(-1) {}

  /// @brief Return a reference to node number nI
  AlphaVectorNode& GetNode(int64_t nI) { return _nodes[nI]; }

  /// @brief Return the number of nodes in the FSC
  int64_t NumNodes() const { return _nodes.size(); }

  /// @brief Add a node to the FSC
  int64_t AddNode(const AlphaVectorNode& node);

  /// @brief Return the node index assosciated with node nI, action a and
  /// observation o. Returns -1 if it does not exist.
  int64_t GetEdgeValue(int64_t nI, int64_t a, int64_t o) const;

  /// @brief Set the node index assosciated with node nI, action a and
  /// observation o to nI_new.
  void UpdateEdge(int64_t nI, int64_t a, int64_t o, int64_t nI_new);
  void UpdateEdge(int64_t nI, const AlphaVectorFSC::EdgeMap& edges);

  const std::vector<int64_t>& GetActionSpace() { return _action_space; }
  const std::vector<int64_t>& GetObsSpace() { return _observation_space; }

  int64_t GetStartNodeIndex() const { return _start_node_index; }
  void SetStartNodeIndex(int64_t idx) { _start_node_index = idx; }
};

}  // namespace MCVI
