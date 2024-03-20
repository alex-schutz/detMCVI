/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _ALPHAVECTORFSC_H_
#define _ALPHAVECTORFSC_H_

#include "AlphaVectorNode.h"

class AlphaVectorFSC {
 private:
  struct PairHash {
    std::size_t operator()(const std::pair<int64_t, int64_t>& p) const {
      size_t hash = 0x9e3779b97f4a7c15;
      hash ^= std::hash<int64_t>{}(p.first) + 0x9e3779b9;
      hash ^= std::hash<int64_t>{}(p.second) + 0x9e3779b9 + (hash << 6) +
              (hash >> 2);
      return hash;
    }
  };

  std::vector<
      std::unordered_map<std::pair<int64_t, int64_t>, int64_t, PairHash>>
      _eta;
  std::vector<AlphaVectorNode> _nodes;
  std::vector<int64_t> _action_space;
  std::vector<int64_t> _observation_space;

 public:
  AlphaVectorFSC(int64_t max_node_size,
                 const std::vector<int64_t>& action_space,
                 const std::vector<int64_t>& observation_space)
      : _eta(max_node_size, std::unordered_map<std::pair<int64_t, int64_t>,
                                               int64_t, PairHash>()),
        _nodes(),
        _action_space(action_space),
        _observation_space(observation_space) {}

  ~AlphaVectorFSC() = default;

  AlphaVectorNode& GetNode(int64_t nI) { return _nodes[nI]; }
  size_t NumNodes() const { return _nodes.size(); }
  void AddNode(const BeliefParticles& state_particles,
               const std::vector<int64_t>& action_space,
               const std::vector<int64_t>& observation_space) {
    _nodes.emplace_back(
        AlphaVectorNode(state_particles, action_space, observation_space));
  }

  /// @brief Return the eta value assosciated with node nI, action a and
  /// observation o. Returns -1 if it does not exist.
  int64_t GetEtaValue(int64_t nI, int64_t a, int64_t o) const;

  void UpdateEta(int64_t nI, int64_t a, int64_t o, int64_t nI_new) {
    _eta[nI][{a, o}] = nI_new;
  }

  const std::vector<int64_t>& GetActionSpace() { return _action_space; }
  const std::vector<int64_t>& GetObsSpace() { return _observation_space; }
};

#endif /* !_ALPHAVECTORFSC_H_ */
