/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _BELIEFTREE_H_
#define _BELIEFTREE_H_

#include <memory>
#include <unordered_map>

#include "BeliefParticles.h"
#include "Bound.h"

class BeliefTreeNode {
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

  BeliefParticles _state_particles;
  std::unordered_map<std::pair<int64_t, int64_t>,
                     std::unique_ptr<BeliefTreeNode>, PairHash>
      _child_nodes;
  double _upper_bound;
  double _lower_bound;

 public:
  BeliefTreeNode(const BeliefParticles& updated_belief)
      : _state_particles(updated_belief) {}
};

void CreateBeliefTreeNode(std::unique_ptr<BeliefTreeNode> parent,
                          int64_t action, int64_t observation,
                          const BeliefParticles& updated_belief) {
  parent->_child_nodes[{action, observation}] =
      std::make_unique<BeliefTreeNode>(updated_belief);
}

#endif /* !_BELIEFTREE_H_ */
