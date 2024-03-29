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
  BeliefParticles _state_particles;
  std::unordered_map<
      int64_t, std::unordered_map<int64_t, std::shared_ptr<BeliefTreeNode>>>
      _child_nodes;
  int64_t _best_action;
  double _upper_bound;
  double _lower_bound;
  int64_t _fsc_node_index;

 public:
  BeliefTreeNode(const BeliefParticles& belief, int64_t best_action,
                 double upper_bound, double lower_bound)
      : _state_particles(belief),
        _best_action(),
        _upper_bound(),
        _lower_bound(),
        _fsc_node_index(-1) {}

  void AddChild(int64_t action, int64_t observation,
                std::shared_ptr<BeliefTreeNode> child);

  const BeliefParticles& GetParticles() const { return _state_particles; }

  int64_t GetFSCNodeIndex() const { return _fsc_node_index; }
  void SetFSCNodeIndex(int64_t idx) { _fsc_node_index = idx; }

  void SetBestAction(int64_t action, double lower_bound);

  double GetUpper() const { return _upper_bound; }
  double GetLower() const { return _lower_bound; }
};

void CreateBeliefTreeNode(std::shared_ptr<BeliefTreeNode> parent,
                          int64_t action, int64_t observation,
                          const BeliefParticles& belief,
                          const std::vector<int64_t>& action_space,
                          QLearning::QLearningPolicy policy, SimInterface* sim);

#endif /* !_BELIEFTREE_H_ */
