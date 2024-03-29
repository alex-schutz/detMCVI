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
                std::shared_ptr<BeliefTreeNode> child) {
    auto& action_children = _child_nodes[action];
    action_children[observation] = child;
  }

  const BeliefParticles& GetParticles() const { return _state_particles; }

  int64_t GetFSCNodeIndex() const { return _fsc_node_index; }
  void SetFSCNodeIndex(int64_t idx) { _fsc_node_index = idx; }

  void SetBestAction(int64_t action, double lower_bound) {
    _best_action = action;
    _lower_bound = lower_bound;
  }
};

void CreateBeliefTreeNode(std::shared_ptr<BeliefTreeNode> parent,
                          int64_t action, int64_t observation,
                          const BeliefParticles& belief,
                          const std::vector<int64_t>& action_space,
                          QLearning::QLearningPolicy policy,
                          SimInterface* sim) {
  const auto [a_best, U] =
      UpperBoundEvaluation(belief, sim, action_space, policy);
  const auto new_tree_node = BeliefTreeNode(
      belief, a_best, U,
      FindRLower(sim, belief, action_space, policy.num_sims,
                 policy.ep_convergence_threshold, policy.sim_depth));
  parent->AddChild(action, observation,
                   std::make_shared<BeliefTreeNode>(new_tree_node));
}

#endif /* !_BELIEFTREE_H_ */
