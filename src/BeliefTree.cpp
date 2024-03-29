#include "../include/BeliefTree.h"

void BeliefTreeNode::AddChild(int64_t action, int64_t observation,
                              std::shared_ptr<BeliefTreeNode> child) {
  auto& action_children = _child_nodes[action];
  action_children[observation] = child;
}

void BeliefTreeNode::SetBestAction(int64_t action, double lower_bound) {
  _best_action = action;
  _lower_bound = lower_bound;
}

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
