#include "BeliefTree.h"

namespace MCVI {

void BeliefTreeNode::AddChild(int64_t action, int64_t observation,
                              std::shared_ptr<BeliefTreeNode> child) {
  auto& action_children = _child_nodes[action];
  action_children[observation] = child;
}

void BeliefTreeNode::SetBestAction(int64_t action, double lower_bound) {
  _best_action = action;
  _lower_bound = lower_bound;
}

std::shared_ptr<BeliefTreeNode> BeliefTreeNode::GetChild(int64_t action,
                                                         int64_t observation) {
  auto& action_child = _child_nodes[action];
  auto it = action_child.find(observation);
  if (it == action_child.end()) return nullptr;
  return it->second;
}

std::unordered_map<int64_t, std::shared_ptr<BeliefTreeNode>>
BeliefTreeNode::GetChildren(int64_t action) const {
  auto it = _child_nodes.find(action);
  if (it == _child_nodes.end()) return {};
  return it->second;
}

void CreateBeliefTreeNode(std::shared_ptr<BeliefTreeNode> parent,
                          int64_t action, int64_t observation,
                          const BeliefDistribution& belief, int64_t num_actions,
                          const PathToTerminal& heuristic, int64_t eval_depth,
                          double eval_epsilon, SimInterface* sim) {
  const auto [a_best, U] = UpperBoundEvaluation(belief, heuristic, eval_depth);
  const auto new_tree_node = BeliefTreeNode(
      belief, a_best, U,
      FindRLower(sim, belief, num_actions, eval_epsilon, eval_depth));
  parent->AddChild(action, observation,
                   std::make_shared<BeliefTreeNode>(new_tree_node));
}

std::shared_ptr<BeliefTreeNode> CreateBeliefRootNode(
    const BeliefDistribution& belief, int64_t num_actions,
    const PathToTerminal& heuristic, int64_t eval_depth, double eval_epsilon,
    SimInterface* sim) {
  const auto [a_best, U] = UpperBoundEvaluation(belief, heuristic, eval_depth);
  const auto root = std::make_shared<BeliefTreeNode>(
      belief, a_best, U,
      FindRLower(sim, belief, num_actions, eval_epsilon, eval_depth));
  return root;
}

int64_t ChooseObservation(
    const std::unordered_map<int64_t, std::shared_ptr<BeliefTreeNode>>&
        children,
    const std::unordered_map<int64_t, double>& weights, double target) {
  double best_gap = -std::numeric_limits<double>::infinity();
  int64_t best_obs = -1;
  for (const auto& [obs, belief_node] : children) {
    const double diff =
        (belief_node->GetUpper() - belief_node->GetLower()) - target;
    const double gap = diff * weights.at(obs);
    if (gap > best_gap) {
      best_gap = gap;
      best_obs = obs;
    }
  }
  if (best_obs == -1) throw std::logic_error("Failed to find best observation");
  return best_obs;
}

std::pair<int64_t, std::unordered_map<int64_t, BeliefDistribution>>
BeliefUpdate(std::shared_ptr<BeliefTreeNode> node, int64_t action,
             SimInterface* pomdp) {
  std::unordered_map<int64_t, BeliefDistribution> next_beliefs;

  double sum_r = 0.0;
  for (const auto& [state, prob] : node->GetBelief()) {
    auto [sNext, obs, reward, done] = pomdp->Step(state, action);
    sum_r += reward * prob;
    auto& obs_belief = next_beliefs[obs];
    obs_belief[sNext] += prob;
  }

  node->SetReward(action, sum_r);
  // Set weight based on likelihood of observations
  int64_t most_prob_obs = -1;
  double best_obs_prob = -0.0;
  std::unordered_map<int64_t, BeliefDistribution> belief_map;
  for (const auto& [o, b] : next_beliefs) {
    double w = 0.0;
    for (const auto& [s, p] : b) w += p;
    node->SetWeight(action, o, w);
    if (w > best_obs_prob) {
      best_obs_prob = w;
      most_prob_obs = o;
    }
    // Renormalise next probabilities
    auto& belief = belief_map[o];
    for (const auto& [s, p] : b) belief[s] = p / w;
  }

  return std::make_pair(most_prob_obs, belief_map);
}

}  // namespace MCVI
