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

void SampleBeliefs(
    std::shared_ptr<BeliefTreeNode> node, int64_t state, int64_t depth,
    int64_t max_depth, SimInterface* pomdp, const PathToTerminal& heuristic,
    int64_t eval_depth, double eval_epsilon,
    std::vector<std::shared_ptr<BeliefTreeNode>>& traversal_list) {
  if (depth >= max_depth) return;
  if (node == nullptr) throw std::logic_error("Invalid node");

  const int64_t action = node->GetBestAction();
  // TODO: should choose an observation that maximises root's (U-L)
  auto [sNext, obs, reward, done] = pomdp->Step(state, action);

  auto child = node->GetChild(action, obs);
  if (child == nullptr) {
    const auto [o, next_beliefs] = BeliefUpdate(node, action, pomdp);
    obs = o;
    for (const auto& [ob, b_next] : next_beliefs)
      CreateBeliefTreeNode(node, action, ob, b_next, pomdp->GetSizeOfA(),
                           heuristic, eval_depth, eval_epsilon, pomdp);
  }

  traversal_list.push_back(node);
  SampleBeliefs(node->GetChild(action, obs), state, depth + 1, max_depth, pomdp,
                heuristic, eval_depth, eval_epsilon, traversal_list);
}

static bool CmpPairSize(const std::pair<int64_t, std::vector<int64_t>>& p1,
                        const std::pair<int64_t, std::vector<int64_t>>& p2) {
  return p1.second.size() < p2.second.size();
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

double UpdateUpperBound(std::shared_ptr<BeliefTreeNode> node, double gamma,
                        int64_t depth) {
  if (!node) return 0.0;
  const auto action = node->GetBestAction();
  if (node->GetFSCNodeIndex() == -1 || !node->HasReward(action))
    return std::pow(gamma, depth) * node->GetUpper();
  const double R_a = node->GetReward(action);
  double esti_U_future = 0.0;
  for (const auto& [o, w] : node->GetWeights(action)) {
    const double U_child =
        UpdateUpperBound(node->GetChild(action, o), gamma, depth + 1);
    esti_U_future += w * U_child;
  }

  node->SetUpper(R_a + gamma * esti_U_future);
  return node->GetUpper();
}

}  // namespace MCVI
