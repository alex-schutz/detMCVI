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
                          const BeliefParticles& belief,
                          const std::vector<int64_t>& action_space,
                          const QLearning& heuristic, SimInterface* sim) {
  const auto [a_best, U] =
      UpperBoundEvaluation(belief, action_space, heuristic);
  const auto new_tree_node = BeliefTreeNode(
      belief, a_best, U,
      FindRLower(sim, belief, action_space, heuristic.GetPolicy().num_sims,
                 heuristic.GetPolicy().ep_convergence_threshold,
                 heuristic.GetPolicy().sim_depth));
  parent->AddChild(action, observation,
                   std::make_shared<BeliefTreeNode>(new_tree_node));
}

std::shared_ptr<BeliefTreeNode> CreateBeliefRootNode(
    const BeliefParticles& belief, const std::vector<int64_t>& action_space,
    const QLearning& heuristic, SimInterface* sim) {
  const auto [a_best, U] =
      UpperBoundEvaluation(belief, action_space, heuristic);
  const auto root = std::make_shared<BeliefTreeNode>(
      belief, a_best, U,
      FindRLower(sim, belief, action_space, heuristic.GetPolicy().num_sims,
                 heuristic.GetPolicy().ep_convergence_threshold,
                 heuristic.GetPolicy().sim_depth));
  return root;
}

void SampleBeliefs(
    std::shared_ptr<BeliefTreeNode> node, int64_t state, int64_t depth,
    int64_t max_depth, int64_t nb_sim, const std::vector<int64_t>& action_space,
    SimInterface* pomdp, const QLearning& heuristic,
    std::vector<std::shared_ptr<BeliefTreeNode>>& traversal_list) {
  if (depth >= max_depth) return;
  if (node == nullptr) throw std::logic_error("Invalid node");

  const int64_t action = node->GetBestAction();
  // should choose an observation that maximises root's (U-L), currently just
  // choose the received observation
  auto [sNext, obs, reward, done] = pomdp->Step(state, action);

  auto child = node->GetChild(action, obs);
  if (child == nullptr) {
    const auto [o, next_beliefs] = BeliefUpdate(node, action, nb_sim, pomdp);
    obs = o;
    for (const auto& [ob, b_next] : next_beliefs)
      CreateBeliefTreeNode(node, action, ob, b_next, action_space, heuristic,
                           pomdp);
  }

  traversal_list.push_back(node);
  SampleBeliefs(node->GetChild(action, obs), state, depth + 1, max_depth,
                nb_sim, action_space, pomdp, heuristic, traversal_list);
}

static bool CmpPairSize(const std::pair<int64_t, std::vector<int64_t>>& p1,
                        const std::pair<int64_t, std::vector<int64_t>>& p2) {
  return p1.second.size() < p2.second.size();
}

std::pair<int64_t, std::unordered_map<int64_t, BeliefParticles>> BeliefUpdate(
    std::shared_ptr<BeliefTreeNode> node, int64_t action, int64_t nb_sim,
    SimInterface* pomdp) {
  std::unordered_map<int64_t, std::vector<int64_t>> next_beliefs;

  double sum_r = 0.0;
  for (int i = 0; i < nb_sim; ++i) {
    const int64_t state = node->GetParticles().SampleOneState();
    auto [sNext, obs, reward, done] = pomdp->Step(state, action);
    sum_r += reward;
    next_beliefs[obs].push_back(state);
  }

  node->SetReward(action, sum_r / nb_sim);
  for (const auto& [o, v] : next_beliefs)
    node->SetWeight(action, o, v.size() / (double)nb_sim);

  const auto most_prob_obs = std::max_element(
      std::begin(next_beliefs), std::end(next_beliefs), CmpPairSize);

  std::unordered_map<int64_t, BeliefParticles> belief_map;
  for (const auto& [o, v] : next_beliefs) belief_map[o] = BeliefParticles(v);
  return std::make_pair(most_prob_obs->first, belief_map);
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
