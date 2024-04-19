#include "BeliefTree.h"

#include <limits>

namespace MCVI {

void ObservationNode::BackUp(AlphaVectorFSC& fsc, int64_t max_belief_samples,
                             double R_lower, int64_t max_depth_sim,
                             SimInterface* pomdp) {
  BackUpFromNextBelief();

  BackUpFromPolicyGraph(fsc, max_belief_samples, R_lower, max_depth_sim, pomdp);
}

void ObservationNode::BackUpFromNextBelief() {
  double nextLower = _next_belief->GetUpper() * _weight + _sum_reward;
  double nextUpper = _next_belief->GetLower() * _weight + _sum_reward;

  if (nextLower > _lower_bound) {
    _lower_bound = nextLower;
    _best_policy_val = nextLower;
    _best_policy_node = _next_belief->GetBestPolicyNode();
  }

  if (nextUpper < _upper_bound) _upper_bound = nextUpper;
}

ActionNode::ActionNode(int64_t action, const BeliefDistribution& belief,
                       int64_t belief_depth, int64_t max_belief_samples,
                       const PathToTerminal& heuristic, int64_t eval_depth,
                       double eval_epsilon, SimInterface* pomdp)
    : _action(action) {
  BeliefUpdate(belief, belief_depth, max_belief_samples, heuristic, eval_depth,
               eval_epsilon, pomdp);
  CalculateBounds();
}

void ActionNode::BeliefUpdate(const BeliefDistribution& belief,
                              int64_t belief_depth, int64_t max_belief_samples,
                              const PathToTerminal& heuristic,
                              int64_t eval_depth, double eval_epsilon,
                              SimInterface* pomdp) {
  std::unordered_map<int64_t, BeliefDistribution> next_beliefs;
  std::unordered_map<int64_t, double> reward_map;

  auto belief_pdf = belief;
  double prob_sum = 0.0;
  for (int64_t sample = 0; sample < max_belief_samples; ++sample) {
    const auto [state, prob] = SamplePDFDestructive(belief_pdf);
    if (state == -1) break;  // Sampled all states in belief
    prob_sum += prob;
    auto [sNext, obs, reward, done] = pomdp->Step(state, GetAction());
    reward_map[obs] += reward * prob;
    auto& obs_belief = next_beliefs[obs];
    obs_belief[sNext] += prob;
  }

  // Set weight based on likelihood of observations
  for (const auto& [o, b] : next_beliefs) {
    double w = 0.0;
    for (const auto& [s, p] : b) w += p;
    w /= prob_sum;
    // Renormalise next probabilities
    auto belief = BeliefDistribution();
    for (const auto& [s, p] : b) belief[s] = p / w;

    const auto belief_node =
        CreateBeliefTreeNode(belief, belief_depth + 1, heuristic, eval_depth,
                             eval_epsilon, max_belief_samples, pomdp);
    _observation_edges.insert(
        {o, ObservationNode(w, reward_map[o], belief_node,
                            belief_node->GetUpper(), belief_node->GetLower())});
  }
}

void ActionNode::CalculateBounds() {
  double lower = 0;
  double upper = 0;

  for (const auto& [obs, child] : _observation_edges) {
    lower += child.GetLower();
    upper += child.GetUpper();
  }
  _avgLower = lower;
  _avgUpper = upper;
}

std::shared_ptr<BeliefTreeNode> ActionNode::GetChild(
    int64_t observation) const {
  auto it = _observation_edges.find(observation);
  if (it == _observation_edges.end()) return nullptr;
  return it->second.GetBelief();
}

std::shared_ptr<BeliefTreeNode> ActionNode::ChooseObservation(
    double target) const {
  double best_gap = -std::numeric_limits<double>::infinity();
  int64_t best_obs = -1;
  for (const auto& [obs, obs_node] : _observation_edges) {
    const double gap = (obs_node.GetUpper() - obs_node.GetLower()) - target;
    if (gap > best_gap) {
      best_gap = gap;
      best_obs = obs;
    }
  }
  if (best_obs == -1) throw std::logic_error("Failed to find best observation");
  return _observation_edges.at(best_obs).GetBelief();
}

void ActionNode::BackUp(AlphaVectorFSC& fsc, int64_t max_belief_samples,
                        double R_lower, int64_t max_depth_sim,
                        SimInterface* pomdp) {
  for (auto& [obs, obs_node] : _observation_edges)
    obs_node.BackUp(fsc, max_belief_samples, R_lower, max_depth_sim, pomdp);

  CalculateBounds();
}

void BeliefTreeNode::AddChild(int64_t action, int64_t max_belief_samples,
                              const PathToTerminal& heuristic,
                              int64_t eval_depth, double eval_epsilon,
                              SimInterface* pomdp) {
  _action_edges.insert({action, ActionNode(action, GetBelief(), _belief_depth,
                                           max_belief_samples, heuristic,
                                           eval_depth, eval_epsilon, pomdp)});
}

void BeliefTreeNode::SetBestActionLBound(int64_t action, double lower_bound) {
  _bestActLBound = action;
  _lower_bound = lower_bound;
}

void BeliefTreeNode::UpdateBestAction() {
  // find best bounds at the belief
  _lower_bound = -std::numeric_limits<double>::infinity();
  _upper_bound = -std::numeric_limits<double>::infinity();
  _bestActLBound = -1;
  _bestActUBound = -1;

  for (const auto& [action, actNode] : _action_edges) {
    if (_lower_bound < actNode.GetAvgLower()) {
      _lower_bound = actNode.GetAvgLower();
      _bestActLBound = action;
    }
    if (_upper_bound < actNode.GetAvgUpper()) {
      _upper_bound = actNode.GetAvgUpper();
      _bestActUBound = action;
    }
  }
}

std::shared_ptr<BeliefTreeNode> BeliefTreeNode::GetChild(
    int64_t action, int64_t observation) const {
  auto it = _action_edges.find(action);
  if (it == _action_edges.cend()) return nullptr;
  return it->second.GetChild(observation);
}

const std::unordered_map<int64_t, ObservationNode>& BeliefTreeNode::GetChildren(
    int64_t action) const {
  auto it = _action_edges.find(action);
  if (it == _action_edges.cend())
    throw std::logic_error("No observation nodes");
  return it->second.GetChildren();
}

std::shared_ptr<BeliefTreeNode> BeliefTreeNode::ChooseObservation(
    double target) {
  auto it = _action_edges.find(_bestActUBound);
  if (it == _action_edges.cend()) UpdateBestAction();
  it = _action_edges.find(_bestActUBound);
  if (it == _action_edges.cend())
    throw std::logic_error("Could not find best action");

  return it->second.ChooseObservation(target);
}

const ActionNode& BeliefTreeNode::GetOrAddChildren(
    int64_t action, int64_t max_belief_samples, const PathToTerminal& heuristic,
    int64_t eval_depth, double eval_epsilon, SimInterface* pomdp) {
  const auto it = _action_edges.find(action);
  if (it != _action_edges.cend()) return it->second;
  AddChild(action, max_belief_samples, heuristic, eval_depth, eval_epsilon,
           pomdp);
  return _action_edges.at(action);
}

void BeliefTreeNode::BackUpActions(AlphaVectorFSC& fsc,
                                   int64_t max_belief_samples, double R_lower,
                                   int64_t max_depth_sim, SimInterface* pomdp) {
  for (auto& [action, actionNode] : _action_edges)
    actionNode.BackUp(fsc, max_belief_samples, R_lower, max_depth_sim, pomdp);
}

void ObservationNode::BackUpFromPolicyGraph(AlphaVectorFSC& fsc,
                                            int64_t max_belief_samples,
                                            double R_lower,
                                            int64_t max_depth_sim,
                                            SimInterface* pomdp) {
  for (int64_t nI = 0; nI < fsc.NumNodes(); ++nI) {
    double node_policy_value_sum = 0.0;
    auto belief_pdf = _next_belief->GetBelief();
    double prob_sum = 0.0;
    for (int64_t sample = 0; sample < max_belief_samples; ++sample) {
      const auto [sNext, prob] = SamplePDFDestructive(belief_pdf);
      if (sNext == -1) break;  // Sampled all states in belief
      prob_sum += prob;
      const double V_nI_sNext =
          fsc.GetNodeAlpha(sNext, nI, R_lower, max_depth_sim, pomdp);
      node_policy_value_sum += V_nI_sNext * prob;
    }
    node_policy_value_sum /= prob_sum;

    if (node_policy_value_sum > _best_policy_val) {
      _best_policy_val = node_policy_value_sum;
      _best_policy_node = nI;
      _lower_bound = node_policy_value_sum;
    }
  }
}

std::shared_ptr<BeliefTreeNode> CreateBeliefTreeNode(
    const BeliefDistribution& belief, int64_t belief_depth,
    const PathToTerminal& heuristic, int64_t eval_depth, double eval_epsilon,
    int64_t max_belief_samples, SimInterface* sim) {
  const auto U =
      UpperBoundEvaluation(belief, heuristic, sim->GetDiscount(), belief_depth,
                           eval_depth, max_belief_samples);
  const auto L =
      FindRLower(sim, belief, sim->GetSizeOfA(), eval_epsilon, eval_depth);
  const auto node =
      std::make_shared<BeliefTreeNode>(belief, belief_depth, U, L);
  return node;
}

}  // namespace MCVI
