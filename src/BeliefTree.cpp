#include "BeliefTree.h"

#include <limits>

namespace MCVI {

void ObservationNode::BackUp(AlphaVectorFSC& fsc, double R_lower,
                             int64_t max_depth_sim, SimInterface* pomdp) {
  BackUpFromNextBelief();

  BackUpFromPolicyGraph(fsc, R_lower, max_depth_sim, pomdp);
}

void ObservationNode::BackUpFromNextBelief() {
  double nextLower = _next_belief->GetLower() + _sum_reward;
  double nextUpper = _next_belief->GetUpper() + _sum_reward;

  if (nextLower > _lower_bound) {
    _lower_bound = nextLower;
    _best_policy_val = nextLower;
    _best_policy_node = _next_belief->GetBestPolicyNode();
  }

  if (nextUpper < _upper_bound) _upper_bound = nextUpper;
}

ActionNode::ActionNode(int64_t action, const BeliefDistribution& belief,
                       int64_t belief_depth, const PathToTerminal& heuristic,
                       int64_t eval_depth, double eval_epsilon,
                       SimInterface* pomdp)
    : _action(action) {
  BeliefUpdate(belief, belief_depth, heuristic, eval_depth, eval_epsilon,
               pomdp);
  CalculateBounds();
}

void ActionNode::BeliefUpdate(const BeliefDistribution& belief,
                              int64_t belief_depth,
                              const PathToTerminal& heuristic,
                              int64_t eval_depth, double eval_epsilon,
                              SimInterface* pomdp) {
  std::unordered_map<int64_t, BeliefDistribution> next_beliefs;
  std::unordered_map<int64_t, double> reward_map;

  for (const auto& [state, prob] : belief) {
    if (pomdp->IsTerminal(state)) continue;
    auto [sNext, obs, reward, done] = pomdp->Step(state, GetAction());
    reward_map[obs] += reward * prob;
    auto& obs_belief = next_beliefs[obs];
    obs_belief[sNext] += prob;
  }

  // Set weight based on likelihood of observations
  for (const auto& [o, b] : next_beliefs) {
    double w = 0.0;
    for (const auto& [s, p] : b) w += p;
    // Renormalise next probabilities
    auto belief = BeliefDistribution();
    for (const auto& [s, p] : b) belief[s] = p / w;

    const auto belief_node = CreateBeliefTreeNode(
        belief, belief_depth + 1, heuristic, eval_depth, eval_epsilon, pomdp);
    const double discounted_reward =
        std::pow(pomdp->GetDiscount(), belief_depth) * reward_map[o] / w;
    _observation_edges.insert(
        {o, ObservationNode(w, discounted_reward, belief_node,
                            belief_node->GetUpper(), belief_node->GetLower())});
  }
}

void ActionNode::CalculateBounds() {
  double upper = 0;
  double lower = 0;

  for (const auto& [obs, child] : _observation_edges) {
    upper += child.GetUpper() * child.GetWeight();
    lower += child.GetLower() * child.GetWeight();
  }
  _avgUpper = upper;
  _avgLower = lower;
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

void ActionNode::BackUp(AlphaVectorFSC& fsc, double R_lower,
                        int64_t max_depth_sim, SimInterface* pomdp) {
  for (auto& [obs, obs_node] : _observation_edges)
    obs_node.BackUp(fsc, R_lower, max_depth_sim, pomdp);

  CalculateBounds();
}

void ActionNode::BackUpNoFSC() {
  for (auto& [obs, obs_node] : _observation_edges)
    obs_node.BackUpFromNextBelief();

  CalculateBounds();
}

void BeliefTreeNode::AddChild(int64_t action, const PathToTerminal& heuristic,
                              int64_t eval_depth, double eval_epsilon,
                              SimInterface* pomdp) {
  _action_edges.insert(
      {action, ActionNode(action, GetBelief(), _belief_depth, heuristic,
                          eval_depth, eval_epsilon, pomdp)});
}

void BeliefTreeNode::UpdateBestAction() {
  if (_action_edges.size() < 1) return;

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
    int64_t action, const PathToTerminal& heuristic, int64_t eval_depth,
    double eval_epsilon, SimInterface* pomdp) {
  const auto it = _action_edges.find(action);
  if (it != _action_edges.cend()) return it->second;
  AddChild(action, heuristic, eval_depth, eval_epsilon, pomdp);
  return _action_edges.at(action);
}

void BeliefTreeNode::BackUpBestActionUpperNoFSC() {
  if (_bestActUBound == -1) return;
  _action_edges.at(_bestActUBound).BackUpNoFSC();
}

void BeliefTreeNode::BackUpActions(AlphaVectorFSC& fsc, double R_lower,
                                   int64_t max_depth_sim, SimInterface* pomdp) {
  for (auto& [action, actionNode] : _action_edges)
    actionNode.BackUp(fsc, R_lower, max_depth_sim, pomdp);
}

void ObservationNode::BackUpFromPolicyGraph(AlphaVectorFSC& fsc, double R_lower,
                                            int64_t max_depth_sim,
                                            SimInterface* pomdp) {
  for (int64_t nI = 0; nI < fsc.NumNodes(); ++nI) {
    double node_policy_value_sum = 0.0;
    for (const auto& [sNext, prob] : _next_belief->GetBelief()) {
      const double V_nI_sNext =
          fsc.GetNodeAlpha(sNext, nI, R_lower, max_depth_sim, pomdp);
      node_policy_value_sum += V_nI_sNext * prob;
    }
    node_policy_value_sum *=
        std::pow(pomdp->GetDiscount(), _next_belief->GetDepth() - 1);
    node_policy_value_sum += _sum_reward;

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
    SimInterface* sim) {
  const auto U = UpperBoundEvaluation(belief, heuristic, sim->GetDiscount(),
                                      belief_depth, eval_depth);
  const auto L =
      FindRLower(sim, belief, sim->GetSizeOfA(), eval_epsilon, eval_depth);
  const auto node =
      std::make_shared<BeliefTreeNode>(belief, belief_depth, U, L);
  return node;
}

void BeliefTreeNode::GenerateGraphviz(std::ostream& out) const {
  out << "tr" << GetId() << " [label=<<B>" << _belief << "</B><BR/>"
      << "BestPolicyNode: " << GetBestPolicyNode() << "<BR/>"
      << "BestActLBound: " << GetBestActLBound() << "<BR/>"
      << "BestActUBound: " << GetBestActUBound() << "<BR/>"
      << "UpperBound: " << GetUpper() << "<BR/>" << "LowerBound: " << GetLower()
      << ">];" << std::endl;

  for (const auto& [act, actNode] : _action_edges) {
    out << "tr" << GetId() << "_" << act
        << " [shape=point, style=filled, fillcolor=black];" << std::endl;
    out << "tr" << GetId() << " -> " << "tr" << GetId() << "_" << act
        << " [label=<a: " << act << ">];" << std::endl;
    for (const auto& [obs, obsChild] : actNode.GetChildren()) {
      out << "tr" << GetId() << "_" << act << " -> " << "tr"
          << obsChild.GetBelief()->GetId() << " [label=<o: " << obs << ">];"
          << std::endl;
      obsChild.GetBelief()->GenerateGraphviz(out);
    }
  }
}

void BeliefTreeNode::DrawBeliefTree(std::ostream& ofs) const {
  ofs << "digraph BeliefTree {" << std::endl;
  GenerateGraphviz(ofs);
  ofs << "}" << std::endl;
}

void BeliefTreeNode::DrawPolicyBranch(std::ostream& out, int64_t& i) const {
  out << "po" << GetId() << " [label=<<B>" << _belief << "</B><BR/>"
      << "BestAction: " << GetBestActUBound() << "<BR/>"
      << "UpperBound: " << GetUpper() << ">];" << std::endl;
  ++i;

  const auto act = GetBestActUBound();
  if (!_action_edges.contains(act)) return;
  const auto actNode = _action_edges.at(act);
  out << "po" << GetId() << "_" << act
      << " [shape=point, style=filled, fillcolor=black];" << std::endl;
  out << "po" << GetId() << " -> " << "po" << GetId() << "_" << act
      << " [label=<a: " << act << ">];" << std::endl;
  for (const auto& [obs, obsChild] : actNode.GetChildren()) {
    out << "po" << GetId() << "_" << act << " -> " << "po"
        << obsChild.GetBelief()->GetId() << " [label=<o: " << obs << ">];"
        << std::endl;
    obsChild.GetBelief()->DrawPolicyBranch(out, i);
  }
}

int64_t BeliefTreeNode::DrawPolicyTree(std::ostream& ofs) const {
  int64_t i = 0;
  ofs << "digraph GreedyPolicyTree {" << std::endl;
  DrawPolicyBranch(ofs, i);
  ofs << "}" << std::endl;
  return i;
}

}  // namespace MCVI
