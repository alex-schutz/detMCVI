/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include "BeliefTree.h"
#include "MCVI.h"

namespace MCVI {

enum class AOSearchType { none, random, depth_first, breadth_first };

using BeliefTreeNodeHistory =
    std::pair<std::shared_ptr<BeliefTreeNode>,
              std::vector<std::shared_ptr<BeliefTreeNode>>>;

bool AOStarTimeExpired(const std::chrono::steady_clock::time_point& begin,
                       int64_t max_computation_ms) {
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
  if (elapsed.count() >= max_computation_ms * 1000) {
    std::cout << "AO* planning complete, reached maximum computation time."
              << std::endl;
    return true;
  }
  return false;
}

bool FringeInGraph(const std::vector<BeliefTreeNodeHistory>& fringe,
                   const std::vector<std::shared_ptr<BeliefTreeNode>>& graph) {
  for (const auto& node_ptr : fringe)
    if (std::find(graph.begin(), graph.end(), node_ptr.first) != graph.end())
      return true;
  return false;
}

bool CmpNodeDepth(const BeliefTreeNodeHistory& n1,
                  const BeliefTreeNodeHistory& n2) {
  return n1.first->GetDepth() < n2.first->GetDepth();
}

BeliefTreeNodeHistory ChooseNode(
    std::vector<BeliefTreeNodeHistory>& fringe,
    const std::vector<std::shared_ptr<BeliefTreeNode>>& graph,
    std::mt19937_64& rng, AOSearchType method) {
  if (fringe.empty()) return {nullptr, {}};

  // Create a vector containing nodes that exist in both fringe and graph
  std::vector<BeliefTreeNodeHistory> common_nodes;
  std::copy_if(fringe.begin(), fringe.end(), std::back_inserter(common_nodes),
               [&graph](const BeliefTreeNodeHistory& node_ptr) {
                 return std::find(graph.begin(), graph.end(), node_ptr.first) !=
                        graph.end();
               });
  if (common_nodes.empty()) return {nullptr, {}};

  switch (method) {
    case AOSearchType::random: {
      std::uniform_int_distribution<size_t> distribution(
          0, common_nodes.size() - 1);
      return common_nodes.at(distribution(rng));
    }
    case AOSearchType::depth_first:
      return *std::max_element(common_nodes.begin(), common_nodes.end(),
                               CmpNodeDepth);
    case AOSearchType::breadth_first:
      return *std::min_element(common_nodes.begin(), common_nodes.end(),
                               CmpNodeDepth);
    default:
      return common_nodes.at(0);
  }
  return common_nodes.at(0);
}

static double InfBoundFunc(const BeliefDistribution& /*belief*/,
                           int64_t /*belief_depth*/, int64_t /*eval_depth*/,
                           SimInterface* /*sim*/) {
  return -std::numeric_limits<double>::infinity();
}

void AOStarIter(std::vector<std::shared_ptr<BeliefTreeNode>>& graph,
                std::vector<BeliefTreeNodeHistory>& fringe,
                const PathToTerminal& heuristic, int64_t eval_depth,
                std::mt19937_64& rng, SimInterface* pomdp,
                AOSearchType method) {
  const auto [belief_node, history] = ChooseNode(fringe, graph, rng, method);

  // remove node from fringe set
  auto it = std::find(fringe.begin(), fringe.end(),
                      std::make_pair(belief_node, history));
  if (it != fringe.end())
    fringe.erase(it);
  else
    throw std::logic_error("Cannot find fringe node to erase");

  // expand node
  auto new_history = history;
  new_history.push_back(belief_node);
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    const auto actNode = belief_node->GetOrAddChildren(a, heuristic, eval_depth,
                                                       InfBoundFunc, pomdp);
    for (const auto& [obs, obsNode] : actNode.GetChildren())
      fringe.push_back({obsNode.GetBelief(), new_history});
  }

  // back up the graph
  for (auto it = new_history.rbegin(); it < new_history.rend(); ++it) {
    (*it)->BackUpBestActionUpperNoFSC();
    (*it)->UpdateBestAction();
  }

  // rebuild the the graph
  graph = {graph.at(0)};
  size_t i = 0;
  while (i < graph.size()) {
    auto node = graph.at(i);
    if (node->GetBestActUBound() != -1) {
      for (const auto& [obs, obsNode] :
           node->GetChildren(node->GetBestActUBound()))
        graph.push_back(obsNode.GetBelief());
    }
    ++i;
  }
}  // namespace MCVI

void RunAOStar(std::shared_ptr<BeliefTreeNode> initial_belief, int64_t max_iter,
               int64_t max_computation_ms, const PathToTerminal& heuristic,
               int64_t eval_depth, std::mt19937_64& rng, SimInterface* pomdp,
               AOSearchType method = AOSearchType::random) {
  std::vector<BeliefTreeNodeHistory> fringe = {{initial_belief, {}}};
  std::vector<std::shared_ptr<BeliefTreeNode>> graph = {initial_belief};

  const auto ao_start = std::chrono::steady_clock::now();
  int64_t iter = 0;
  while (++iter <= max_iter && FringeInGraph(fringe, graph)) {
    AOStarIter(graph, fringe, heuristic, eval_depth, rng, pomdp, method);
    if (AOStarTimeExpired(ao_start, max_computation_ms)) {
      std::cout << "AO* planning complete, reached maximum computation time."
                << std::endl;
      return;
    }
  }
  if (!FringeInGraph(fringe, graph))
    std::cout << "AO* planning complete, spanned the graph." << std::endl;
  else
    std::cout << "AO* planning complete, reached maximum iterations."
              << std::endl;
}

size_t RunAOEvaluation(std::shared_ptr<BeliefTreeNode> initial_belief,
                       int64_t time_sum, int64_t max_eval_steps,
                       int64_t n_eval_trials, int64_t nb_particles_b0,
                       std::mt19937_64& rng, const PathToTerminal& ptt,
                       std::optional<StateValueFunction> valFunc,
                       SimInterface* pomdp) {
  std::cout << "Evaluation of alternative (AO* greedy) policy ("
            << max_eval_steps << " steps, " << n_eval_trials
            << " trials) at time " << time_sum / 1e6 << ":" << std::endl;
  const auto completed_states = EvaluationWithGreedyTreePolicy(
      initial_belief, max_eval_steps, n_eval_trials, nb_particles_b0, pomdp,
      rng, ptt, valFunc, "AO*");

  const std::string policy_tree_file =
      "greedy_policy_tree_" + std::to_string(time_sum) + ".dot";
  std::fstream policy_tree(policy_tree_file, std::fstream::out);
  const int64_t n_greedy_nodes = initial_belief->DrawPolicyTree(policy_tree);
  policy_tree.close();
  std::remove(policy_tree_file.c_str());
  std::cout << "AO* greedy policy tree contains " << n_greedy_nodes << " nodes."
            << std::endl;

  return completed_states.size();
}

void RunAOStarAndEvaluate(std::shared_ptr<BeliefTreeNode> initial_belief,
                          int64_t max_iter, int64_t max_computation_ms,
                          const PathToTerminal& heuristic, int64_t eval_depth,
                          int64_t max_eval_steps, int64_t n_eval_trials,
                          int64_t nb_particles_b0, int64_t eval_interval_ms,
                          int64_t completion_threshold, int64_t completion_reps,
                          std::mt19937_64& rng, const PathToTerminal& ptt,
                          std::optional<StateValueFunction> valFunc,
                          SimInterface* pomdp,
                          AOSearchType method = AOSearchType::random) {
  std::vector<std::pair<std::shared_ptr<BeliefTreeNode>,
                        std::vector<std::shared_ptr<BeliefTreeNode>>>>
      fringe = {{initial_belief, {}}};
  std::vector<std::shared_ptr<BeliefTreeNode>> graph = {initial_belief};

  int64_t iter = 0;
  int64_t time_sum = 0;
  int64_t last_eval = -eval_interval_ms * 1000;
  int64_t completed_times = 0;
  while (++iter <= max_iter && FringeInGraph(fringe, graph)) {
    const auto iter_start = std::chrono::steady_clock::now();

    AOStarIter(graph, fringe, heuristic, eval_depth, rng, pomdp, method);

    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - iter_start);
    time_sum += elapsed.count();

    if (time_sum - last_eval >= eval_interval_ms * 1000) {
      last_eval = time_sum;
      const int64_t completed_count = RunAOEvaluation(
          initial_belief, time_sum, max_eval_steps, n_eval_trials,
          nb_particles_b0, rng, ptt, valFunc, pomdp);
      if (completed_count >= completion_threshold)
        completed_times++;
      else
        completed_times = 0;
      if (completed_times >= completion_reps) return;
    }
    if (time_sum >= max_computation_ms * 1000) {
      std::cout << "AO* planning complete, reached computation time."
                << std::endl;
      RunAOEvaluation(initial_belief, time_sum, max_eval_steps, n_eval_trials,
                      nb_particles_b0, rng, ptt, valFunc, pomdp);
      return;
    }
  }
  if (!FringeInGraph(fringe, graph))
    std::cout << "AO* planning complete, spanned the graph." << std::endl;
  else
    std::cout << "AO* planning complete, reached maximum iterations."
              << std::endl;
  RunAOEvaluation(initial_belief, time_sum, max_eval_steps, n_eval_trials,
                  nb_particles_b0, rng, ptt, valFunc, pomdp);
  return;
}

}  // namespace MCVI
