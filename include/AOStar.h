/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include "BeliefTree.h"
#include "MCVI.h"

namespace MCVI {

bool AOStarTimeExpired(const std::chrono::steady_clock::time_point& begin,
                       int64_t max_computation_ms) {
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - begin);
  if (elapsed.count() >= max_computation_ms) {
    std::cout << "AO* planning complete, reached maximum computation time."
              << std::endl;
    return true;
  }
  return false;
}

void AOStarFinalise(
    const std::vector<std::shared_ptr<BeliefTreeNode>>& backup_list) {
  for (auto it = backup_list.rbegin(); it < backup_list.rend(); ++it) {
    (*it)->BackUpBestActionUpperNoFSC();
    (*it)->UpdateBestAction();
  }
}

void RunAOStar(std::shared_ptr<BeliefTreeNode> initial_belief, int64_t max_iter,
               int64_t max_computation_ms, const PathToTerminal& heuristic,
               int64_t eval_depth, double eval_epsilon, SimInterface* pomdp) {
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a)
    initial_belief->GetOrAddChildren(a, heuristic, eval_depth, eval_epsilon,
                                     pomdp);
  initial_belief->UpdateBestAction();

  const auto iter_start = std::chrono::steady_clock::now();
  int64_t iter = 0;
  while (++iter <= max_iter) {
    std::vector<std::shared_ptr<BeliefTreeNode>> traversal_list = {
        initial_belief};
    std::vector<std::shared_ptr<BeliefTreeNode>> to_expand;

    size_t i = 0;
    while (i < traversal_list.size()) {
      const auto belief_node = traversal_list[i];
      for (const auto& [obs, obsNode] :
           belief_node->GetChildren(belief_node->GetBestActUBound())) {
        if (obsNode.GetBelief()->GetBestActUBound() == -1) {  // Leaf node
          to_expand.push_back(obsNode.GetBelief());
          continue;
        }
        traversal_list.push_back(obsNode.GetBelief());
      }
      ++i;
    }

    if (to_expand.empty()) {
      std::cout << "AO* planning complete, reached policy convergence."
                << std::endl;
      return;
    }

    std::vector<std::shared_ptr<BeliefTreeNode>> backup_list = traversal_list;
    for (const auto& node : to_expand) {
      for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a)
        node->GetOrAddChildren(a, heuristic, eval_depth, eval_epsilon, pomdp);
      backup_list.push_back(node);

      if (AOStarTimeExpired(iter_start, max_computation_ms)) return;
    }

    AOStarFinalise(backup_list);
    if (AOStarTimeExpired(iter_start, max_computation_ms)) return;
  }
  std::cout << "AO* planning complete, reached maximum iterations."
            << std::endl;
}

bool FringeInGraph(
    const std::vector<std::pair<std::shared_ptr<BeliefTreeNode>,
                                std::vector<std::shared_ptr<BeliefTreeNode>>>>&
        fringe,
    const std::vector<std::shared_ptr<BeliefTreeNode>>& graph) {
  for (const auto& node_ptr : fringe)
    if (std::find(graph.begin(), graph.end(), node_ptr.first) != graph.end())
      return true;
  return false;
}

std::pair<std::shared_ptr<BeliefTreeNode>,
          std::vector<std::shared_ptr<BeliefTreeNode>>>
ChooseNode(std::vector<std::pair<std::shared_ptr<BeliefTreeNode>,
                                 std::vector<std::shared_ptr<BeliefTreeNode>>>>&
               fringe,
           const std::vector<std::shared_ptr<BeliefTreeNode>>& graph,
           std::mt19937_64& rng) {
  if (fringe.empty()) return {nullptr, {}};

  // Create a vector containing nodes that exist in both fringe and graph
  std::vector<std::pair<std::shared_ptr<BeliefTreeNode>,
                        std::vector<std::shared_ptr<BeliefTreeNode>>>>
      common_nodes;
  std::copy_if(
      fringe.begin(), fringe.end(), std::back_inserter(common_nodes),
      [&graph](const std::pair<std::shared_ptr<BeliefTreeNode>,
                               std::vector<std::shared_ptr<BeliefTreeNode>>>&
                   node_ptr) {
        return std::find(graph.begin(), graph.end(), node_ptr.first) !=
               graph.end();
      });
  if (common_nodes.empty()) return {nullptr, {}};

  // Randomly select a node
  std::uniform_int_distribution<size_t> distribution(0,
                                                     common_nodes.size() - 1);
  auto it = std::next(common_nodes.begin(), distribution(rng));
  return *it;
}

void RunAOStar2(std::shared_ptr<BeliefTreeNode> initial_belief,
                int64_t max_iter, int64_t max_computation_ms,
                const PathToTerminal& heuristic, int64_t eval_depth,
                double eval_epsilon, std::mt19937_64& rng,
                SimInterface* pomdp) {
  std::vector<std::pair<std::shared_ptr<BeliefTreeNode>,
                        std::vector<std::shared_ptr<BeliefTreeNode>>>>
      fringe = {{initial_belief, {}}};
  std::vector<std::shared_ptr<BeliefTreeNode>> graph = {initial_belief};

  const auto iter_start = std::chrono::steady_clock::now();
  int64_t iter = 0;
  while (++iter <= max_iter && FringeInGraph(fringe, graph)) {
    const auto [belief_node, history] = ChooseNode(fringe, graph, rng);

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
      const auto actNode = belief_node->GetOrAddChildren(
          a, heuristic, eval_depth, eval_epsilon, pomdp);
      for (const auto& [obs, obsNode] : actNode.GetChildren())
        fringe.push_back({obsNode.GetBelief(), new_history});
    }

    // back up the graph
    AOStarFinalise(new_history);

    // rebuild the the graph
    graph = {initial_belief};
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

    if (AOStarTimeExpired(iter_start, max_computation_ms)) return;
  }
  if (!FringeInGraph(fringe, graph))
    std::cout << "AO* planning complete, spanned the graph." << std::endl;
  else
    std::cout << "AO* planning complete, reached maximum iterations."
              << std::endl;
}

std::vector<std::pair<int64_t, std::vector<State>>> RunAOStarAndEvaluate(
    std::shared_ptr<BeliefTreeNode> initial_belief, int64_t max_iter,
    int64_t max_computation_ms, const PathToTerminal& heuristic,
    int64_t eval_depth, double eval_epsilon, int64_t max_eval_steps,
    int64_t n_eval_trials, int64_t nb_particles_b0, int64_t eval_interval_ms,
    int64_t completion_threshold, int64_t completion_reps, std::mt19937_64& rng,
    const PathToTerminal& ptt, SimInterface* pomdp) {
  std::vector<std::pair<std::shared_ptr<BeliefTreeNode>,
                        std::vector<std::shared_ptr<BeliefTreeNode>>>>
      fringe = {{initial_belief, {}}};
  std::vector<std::shared_ptr<BeliefTreeNode>> graph = {initial_belief};

  int64_t iter = 0;
  int64_t time_sum = 0;
  int64_t last_eval = -eval_interval_ms;
  int64_t completed_times = 0;
  std::vector<std::pair<int64_t, std::vector<State>>> completed_time_states;
  while (++iter <= max_iter && FringeInGraph(fringe, graph)) {
    const auto iter_start = std::chrono::steady_clock::now();
    const auto [belief_node, history] = ChooseNode(fringe, graph, rng);

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
      const auto actNode = belief_node->GetOrAddChildren(
          a, heuristic, eval_depth, eval_epsilon, pomdp);
      for (const auto& [obs, obsNode] : actNode.GetChildren())
        fringe.push_back({obsNode.GetBelief(), new_history});
    }

    // back up the graph
    AOStarFinalise(new_history);

    // rebuild the the graph
    graph = {initial_belief};
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

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - iter_start);
    time_sum += elapsed.count();

    if (time_sum - last_eval >= eval_interval_ms) {
      last_eval = time_sum;
      std::cout << "Evaluation of alternative (AO* greedy) policy ("
                << max_eval_steps << " steps, " << n_eval_trials
                << " trials) at time " << time_sum / 1000.0 << ":" << std::endl;
      const auto completed_states = EvaluationWithGreedyTreePolicy(
          initial_belief, max_eval_steps, n_eval_trials, nb_particles_b0, pomdp,
          rng, ptt, "AO*");
      completed_time_states.push_back({time_sum, completed_states});
      std::fstream policy_tree(
          "greedy_policy_tree_" + std::to_string(time_sum) + ".dot",
          std::fstream::out);
      const int64_t n_greedy_nodes =
          initial_belief->DrawPolicyTree(policy_tree);
      policy_tree.close();
      std::cout << "AO* greedy policy tree contains " << n_greedy_nodes
                << " nodes." << std::endl;
      if (completed_states.size() >= (size_t)completion_threshold)
        completed_times++;
      else
        completed_times = 0;
      if (completed_times >= completion_reps) return completed_time_states;
    }
    if (time_sum >= max_computation_ms) return completed_time_states;
  }
  if (!FringeInGraph(fringe, graph))
    std::cout << "AO* planning complete, spanned the graph." << std::endl;
  else
    std::cout << "AO* planning complete, reached maximum iterations."
              << std::endl;

  std::cout << "Evaluation of alternative (AO* greedy) policy ("
            << max_eval_steps << " steps, " << n_eval_trials
            << " trials) at time " << time_sum / 1000.0 << ":" << std::endl;
  EvaluationWithGreedyTreePolicy(initial_belief, max_eval_steps, n_eval_trials,
                                 nb_particles_b0, pomdp, rng, ptt, "AO*");
  std::fstream policy_tree(
      "greedy_policy_tree_" + std::to_string(time_sum) + ".dot",
      std::fstream::out);
  const int64_t n_greedy_nodes = initial_belief->DrawPolicyTree(policy_tree);
  policy_tree.close();
  std::cout << "AO* greedy policy tree contains " << n_greedy_nodes << " nodes."
            << std::endl;
  return completed_time_states;
}

}  // namespace MCVI
