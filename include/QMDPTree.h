/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include <deque>

#include "BeliefTree.h"
#include "MCVI.h"

namespace MCVI {

bool QMDPTimeExpired(const std::chrono::steady_clock::time_point& begin,
                     int64_t max_computation_ms) {
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
  if (elapsed.count() >= max_computation_ms * 1e3) return true;

  return false;
}

bool terminalBelief(const std::shared_ptr<BeliefTreeNode> root,
                    SimInterface* pomdp) {
  for (const auto& [s, p] : root->GetBelief())
    if (!pomdp->IsTerminal(s)) return false;
  return true;
}

static double QMDPInfBoundFunc(const BeliefDistribution& /*belief*/,
                               int64_t /*belief_depth*/, int64_t /*eval_depth*/,
                               SimInterface* /*sim*/) {
  return -std::numeric_limits<double>::infinity();
}

bool QMDPIter(std::deque<std::shared_ptr<BeliefTreeNode>>& fringe,
              const OptimalPath& heuristic, int64_t max_eval_depth,
              SimInterface* pomdp) {
  if (fringe.empty()) return true;
  std::shared_ptr<BeliefTreeNode> b = fringe.back();
  fringe.pop_back();
  const int64_t eval_depth = max_eval_depth - b->GetDepth();
  if (eval_depth < 1) return false;
  if (terminalBelief(b, pomdp)) return false;
  // expand node
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    b->GetOrAddChildren(a, heuristic, eval_depth, QMDPInfBoundFunc, pomdp);
  }
  // build tree for children of best action
  b->UpdateBestAction();
  const auto actChildren = b->GetChildren(b->GetBestActUBound());
  for (const auto& [obs, obsNode] : actChildren) {
    fringe.push_back(obsNode.GetBelief());
  }
  return false;
}

int64_t RunQMDP(std::shared_ptr<BeliefTreeNode> initial_belief,
                int64_t max_computation_ms, const OptimalPath& heuristic,
                int64_t eval_depth, SimInterface* pomdp) {
  const auto start = std::chrono::steady_clock::now();
  std::deque<std::shared_ptr<BeliefTreeNode>> fringe;
  fringe.push_back(initial_belief);
  while (true) {
    if (QMDPIter(fringe, heuristic, eval_depth, pomdp)) {
      std::cout << "QMDP planning complete, spanned the graph." << std::endl;
      break;
    }
    if (QMDPTimeExpired(start, max_computation_ms)) {
      std::cout << "QMDP planning complete, reached maximum computation time."
                << std::endl;
      break;
    }
  }

  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

size_t RunQMDPEvaluation(std::shared_ptr<BeliefTreeNode> initial_belief,
                         int64_t time_sum, int64_t max_eval_steps,
                         int64_t n_eval_trials, int64_t nb_particles_b0,
                         std::mt19937_64& rng, const OptimalPath& solver,
                         std::optional<StateValueFunction> valFunc,
                         SimInterface* pomdp) {
  std::cout << "Evaluation of QMDP policy (" << max_eval_steps << " steps, "
            << n_eval_trials << " trials) at time " << time_sum / 1e6 << ":"
            << std::endl;
  const auto completed_states = EvaluationWithGreedyTreePolicy(
      initial_belief, max_eval_steps, n_eval_trials, nb_particles_b0, pomdp,
      rng, solver, valFunc, "QMDP");

  const std::string policy_tree_file =
      "qmdp_policy_tree_" + std::to_string(time_sum) + ".dot";
  std::fstream policy_tree(policy_tree_file, std::fstream::out);
  const int64_t n_greedy_nodes = initial_belief->DrawPolicyTree(policy_tree);
  policy_tree.close();
  std::remove(policy_tree_file.c_str());
  std::cout << "QMDP greedy policy tree contains " << n_greedy_nodes
            << " nodes." << std::endl;

  return completed_states.size();
}

void RunQMDPAndEvaluate(std::shared_ptr<BeliefTreeNode> initial_belief,
                        int64_t max_iter, int64_t max_computation_ms,
                        const OptimalPath& heuristic, int64_t eval_depth,
                        int64_t max_eval_steps, int64_t n_eval_trials,
                        int64_t nb_particles_b0, int64_t eval_interval_ms,
                        int64_t completion_threshold, int64_t completion_reps,
                        int64_t node_limit, std::mt19937_64& rng,
                        const OptimalPath& solver,
                        std::optional<StateValueFunction> valFunc,
                        SimInterface* pomdp) {
  std::deque<std::shared_ptr<BeliefTreeNode>> fringe;
  fringe.push_back(initial_belief);

  int64_t iter = 0;
  int64_t time_sum = 0;                          // us
  int64_t last_eval = -eval_interval_ms * 1000;  // us
  int64_t completed_times = 0;
  while (++iter <= max_iter && !fringe.empty()) {
    const auto iter_start = std::chrono::steady_clock::now();

    const bool ret = QMDPIter(fringe, heuristic, eval_depth, pomdp);

    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - iter_start);
    time_sum += elapsed.count();

    if (time_sum - last_eval >= eval_interval_ms * 1000) {
      last_eval = time_sum;
      const int64_t completed_count = RunQMDPEvaluation(
          initial_belief, time_sum, max_eval_steps, n_eval_trials,
          nb_particles_b0, rng, solver, valFunc, pomdp);
      if (completed_count >= completion_threshold)
        completed_times++;
      else
        completed_times = 0;
      if (completed_times >= completion_reps) return;

      const std::string policy_tree_file =
          "qmdp_policy_tree_" + std::to_string(time_sum) + ".dot";
      std::fstream policy_tree(policy_tree_file, std::fstream::out);
      const int64_t node_count = initial_belief->DrawPolicyTree(policy_tree);
      policy_tree.close();
      std::remove(policy_tree_file.c_str());
      if (node_count >= node_limit) {
        std::cout << "QMDP planning complete, reached node limit." << std::endl;
        return;
      }
      if (ret) break;
    }
    if (time_sum >= max_computation_ms * 1000) {
      std::cout << "QMDP planning complete, reached computation time."
                << std::endl;
      RunQMDPEvaluation(initial_belief, time_sum, max_eval_steps, n_eval_trials,
                        nb_particles_b0, rng, solver, valFunc, pomdp);
      return;
    }
  }
  if (fringe.empty())
    std::cout << "QMDP planning complete, spanned the graph." << std::endl;
  else
    std::cout << "QMDP planning complete, reached maximum iterations."
              << std::endl;
  RunQMDPEvaluation(initial_belief, time_sum, max_eval_steps, n_eval_trials,
                    nb_particles_b0, rng, solver, valFunc, pomdp);
  return;
}

}  // namespace MCVI
