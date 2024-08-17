/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include "BeliefTree.h"
#include "MCVI.h"

namespace MCVI {

bool QMDPTimeExpired(const std::chrono::steady_clock::time_point& begin,
                     int64_t max_computation_ms) {
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
  if (elapsed.count() >= max_computation_ms * 1000) return true;

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

bool QMDPIter(std::shared_ptr<BeliefTreeNode> root,
              const OptimalPath& heuristic, int64_t eval_depth,
              std::mt19937_64& rng, SimInterface* pomdp,
              const std::chrono::steady_clock::time_point& start,
              int64_t max_time_ms) {
  if (QMDPTimeExpired(start, max_time_ms)) return true;
  if (eval_depth < 1) return false;
  if (terminalBelief(root, pomdp)) return false;
  // expand node
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    root->GetOrAddChildren(a, heuristic, eval_depth, QMDPInfBoundFunc, pomdp);
  }
  // build tree for children of best action
  root->UpdateBestAction();
  const auto actChildren = root->GetChildren(root->GetBestActUBound());
  for (const auto& [obs, obsNode] : actChildren) {
    if (QMDPIter(obsNode.GetBelief(), heuristic, eval_depth - 1, rng, pomdp,
                 start, max_time_ms))
      return true;
  }
  return false;
}

int64_t RunQMDP(std::shared_ptr<BeliefTreeNode> initial_belief,
                int64_t max_computation_ms, const OptimalPath& heuristic,
                int64_t eval_depth, std::mt19937_64& rng, SimInterface* pomdp) {
  const auto start = std::chrono::steady_clock::now();
  QMDPIter(initial_belief, heuristic, eval_depth, rng, pomdp, start,
           max_computation_ms);

  if (QMDPTimeExpired(start, max_computation_ms)) {
    std::cout << "QMDP planning complete, reached maximum computation time."
              << std::endl;
  } else
    std::cout << "QMDP planning complete, spanned the graph." << std::endl;
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
                        int64_t max_computation_ms,
                        const OptimalPath& heuristic, int64_t eval_depth,
                        int64_t max_eval_steps, int64_t n_eval_trials,
                        int64_t nb_particles_b0, std::mt19937_64& rng,
                        const OptimalPath& solver,
                        std::optional<StateValueFunction> valFunc,
                        SimInterface* pomdp) {
  const int64_t t = RunQMDP(initial_belief, max_computation_ms, heuristic,
                            eval_depth, rng, pomdp);
  RunQMDPEvaluation(initial_belief, t, max_eval_steps, n_eval_trials,
                    nb_particles_b0, rng, solver, valFunc, pomdp);
  return;
}

}  // namespace MCVI
