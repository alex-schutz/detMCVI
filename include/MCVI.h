/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <atomic>
#include <chrono>
#include <iostream>

#include "AlphaVectorFSC.h"
#include "BeliefDistribution.h"
#include "BeliefTree.h"
#include "Bound.h"
#include "SimInterface.h"

namespace MCVI {

class MCVIPlanner {
 private:
  SimInterface* _pomdp;
  AlphaVectorFSC _fsc;
  BeliefDistribution _b0;
  PathToTerminal _heuristic;
  std::mt19937_64& _rng;

 public:
  MCVIPlanner(SimInterface* pomdp, const AlphaVectorFSC& init_fsc,
              const BeliefDistribution& init_belief,
              const PathToTerminal& heuristic, std::mt19937_64& rng)
      : _pomdp(pomdp),
        _fsc(init_fsc),
        _b0(init_belief),
        _heuristic(heuristic),
        _rng(rng) {}

  /// @brief Run the MCVI planner
  /// @param max_depth_sim Maximum depth to simulate
  /// @param epsilon Threshold for difference between upper and lower bounds
  /// @param max_nb_iter Maximum number of tree traversals
  /// @return The FSC for the pomdp
  std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> Plan(
      int64_t max_depth_sim, double epsilon, int64_t max_nb_iter,
      int64_t max_computation_ms, int64_t eval_depth, double eval_epsilon,
      std::atomic<bool>& exit_flag);

  // fsc, root node, converged, timed out, reached max iter
  std::tuple<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>, double, bool,
             bool, bool>
  PlanIncrement(std::shared_ptr<BeliefTreeNode> Tr_root, double R_lower,
                int64_t iter, int64_t ms_remaining, int64_t max_depth_sim,
                double epsilon, int64_t max_nb_iter, int64_t eval_depth,
                double eval_epsilon, std::atomic<bool>& exit_flag);

  double MCVIIteration(std::shared_ptr<BeliefTreeNode> Tr_root, double R_lower,
                       int64_t ms_remaining, int64_t max_depth_sim,
                       int64_t eval_depth, double eval_epsilon,
                       std::atomic<bool>& exit_flag);

  // run evaluation after each iteration
  std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> PlanAndEvaluate(
      int64_t max_depth_sim, double epsilon, int64_t max_nb_iter,
      int64_t max_computation_ms, int64_t eval_depth, double eval_epsilon,
      int64_t max_eval_steps, int64_t n_eval_trials, int64_t nb_particles_b0,
      int64_t eval_interval_ms, int64_t completion_threshold,
      int64_t completion_reps, std::atomic<bool>& exit_flag);

  /// @brief Simulate an FSC execution from the initial belief
  void SimulationWithFSC(int64_t steps) const;

  /// @brief Evaluate the FSC bounds through multiple simulations. Reverts to
  /// greedy policy when policy runs out
  int64_t EvaluationWithSimulationFSC(int64_t max_steps, int64_t num_sims,
                                      int64_t init_belief_samples) const;

  int64_t EvaluationWithSimulationFSCFixedDist(
      int64_t max_steps, std::vector<State> init_dist) const;

  std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> PlanAndEvaluate2(
      int64_t max_depth_sim, double epsilon, int64_t max_nb_iter,
      int64_t max_computation_ms, int64_t eval_depth, double eval_epsilon,
      int64_t max_eval_steps,
      const std::vector<std::pair<int64_t, std::vector<State>>>& eval_data,
      int64_t completion_threshold, int64_t completion_reps);

 private:
  int64_t GetFirstAction(std::shared_ptr<BeliefTreeNode> Tr_node,
                         double R_lower, int64_t max_depth_sim,
                         int64_t eval_depth, double eval_epsilon);

  /// @brief Perform a monte-carlo backup on the given belief node
  void BackUp(std::shared_ptr<BeliefTreeNode> Tr_node, double R_lower,
              int64_t max_depth_sim, int64_t eval_depth, double eval_epsilon);

  /// @brief Find a node matching the given node and edges, or insert it if it
  /// does not exist
  int64_t FindOrInsertNode(const AlphaVectorNode& node,
                           const std::unordered_map<int64_t, int64_t>& edges);

  /// @brief Insert the given node into the fsc
  int64_t InsertNode(const AlphaVectorNode& node,
                     const std::unordered_map<int64_t, int64_t>& edges);

  void SampleBeliefs(
      std::vector<std::shared_ptr<BeliefTreeNode>>& traversal_list,
      int64_t eval_depth, double eval_epsilon, double target, double R_lower,
      int64_t max_depth_sim);
};

std::vector<State> EvaluationWithGreedyTreePolicy(
    std::shared_ptr<BeliefTreeNode> root, int64_t max_steps, int64_t num_sims,
    int64_t init_belief_samples, SimInterface* pomdp, std::mt19937_64& rng,
    const PathToTerminal& ptt, const std::string& alg_name);

BeliefDistribution SampleInitialBelief(int64_t N, SimInterface* pomdp);

BeliefDistribution DownsampleBelief(const BeliefDistribution& belief,
                                    int64_t max_belief_samples,
                                    std::mt19937_64& rng);
}  // namespace MCVI
