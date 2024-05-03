/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

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
      int64_t eval_depth, double eval_epsilon);

  /// @brief Simulate an FSC execution from the initial belief
  void SimulationWithFSC(int64_t steps) const;

  /// @brief Evaluate the FSC bounds through multiple simulations. Reverts to
  /// greedy policy when policy runs out
  void EvaluationWithSimulationFSC(int64_t max_steps, int64_t num_sims) const;

 private:
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

  int64_t RandomAction() const;

  void SampleBeliefs(
      std::shared_ptr<BeliefTreeNode> node, int64_t state, int64_t depth,
      int64_t max_depth, SimInterface* pomdp, const PathToTerminal& heuristic,
      int64_t eval_depth, double eval_epsilon,
      std::vector<std::shared_ptr<BeliefTreeNode>>& traversal_list,
      double target, double R_lower, int64_t max_depth_sim);
};

void EvaluationWithGreedyTreePolicy(std::shared_ptr<BeliefTreeNode> root,
                                    int64_t max_steps, int64_t num_sims,
                                    SimInterface* pomdp, std::mt19937_64& rng);
}  // namespace MCVI
