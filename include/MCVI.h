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
#include "SimInterface.h"

namespace MCVI {

class MCVIPlanner {
 private:
  SimInterface* _pomdp;
  AlphaVectorFSC _fsc;
  BeliefDistribution _b0;
  PathToTerminal _heuristic;
  mutable std::mt19937_64 _rng;

 public:
  MCVIPlanner(SimInterface* pomdp, const AlphaVectorFSC& init_fsc,
              const BeliefDistribution& init_belief)
      : _pomdp(pomdp),
        _fsc(init_fsc),
        _b0(init_belief),
        _heuristic(_pomdp),
        _rng(std::random_device{}()) {}

  /// @brief Run the MCVI planner
  /// @param max_depth_sim Maximum depth to simulate
  /// @param epsilon Threshold for difference between upper and lower bounds
  /// @param max_nb_iter Maximum number of tree traversals
  /// @return The FSC for the pomdp
  AlphaVectorFSC Plan(int64_t max_depth_sim, double epsilon,
                      int64_t max_nb_iter, int64_t eval_depth,
                      double eval_epsilon);

  /// @brief Simulate an FSC execution from the initial belief
  void SimulationWithFSC(int64_t steps) const;

 private:
  /// @brief Perform a monte-carlo backup on the given belief node
  void BackUp(std::shared_ptr<BeliefTreeNode> Tr_node, double R_lower,
              int64_t max_depth_sim);

  /// @brief Simulate a trajectory using the policy graph beginning at node nI
  /// and the given state, returning the discounted reward of the simulation
  double SimulateTrajectory(int64_t nI, int64_t state, int64_t max_depth,
                            double R_lower) const;

  /// @brief Find a node matching the given node and edges, or insert it if it
  /// does not exist
  int64_t FindOrInsertNode(const AlphaVectorNode& node,
                           const std::unordered_map<int64_t, int64_t>& edges);

  /// @brief Insert the given node into the fsc
  int64_t InsertNode(const AlphaVectorNode& node,
                     const std::unordered_map<int64_t, int64_t>& edges);

  int64_t RandomAction() const;
};

}  // namespace MCVI
