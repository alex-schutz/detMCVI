/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <iostream>

#include "AlphaVectorFSC.h"
#include "BeliefParticles.h"
#include "SimInterface.h"

namespace MCVI {

class MCVIPlanner {
 private:
  SimInterface* _pomdp;
  AlphaVectorFSC _fsc;
  BeliefParticles _b0;
  QLearning _heuristic;
  mutable std::mt19937_64 _rng;

 public:
  MCVIPlanner(SimInterface* pomdp, const AlphaVectorFSC& init_fsc,
              const BeliefParticles& init_belief,
              const QLearningPolicy& q_policy)
      : _pomdp(pomdp),
        _fsc(init_fsc),
        _b0(init_belief),
        _heuristic(_pomdp, q_policy),
        _rng(std::random_device{}()) {
    _heuristic.Train(_b0);
  }

  /// @brief Run the MCVI planner
  /// @param max_depth_sim Maximum depth to simulate
  /// @param nb_sample Number of samples in belief expansion
  /// @param epsilon Threshold for difference between upper and lower bounds
  /// @param max_nb_iter Maximum number of tree traversals
  /// @param actions Action names
  /// @param observations Observation names
  /// @return The FSC for the pomdp
  AlphaVectorFSC Plan(int64_t max_depth_sim, int64_t nb_sample, double epsilon,
                      int64_t max_nb_iter,
                      const std::vector<std::string>& actions = {},
                      const std::vector<std::string>& observations = {});

  /// @brief Simulate an FSC execution from the initial belief
  void SimulationWithFSC(int64_t steps) const;

 private:
  /// @brief Perform a monte-carlo backup on the fsc node given by `nI_new`.
  void BackUp(std::shared_ptr<BeliefTreeNode> Tr_node, double R_lower,
              int64_t max_depth_sim, int64_t nb_sample,
              const std::vector<int64_t>& action_space,
              const std::vector<int64_t>& observation_space);

  /// @brief Simulate a trajectory using the policy graph beginning at node nI
  /// and the given state, returning the discounted reward of the simulation
  double SimulateTrajectory(int64_t nI, int64_t state, int64_t max_depth,
                            double R_lower) const;

  /// @brief Find the node in the V_a_o_n set of the node with the highest value
  std::pair<double, int64_t> FindMaxValueNode(const AlphaVectorNode& node,
                                              int64_t a, int64_t o) const;

  /// @brief Find a node matching the given node and edges, or insert it if it
  /// does not exist
  int64_t FindOrInsertNode(const AlphaVectorNode& node,
                           const AlphaVectorFSC::EdgeMap& edges,
                           const std::vector<int64_t>& observation_space);

  /// @brief Insert the given node into the fsc
  int64_t InsertNode(const AlphaVectorNode& node,
                     const AlphaVectorFSC::EdgeMap& edges);

  int64_t RandomAction(const std::vector<int64_t>& action_space) const;
};

}  // namespace MCVI
