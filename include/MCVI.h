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

class MCVI {
 private:
  SimInterface* _pomdp;
  AlphaVectorFSC _fsc;
  QLearningPolicy _policy;

 public:
  MCVI(SimInterface* pomdp, const AlphaVectorFSC& init_fsc,
       const QLearningPolicy& policy)
      : _pomdp(pomdp), _fsc(init_fsc), _policy(policy) {}

  /// @brief Run the MCVI planner
  /// @param b0 Initial belief particles
  /// @param fsc Initial FSC
  /// @param pomdp Simulator
  /// @param max_depth_sim Maximum depth to simulate
  /// @param nb_sample Number of samples in belief expansion
  /// @param nb_iter Number of tree traversals
  /// @param policy Q-learning policy for bound seting
  /// @return The FSC for the pomdp
  AlphaVectorFSC MCVIPlanning(const BeliefParticles& b0, int64_t max_depth_sim,
                              int64_t nb_sample, int64_t nb_iter);

  /// @brief Simulate an FSC execution from an initial belief
  void SimulationWithFSC(const BeliefParticles& b0, int64_t steps) const;

 private:
  /// @brief Perform a monte-carlo backup on the fsc node given by `nI_new`.
  void BackUp(std::shared_ptr<BeliefTreeNode> Tr_node, int64_t max_depth_sim,
              int64_t nb_sample, const std::vector<int64_t>& action_space,
              const std::vector<int64_t>& observation_space);

  /// @brief Simulate a trajectory using the policy graph beginning at node nI
  /// and the given state, returning the discounted reward of the simulation
  double SimulateTrajectory(int64_t nI, int64_t state, int64_t max_depth) const;

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
};

}  // namespace MCVI
