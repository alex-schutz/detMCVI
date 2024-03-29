/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _MCVIPLANNER_H_
#define _MCVIPLANNER_H_

#include <iostream>

#include "AlphaVectorFSC.h"
#include "BeliefParticles.h"
#include "SimInterface.h"

class MCVI {
 private:
  /* data */
 public:
  MCVI(/* args */) {}
};

/// @brief Perform a monte-carlo backup on the fsc node given by `nI_new`.
void BackUp(BeliefTreeNode& Tr_node, AlphaVectorFSC& fsc, int64_t max_depth_sim,
            int64_t nb_sample, SimInterface* pomdp,
            const std::vector<int64_t>& action_space,
            const std::vector<int64_t>& observation_space);

/// @brief Simulate a trajectory using the policy graph beginning at node nI and
/// the given state, returning the discounted reward of the simulation
double SimulateTrajectory(int64_t nI, AlphaVectorFSC& fsc, int64_t state,
                          int64_t max_depth, SimInterface* pomdp);

/// @brief Find the node in the V_a_o_n set of the node with the highest value
std::pair<double, int64_t> FindMaxValueNode(const AlphaVectorNode& node,
                                            int64_t a, int64_t o);

/// @brief Find a node matching the given node and edges, or insert it if it
/// does not exist
int64_t FindOrInsertNode(const AlphaVectorNode& node,
                         const AlphaVectorFSC::EdgeMap& edges,
                         const std::vector<int64_t>& observation_space,
                         AlphaVectorFSC& fsc);

/// @brief Insert the given node into the fsc
int64_t InsertNode(const AlphaVectorNode& node,
                   const AlphaVectorFSC::EdgeMap& edges, AlphaVectorFSC& fsc);

#endif /* !_MCVIPLANNER_H_ */
