/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <unordered_map>

#include "BeliefParticles.h"
#include "QLearning.h"
#include "SimInterface.h"

namespace MCVI {

/**
 * @brief Return an upper bound for the value of the belief.
 *
 * The upper bound value of the belief is the expected sum of each possible
 * state's MDP value
 *
 * @param belief A set of belief particles
 * @param sim A POMDP simulator object
 * @param action_space The set of accessible actions
 * @param policy Q-learning policy to use for the evaluation
 * @param seed Random seed
 * @return tuple<int64_t, double> best action, upper bound
 */
std::tuple<int64_t, double> UpperBoundEvaluation(
    const BeliefParticles& belief, SimInterface* sim,
    const std::vector<int64_t>& action_space, QLearningPolicy policy,
    uint64_t seed = std::random_device{}());

/**
 * @brief Determine the lower bound reward of the belief, by choosing an action
 * that maximises the minimum instant reward for all situations
 *
 * @param belief A set of belief particles
 * @param sim A POMDP simulator object
 * @param action_space The set of accessible actions
 * @param max_restarts Number of simulations to run
 * @param epsilon Threshold for maximum decay
 * @param max_depth Maximum depth of a simulation run
 * @return double
 */
double FindRLower(SimInterface* pomdp, const BeliefParticles& b0,
                  const std::vector<int64_t>& action_space,
                  int64_t max_restarts, double epsilon, int64_t max_depth);

}  // namespace MCVI
