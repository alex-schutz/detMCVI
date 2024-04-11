/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <tuple>
#include <unordered_map>

#include "BeliefDistribution.h"
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
 * @param heuristic Pre-trained Q-learning heuristic to use for the evaluation
 * @return tuple<int64_t, double> best action, upper bound
 */
std::tuple<int64_t, double> UpperBoundEvaluation(const BeliefParticles& belief,
                                                 int64_t num_actions,
                                                 const QLearning& heuristic);

/**
 * @brief Return an upper bound for the value of the belief and the best action
 * in this belief.
 *
 * The upper bound value of the belief is the expected sum of each possible
 * state's MDP value
 *
 * @param belief A belief distribution
 * @param solver A shortest path solver for the det-POMDP
 * @param max_depth Max depth of a simulation run
 * @return tuple<int64_t, double> best action, upper bound
 */
std::tuple<int64_t, double> UpperBoundEvaluation(
    const BeliefDistribution& belief, const PathToTerminal& solver,
    int64_t max_depth);

/**
 * @brief Determine the lower bound reward of the belief, by choosing an action
 * that maximises the minimum instant reward for all situations
 *
 * @param belief A set of belief particles
 * @param sim A POMDP simulator object
 * @param epsilon Threshold for maximum decay
 * @param max_depth Maximum depth of a simulation run
 * @return double
 */
double FindRLower(SimInterface* pomdp, const BeliefDistribution& b0,
                  int64_t num_actions, double epsilon, int64_t max_depth);

}  // namespace MCVI
