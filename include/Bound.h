/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <functional>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "BeliefDistribution.h"
#include "ShortestPath.h"
#include "SimInterface.h"

namespace MCVI {

/**
 * @brief Method to calculate the shortest path in a deterministic POMDP from a
 * given state to a terminal state.
 */
class PathToTerminal : public MaximiseReward {
 public:
  PathToTerminal(SimInterface* pomdp) : pomdp(pomdp), terminalStates() {}

  bool hasPathToTerminal(const State& source, int64_t max_depth) const;

  std::vector<std::tuple<int64_t, State, double>> getSuccessors(
      const State& state) const override;

 private:
  SimInterface* pomdp;
  mutable std::unordered_set<State, StateHash, StateEqual> terminalStates;
};

/**
 * @brief Return an upper bound for the value of the belief and the best action
 * in this belief.
 *
 * The upper bound value of the belief is the expected sum of each possible
 * state's MDP value
 */
double UpperBoundEvaluation(const BeliefDistribution& belief,
                            const PathToTerminal& solver, double gamma,
                            int64_t belief_depth, int64_t max_depth);

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
                  double epsilon, int64_t max_depth);

double CalculateUpperBound(const BeliefDistribution& belief,
                           int64_t belief_depth, int64_t eval_depth,
                           const PathToTerminal& heuristic, SimInterface* sim);

// belief, belief_depth, eval_depth, sim
using BoundFunction = std::function<double(const BeliefDistribution&, int64_t,
                                           int64_t, SimInterface*)>;

double CalculateLowerBound(const BeliefDistribution& belief,
                           int64_t belief_depth, int64_t eval_depth,
                           const BoundFunction& func, SimInterface* sim);

}  // namespace MCVI
