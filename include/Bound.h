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

using StateValueFunction =
    std::function<std::pair<double, bool>(const State&, int64_t)>;

class OptimalPath : public MaximiseReward {
 public:
  OptimalPath(SimInterface* pomdp, size_t cache_capacity = 250000)
      : MaximiseReward(pomdp->GetDiscount(), cache_capacity), pomdp(pomdp) {}

  std::vector<std::tuple<int64_t, State, double, bool>> getSuccessors(
      const State& state) const override;

 private:
  SimInterface* pomdp;
};

/**
 * @brief Return an upper bound for the value of the belief and the best action
 * in this belief.
 *
 * The upper bound value of the belief is the expected sum of each possible
 * state's MDP value
 */
double UpperBoundEvaluation(const BeliefDistribution& belief,
                            const OptimalPath& solver, double gamma,
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
                           const OptimalPath& heuristic, SimInterface* sim);

// belief, belief_depth, eval_depth, sim
using BoundFunction = std::function<double(const BeliefDistribution&, int64_t,
                                           int64_t, SimInterface*)>;

double CalculateLowerBound(const BeliefDistribution& belief,
                           int64_t belief_depth, int64_t eval_depth,
                           const BoundFunction& func, SimInterface* sim);

}  // namespace MCVI
