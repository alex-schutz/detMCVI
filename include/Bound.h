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
class PathToTerminal : public ShortestPathFasterAlgorithm {
 public:
  PathToTerminal(SimInterface* pomdp)
      : pomdp(pomdp), terminalStates({State({})}) {}

  /// @brief Return the first action and associated total reward for the end
  /// state reachable from source within max_depth steps.
  std::tuple<int64_t, double> path(const State& source,
                                   int64_t max_depth) const;

  bool is_terminal(const State& source, int64_t max_depth) const;

  std::vector<std::tuple<State, double, int64_t>> getEdges(
      const State& state) const override;

  struct PathNode {
    int64_t id;
    int64_t action;
    std::unordered_set<State, StateHash, StateEqual> states;
    std::shared_ptr<PathNode> nextNode;
    std::vector<std::shared_ptr<PathNode>> prevNodes;
  };

  /// @brief Return the starting action node for each previously calculated path
  /// leading to a sequence of action nodes, where common sub-paths have been
  /// combined
  StateMap<std::shared_ptr<PathNode>> buildPathTree() const;

 private:
  SimInterface* pomdp;
  mutable std::unordered_set<State, StateHash, StateEqual> terminalStates;
  mutable StateMap<std::pair<int64_t, State>> paths;  // node: action, next node

  std::shared_ptr<PathNode> createPathNode(
      int64_t& id, int64_t action,
      const std::unordered_set<State, StateHash, StateEqual>& states) const;

  std::shared_ptr<PathNode> findActionChild(std::shared_ptr<PathNode> node,
                                            int64_t action) const;

  std::shared_ptr<PathNode> findOrCreateNode(std::shared_ptr<PathNode> nextNode,
                                             int64_t& id, int64_t action) const;
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
