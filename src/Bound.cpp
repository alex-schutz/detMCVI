#include "Bound.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace MCVI {

static bool CmpPair(const std::pair<int64_t, double>& p1,
                    const std::pair<int64_t, double>& p2) {
  return p1.second < p2.second;
}

double UpperBoundEvaluation(const BeliefDistribution& belief,
                            const PathToTerminal& solver, double gamma,
                            int64_t belief_depth, int64_t max_depth) {
  double V_upper_bound = 0.0;
  for (const auto& [state, prob] : belief) {
    const auto [reward, path] = solver.getMaxReward(state, max_depth, gamma);
    V_upper_bound += std::pow(gamma, belief_depth) * reward * prob;
  }

  return V_upper_bound;
}

double FindRLower(SimInterface* pomdp, const BeliefDistribution& b0,
                  double epsilon, int64_t max_depth) {
  std::unordered_map<int64_t, double> action_min_reward;
  for (int64_t action = 0; action < pomdp->GetSizeOfA(); ++action) {
    double min_reward = std::numeric_limits<double>::infinity();
    for (const auto& [s, p] : b0) {
      State state = s;
      int64_t step = 0;
      while ((step < max_depth) &&
             (std::pow(pomdp->GetDiscount(), step) >= epsilon)) {
        const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
        if (reward < min_reward) {
          action_min_reward[action] = reward;
          min_reward = reward;
        }
        if (done) break;
        state = sNext;
        ++step;
      }
    }
  }
  const double max_min_reward =
      std::max_element(std::begin(action_min_reward),
                       std::end(action_min_reward), CmpPair)
          ->second;
  if (pomdp->GetDiscount() >= 1) return max_min_reward * max_depth;
  return max_min_reward / (1 - pomdp->GetDiscount());
}

std::vector<std::tuple<int64_t, State, double, bool>>
PathToTerminal::getSuccessors(const State& state) const {
  if (terminalStates.contains(state)) return {{-1, state, 0.0, true}};
  std::vector<std::tuple<int64_t, State, double, bool>> successors;
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    const auto& [sNext, o, reward, done] = pomdp->Step(state, a);
    successors.push_back({a, sNext, reward, false});
    if (done) terminalStates.insert(sNext);
  }
  return successors;
}

bool PathToTerminal::hasPathToTerminal(const State& source,
                                       int64_t max_depth) const {
  const auto [reward, path] =
      getMaxReward(source, max_depth, pomdp->GetDiscount());
  State state = source;
  for (const auto& [action, state] : path) {
    if (terminalStates.contains(state)) return true;
  }
  return false;
}

double CalculateUpperBound(const BeliefDistribution& belief,
                           int64_t belief_depth, int64_t eval_depth,
                           const PathToTerminal& heuristic, SimInterface* sim) {
  const auto H_Uval = sim->GetHeuristicUpper(belief, eval_depth - belief_depth);
  if (H_Uval.has_value())
    return std::pow(sim->GetDiscount(), belief_depth) * H_Uval.value();
  return UpperBoundEvaluation(belief, heuristic, sim->GetDiscount(),
                              belief_depth, eval_depth);
}

double CalculateLowerBound(const BeliefDistribution& belief,
                           int64_t belief_depth, int64_t eval_depth,
                           const BoundFunction& func, SimInterface* sim) {
  const auto H_Lval = sim->GetHeuristicLower(belief, eval_depth);
  if (H_Lval.has_value())
    return std::pow(sim->GetDiscount(), belief_depth) * H_Lval.value();
  return func(belief, belief_depth, eval_depth, sim);
}

}  // namespace MCVI
