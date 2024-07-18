#include "Bound.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace MCVI {

double UpperBoundEvaluation(const BeliefDistribution& belief,
                            const PathToTerminal& solver, double gamma,
                            int64_t belief_depth, int64_t max_depth) {
  double V_upper_bound = 0.0;
  for (const auto& [state, prob] : belief) {
    const auto [reward, path] = solver.getMaxReward(state, max_depth);
    V_upper_bound += std::pow(gamma, belief_depth) * reward * prob;
  }

  return V_upper_bound;
}

double FindRLower(SimInterface* pomdp, const BeliefDistribution& b0,
                  double epsilon, int64_t max_depth) {
  const int64_t action_default = 0;
  double belief_val = 0;
  for (const auto& [s, p] : b0) {
    State state = s;
    int64_t step = 0;
    while ((step < max_depth) &&
           (std::pow(pomdp->GetDiscount(), step) >= epsilon)) {
      const auto [sNext, obs, reward, done] =
          pomdp->Step(state, action_default);
      belief_val += std::pow(pomdp->GetDiscount(), step) * reward * p;
      if (done) break;
      state = sNext;
      ++step;
    }
  }
  return belief_val;
}

std::vector<std::tuple<int64_t, State, double, bool>>
PathToTerminal::getSuccessors(const State& state) const {
  const bool isTerminal = pomdp->IsTerminal(state);
  if (isTerminal) return {{-1, state, 0.0, true}};
  std::vector<std::tuple<int64_t, State, double, bool>> successors;
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    State sNext;
    const auto& reward = pomdp->applyActionToState(state, a, sNext);
    successors.push_back({a, sNext, reward, isTerminal});
  }
  return successors;
}

bool PathToTerminal::hasPathToTerminal(
    const State& source,
    const std::vector<std::pair<int64_t, State>>& path) const {
  if (pomdp->IsTerminal(source)) return true;
  for (const auto& [action, state] : path)
    if (pomdp->IsTerminal(state)) return true;

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
