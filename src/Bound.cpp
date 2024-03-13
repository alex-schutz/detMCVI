#include "../include/Bound.h"

#include "../include/QLearning.h"

double UpperBoundEvaluation(const Belief& belief, SimInterface* sim) {
  // We only use the underlying MDP dynamics of sim to compute the value
  auto q_engine = QLearning(sim, 10, 0.7, 0.005, 50);
  // Estimate the value of each state in the belief
  for (const auto& [state, prob] : belief) q_engine.EstimateValue(state);

  // Calculate the upper bound
  double V_upper_b = 0.0;
  for (const auto& [state, prob] : belief)
    V_upper_b += prob * get<0>(q_engine.MaxQ(state));

  return V_upper_b;
}
