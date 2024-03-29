#include "../include/Bound.h"

#include <algorithm>
#include <limits>

static bool CmpPair(const std::pair<int64_t, double>& p1,
                    const std::pair<int64_t, double>& p2) {
  return p1.second < p2.second;
}

tuple<int64_t, double> UpperBoundEvaluation(
    const BeliefParticles& belief, SimInterface* sim,
    const std::vector<int64_t>& action_space, QLearning::QLearningPolicy policy,
    uint64_t seed) {
  auto q_engine = QLearning(sim, policy, seed);
  q_engine.Train(belief);

  std::unordered_map<int64_t, double> action_vals;
  for (const auto& a : action_space) {
    for (const auto& state : belief.GetParticles())
      action_vals[a] += q_engine.GetQValue(state, a);
  }

  // Calculate the upper bound and best action
  const auto best =
      std::max_element(std::begin(action_vals), std::end(action_vals), CmpPair);
  return std::make_tuple(best->first, best->second / belief.GetParticleCount());
}

double FindRLower(SimInterface* pomdp, const BeliefParticles& b0,
                  const std::vector<int64_t>& action_space,
                  int64_t max_restarts, double epsilon, int64_t max_depth) {
  std::unordered_map<int64_t, double> action_min_reward;
  for (const auto& action : action_space) {
    double min_reward = std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < max_restarts; ++i) {
      int64_t state = b0.SampleOneState();
      int64_t step = 0;
      while ((step < max_depth) &&
             (std::pow(pomdp->GetDiscount(), step) > epsilon)) {
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
  return max_min_reward / (1 - pomdp->GetDiscount());
}
