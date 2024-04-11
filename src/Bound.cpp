#include "Bound.h"

#include <algorithm>
#include <limits>

namespace MCVI {

static bool CmpPair(const std::pair<int64_t, double>& p1,
                    const std::pair<int64_t, double>& p2) {
  return p1.second < p2.second;
}

std::tuple<int64_t, double> UpperBoundEvaluation(const BeliefParticles& belief,
                                                 int64_t num_actions,
                                                 const QLearning& heuristic) {
  std::unordered_map<int64_t, double> action_vals;
  for (int64_t a = 0; a < num_actions; ++a) {
    for (const auto& state : belief.GetParticles())
      action_vals[a] += heuristic.GetQValue(state, a);
  }

  // Calculate the upper bound and best action
  const auto best =
      std::max_element(std::begin(action_vals), std::end(action_vals), CmpPair);
  return std::make_tuple(best->first, best->second / belief.GetParticleCount());
}

double FindRLower(SimInterface* pomdp, const BeliefParticles& b0,
                  int64_t num_actions, int64_t max_restarts, double epsilon,
                  int64_t max_depth) {
  std::unordered_map<int64_t, double> action_min_reward;
  for (int64_t action = 0; action < num_actions; ++action) {
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

}  // namespace MCVI
