#include "BeliefDistribution.h"

namespace MCVI {
State SampleOneState(const BeliefDistribution& belief, std::mt19937_64& rng) {
  return SamplePMF<State>(belief, rng);
}

std::ostream& operator<<(std::ostream& os, const BeliefDistribution& bd) {
  if (bd.size() > 5) {
    os << "{ States: " << bd.size() << "}";
  } else {
    os << "{ ";
    for (const auto& pair : bd) {
      os << "[";
      const auto& v = pair.first;
      for (const auto& state_elem : v) {
        os << state_elem << ", ";
      }
      os << "]: " << pair.second << ", ";
    }
    os << "}";
  }
  return os;
}

BeliefDistribution SampleInitialBelief(int64_t N, SimInterface* pomdp) {
  StateMap<int64_t> state_counts;
  for (int64_t i = 0; i < N; ++i) state_counts[pomdp->SampleStartState()] += 1;
  auto init_belief = BeliefDistribution();
  for (const auto& [state, count] : state_counts)
    init_belief[state] = (double)count / N;
  return init_belief;
}

BeliefDistribution DownsampleBelief(const BeliefDistribution& belief,
                                    int64_t max_belief_samples,
                                    std::mt19937_64& rng) {
  const auto shuffled_init = weightedShuffle(belief, rng, max_belief_samples);
  double prob_sum = 0.0;
  for (const auto& [state, prob] : shuffled_init) prob_sum += prob;
  auto b = BeliefDistribution();
  for (const auto& [state, prob] : shuffled_init) b[state] = prob / prob_sum;
  return b;
}
}  // namespace MCVI
