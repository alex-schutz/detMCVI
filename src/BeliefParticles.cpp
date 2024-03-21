#include "../include/BeliefParticles.h"

int64_t BeliefParticles::SampleOneState() const {
  std::uniform_int_distribution<> dist(0, _particles.size() - 1);
  return _particles[dist(_rng)];
}

std::unordered_map<int64_t, std::vector<int64_t>> BeliefUpdate(
    const BeliefParticles& b, int64_t action, int64_t num_sims,
    SimInterface* pomdp) {
  std::unordered_map<int64_t, std::vector<int64_t>> next_beliefs;
  for (int64_t i = 0; i < num_sims; ++i) {
    const int64_t state = b.SampleOneState();
    const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
    next_beliefs[obs].push_back(state);
  }
  return next_beliefs;
}
