#include "../include/BeliefParticles.h"

int64_t BeliefParticles::SampleOneState() const {
  std::uniform_int_distribution<> dist(0, _particles.size() - 1);
  return _particles[dist(_rng)];
}
