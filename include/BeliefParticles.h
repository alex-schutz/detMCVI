/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _BELIEFPARTICLES_H_
#define _BELIEFPARTICLES_H_

#include <random>
#include <vector>

class BeliefParticles {
 private:
  std::vector<int64_t> _particles;
  mutable std::mt19937_64 _rng;

 public:
  BeliefParticles(const std::vector<int64_t> &particles,
                  uint64_t seed = std::random_device{}())
      : _particles(particles), _rng(seed) {}

  int64_t SampleOneState() const;
  size_t GetParticleSize() const { return _particles.size(); }
  double operator[](int i) { return _particles[i]; }
  bool operator==(BeliefParticles &o) { return _particles == o._particles; }
};

#endif
