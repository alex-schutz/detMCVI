/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _BELIEFPARTICLES_H_
#define _BELIEFPARTICLES_H_

#include <random>
#include <unordered_map>
#include <vector>

#include "SimInterface.h"

class BeliefParticles {
 private:
  std::vector<int64_t> _particles;
  mutable std::mt19937_64 _rng;

 public:
  BeliefParticles(const std::vector<int64_t>& particles,
                  uint64_t seed = std::random_device{}())
      : _particles(particles), _rng(seed) {}

  /// @brief Return one state sampled from the set of particles
  int64_t SampleOneState() const;

  /// @brief Return the number of particles in the belief
  size_t GetParticleCount() const { return _particles.size(); }

  /// @brief Return the entire vector of state particles
  const std::vector<int64_t>& GetParticles() const { return _particles; }

  double operator[](int64_t i) { return _particles[i]; }
  bool operator==(BeliefParticles& o) { return _particles == o._particles; }
};

/// @brief Generate a set of next beliefs mapped by observation, obtained by
/// taking `action` in belief state `b`
std::unordered_map<int64_t, std::vector<int64_t>> BeliefUpdate(
    const BeliefParticles& b, int64_t action, int64_t num_sims,
    SimInterface* pomdp);

#endif /* !_BELIEFPARTICLES_H_ */
