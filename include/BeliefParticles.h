/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <random>
#include <unordered_map>
#include <vector>

#include "SimInterface.h"

namespace MCVI {

class BeliefParticles {
 private:
  std::vector<int64_t> _particles;
  mutable std::mt19937_64 _rng;

 public:
  BeliefParticles() = default;
  BeliefParticles(const std::vector<int64_t>& particles,
                  uint64_t seed = std::random_device{}())
      : _particles(particles), _rng(seed) {}

  bool operator==(const BeliefParticles& other) const {
    return _particles == other._particles;
  }

  /// @brief Return one state sampled from the set of particles
  int64_t SampleOneState() const;

  /// @brief Return the number of particles in the belief
  size_t GetParticleCount() const { return _particles.size(); }

  /// @brief Return the entire vector of state particles
  const std::vector<int64_t>& GetParticles() const { return _particles; }

  double operator[](int64_t i) { return _particles[i]; }
  bool operator==(BeliefParticles& o) { return _particles == o._particles; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const BeliefParticles& obj) {
    std::map<int64_t, int> countMap;
    for (const auto& particle : obj._particles) {
      countMap[particle]++;
    }
    os << "Belief: ";
    for (const auto& pair : countMap)
      os << pair.first << ": " << pair.second << " ";
    return os;
  }
};

}  // namespace MCVI
