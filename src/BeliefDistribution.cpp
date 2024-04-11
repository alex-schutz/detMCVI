#include "BeliefDistribution.h"

namespace MCVI {
int64_t SampleOneState(const BeliefDistribution& belief) {
  static std::mt19937_64 rng(std::random_device{}());
  static std::uniform_real_distribution<> dist(0, 1);
  const double u = dist(rng);
  double sum_p = 0.0;
  for (const auto& [s, p] : belief) {
    sum_p += p;
    if (sum_p > u) return s;
  }
  return -1;
}
}  // namespace MCVI
