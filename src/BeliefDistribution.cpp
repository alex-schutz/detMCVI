#include "BeliefDistribution.h"

namespace MCVI {
int64_t SampleOneState(const BeliefDistribution& belief, std::mt19937_64& rng) {
  return SamplePMF(belief, rng);
}

std::ostream& operator<<(std::ostream& os, const BeliefDistribution& bd) {
  if (bd.size() > 5) {
    os << "{ States: " << bd.size() << "}";
  } else {
    os << "{ ";
    for (const auto& pair : bd) os << pair.first << ": " << pair.second << ", ";
    os << "}";
  }
  return os;
}
}  // namespace MCVI
