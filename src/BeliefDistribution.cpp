#include "BeliefDistribution.h"

namespace MCVI {
int64_t SampleOneState(const BeliefDistribution& belief) {
  return SamplePDF(belief);
}

std::ostream& operator<<(std::ostream& os, const BeliefDistribution& bd) {
  os << "{ ";
  for (const auto& pair : bd) os << pair.first << ": " << pair.second << ", ";
  os << "}";
  return os;
}
}  // namespace MCVI
