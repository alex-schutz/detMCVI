#include "BeliefDistribution.h"

namespace MCVI {
int64_t SampleOneState(const BeliefDistribution& belief) {
  return SamplePDF(belief);
}
}  // namespace MCVI
