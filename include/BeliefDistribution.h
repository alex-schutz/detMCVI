#pragma once
#include <Sample.h>

namespace MCVI {

using BeliefDistribution = std::unordered_map<int64_t, double>;

int64_t SampleOneState(const BeliefDistribution& belief);

}  // namespace MCVI
