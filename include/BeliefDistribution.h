#pragma once
#include <cstdint>
#include <random>
#include <unordered_map>

namespace MCVI {

using BeliefDistribution = std::unordered_map<int64_t, double>;

int64_t SampleOneState(const BeliefDistribution& belief);

}  // namespace MCVI
