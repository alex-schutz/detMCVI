#pragma once
#include <Sample.h>

#include <iostream>

namespace MCVI {

using BeliefDistribution = std::unordered_map<int64_t, double>;

int64_t SampleOneState(const BeliefDistribution& belief, std::mt19937_64& rng);

std::ostream& operator<<(std::ostream& os, const BeliefDistribution& bd);

}  // namespace MCVI
