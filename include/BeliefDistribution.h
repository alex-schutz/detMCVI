#pragma once
#include <Sample.h>

#include <iostream>

#include "StateVector.h"

namespace MCVI {

using BeliefDistribution = StateMap<double>;

const State& SampleOneState(const BeliefDistribution& belief,
                            std::mt19937_64& rng);

std::ostream& operator<<(std::ostream& os, const BeliefDistribution& bd);

}  // namespace MCVI
