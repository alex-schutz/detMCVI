#pragma once
#include <Sample.h>

#include <iostream>

#include "SimInterface.h"
#include "StateVector.h"

namespace MCVI {

using BeliefDistribution = StateMap<double>;

State SampleOneState(const BeliefDistribution& belief, std::mt19937_64& rng);

std::ostream& operator<<(std::ostream& os, const BeliefDistribution& bd);

BeliefDistribution SampleInitialBelief(int64_t N, SimInterface* pomdp);

BeliefDistribution DownsampleBelief(const BeliefDistribution& belief,
                                    int64_t max_belief_samples,
                                    std::mt19937_64& rng);
}  // namespace MCVI
