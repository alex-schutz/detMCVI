#include "BeliefDistribution.h"

namespace MCVI {
const State& SampleOneState(const BeliefDistribution& belief,
                            std::mt19937_64& rng) {
  return SamplePMF<State>(belief, rng);
}

std::ostream& operator<<(std::ostream& os, const BeliefDistribution& bd) {
  if (bd.size() > 5) {
    os << "{ States: " << bd.size() << "}";
  } else {
    os << "{ ";
    for (const auto& pair : bd) {
      os << "<";
      const auto& v = pair.first;
      for (const auto& state_elem : v) {
        os << state_elem;
        if (state_elem != v.back()) os << ", ";
      }
      os << ">: " << pair.second << ", ";
    }
    os << "}";
  }
  return os;
}
}  // namespace MCVI
