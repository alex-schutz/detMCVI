#include "Sample.h"

#include <iostream>
#include <sstream>

namespace MCVI {

static std::string print_inf(double num) {
  if (std::isinf(num)) {
    if (num > 0)
      return "inf";
    else
      return "-inf";
  } else {
    std::ostringstream stream;
    stream << std::fixed << num;
    return stream.str();
  }
}

void PrintStats(const Welford& stats, const std::string& alg_name) {
  std::cout << alg_name << " Count: " << stats.getCount() << std::endl;
  std::cout << alg_name << " Average regret: " << print_inf(stats.getMean())
            << std::endl;
  std::cout << alg_name << " Highest regret: " << print_inf(stats.getMax())
            << std::endl;
  std::cout << alg_name << " Lowest regret: " << print_inf(stats.getMin())
            << std::endl;
  std::cout << alg_name
            << " Regret variance: " << print_inf(stats.getVariance())
            << std::endl;
}

}  // namespace MCVI
