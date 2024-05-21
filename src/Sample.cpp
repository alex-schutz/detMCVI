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
    stream << num;
    return stream.str();
  }
}

void PrintStats(const Welford& stats, const std::string& alg_name) {
  std::cout << alg_name << " Count: " << stats.getCount() << std::endl;
  std::cout << alg_name << " Average reward: " << print_inf(stats.getMean())
            << std::endl;
  std::cout << alg_name << " Highest reward: " << print_inf(stats.getMax())
            << std::endl;
  std::cout << alg_name << " Lowest reward: " << print_inf(stats.getMin())
            << std::endl;
  std::cout << alg_name
            << " Reward variance: " << print_inf(stats.getVariance())
            << std::endl;
}

static double SumPMF(const std::unordered_map<int64_t, double>& pmf) {
  double sum_p = 0.0;
  for (const auto& [s, p] : pmf) sum_p += p;
  return sum_p;
}

int64_t SamplePMF(const std::unordered_map<int64_t, double>& pmf,
                  std::mt19937_64& rng) {
  const double max_p = SumPMF(pmf);
  std::uniform_real_distribution<> dist(0, max_p);
  const double u = dist(rng);
  double sum_p = 0.0;
  for (const auto& [s, p] : pmf) {
    sum_p += p;
    if (sum_p > u) return s;
  }
  return -1;
}
std::vector<std::pair<int64_t, double>> weightedShuffle(
    const std::unordered_map<int64_t, double>& pmf, std::mt19937_64& rng,
    size_t sample_cap) {
  auto exp_dist = std::exponential_distribution<double>();

  std::vector<std::pair<double, std::pair<int64_t, double>>> index_pairs;
  index_pairs.reserve(pmf.size());
  for (const auto& elem : pmf) {
    const double p = elem.second;
    index_pairs.emplace_back(exp_dist(rng) / p, elem);
  }
  std::sort(index_pairs.begin(), index_pairs.end());

  std::vector<std::pair<int64_t, double>> indices;
  indices.reserve(std::min(pmf.size(), sample_cap));
  for (const auto& [w, pair] : index_pairs) {
    if (indices.capacity() == indices.size()) break;
    indices.emplace_back(pair);
  }
  return indices;
}

}  // namespace MCVI
