#include "Sample.h"

namespace MCVI {

int64_t SamplePDF(const std::unordered_map<int64_t, double>& pdf) {
  static std::mt19937_64 rng(std::random_device{}());
  static std::uniform_real_distribution<> dist(0, 1);
  const double u = dist(rng);
  double sum_p = 0.0;
  for (const auto& [s, p] : pdf) {
    sum_p += p;
    if (sum_p > u) return s;
  }
  return -1;
}

std::vector<std::pair<int64_t, double>> CreateCDF(
    const std::unordered_map<int64_t, double>& pdf) {
  std::vector<std::pair<int64_t, double>> cdf = {{-1, 0.0}};
  double sum_p = 0.0;
  for (const auto& [s, p] : pdf) {
    sum_p += p;
    cdf.push_back({s, sum_p});
  }
  return cdf;
}

size_t SampleCDF(const std::vector<std::pair<int64_t, double>>& cdf) {
  static std::mt19937_64 rng(std::random_device{}());
  std::uniform_real_distribution<> dist(0, cdf.back().second);
  const double u = dist(rng);
  for (size_t i = 0; i < cdf.size(); ++i)
    if (cdf[i].second > u) return i;
  return -1;
}

std::pair<int64_t, double> SampleCDFDestructive(
    std::vector<std::pair<int64_t, double>>& cdf) {
  const size_t idx = SampleCDF(cdf);
  const int64_t s = cdf[idx].first;
  const double prob = cdf[idx].second - cdf[idx - 1].second;
  cdf.erase(cdf.begin() + idx);
  return {s, prob};
}

}  // namespace MCVI
