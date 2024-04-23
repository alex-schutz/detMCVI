#include "Sample.h"

namespace MCVI {

static double SumPDF(const std::unordered_map<int64_t, double>& pdf) {
  double sum_p = 0.0;
  for (const auto& [s, p] : pdf) sum_p += p;
  return sum_p;
}

int64_t SamplePDF(const std::unordered_map<int64_t, double>& pdf,
                  std::mt19937_64& rng) {
  const double max_p = SumPDF(pdf);
  std::uniform_real_distribution<> dist(0, max_p);
  const double u = dist(rng);
  double sum_p = 0.0;
  for (const auto& [s, p] : pdf) {
    sum_p += p;
    if (sum_p > u) return s;
  }
  return -1;
}
std::vector<std::pair<int64_t, double>> weightedShuffle(
    const std::unordered_map<int64_t, double>& pdf, std::mt19937_64& rng,
    size_t sample_cap) {
  auto exp_dist = std::exponential_distribution<double>();

  std::vector<std::pair<double, std::pair<int64_t, double>>> index_pairs;
  index_pairs.reserve(pdf.size());
  for (const auto& elem : pdf) {
    const double p = elem.second;
    index_pairs.emplace_back(exp_dist(rng) / p, elem);
  }
  std::sort(index_pairs.begin(), index_pairs.end());

  std::vector<std::pair<int64_t, double>> indices;
  indices.reserve(std::min(pdf.size(), sample_cap));
  for (const auto& [w, pair] : index_pairs) {
    if (indices.capacity() == indices.size()) break;
    indices.emplace_back(pair);
  }
  return indices;
}

}  // namespace MCVI
