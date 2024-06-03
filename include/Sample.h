#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>

namespace MCVI {

class Welford {
 private:
  size_t n;
  double mean;
  double M2;
  double max_val;
  double min_val;

 public:
  Welford()
      : n(0),
        mean(0.0),
        M2(0.0),
        max_val(-std::numeric_limits<double>::infinity()),
        min_val(std::numeric_limits<double>::infinity()) {}

  void update(double x) {
    n++;
    double delta = x - mean;
    mean += delta / n;
    double delta2 = x - mean;
    M2 += delta * delta2;

    // Update max and min values
    if (x > max_val) {
      max_val = x;
    }
    if (x < min_val) {
      min_val = x;
    }
  }

  double getCount() const { return n; }

  double getMean() const { return mean; }

  double getVariance() const {
    if (n < 2)
      return 0.0;
    else
      return M2 / (n - 1);
  }

  double getMax() const { return max_val; }

  double getMin() const { return min_val; }
};

typedef struct {
  Welford complete;
  Welford off_policy;
  Welford max_iterations;
  Welford no_solution_on_policy;
  Welford no_solution_off_policy;
} EvaluationStats;

void PrintStats(const Welford& stats, const std::string& alg_name);

/// @brief Sample from a PMF
template <typename T>
double SumPMF(const std::unordered_map<T, double>& pmf) {
  double sum_p = 0.0;
  for (const auto& [s, p] : pmf) sum_p += p;
  return sum_p;
}

template <typename T>
T SamplePMF(const std::unordered_map<T, double>& pmf, std::mt19937_64& rng) {
  const double max_p = SumPMF<T>(pmf);
  std::uniform_real_distribution<> dist(0, max_p);
  const double u = dist(rng);
  double sum_p = 0.0;
  for (const auto& [s, p] : pmf) {
    sum_p += p;
    if (sum_p > u) return s;
  }
  return -1;
}

template <typename T>
std::vector<std::pair<T, double>> weightedShuffle(
    const std::unordered_map<T, double>& pmf, std::mt19937_64& rng,
    size_t sample_cap) {
  auto exp_dist = std::exponential_distribution<double>();

  std::vector<std::pair<double, std::pair<T, double>>> index_pairs;
  index_pairs.reserve(pmf.size());
  for (const auto& elem : pmf) {
    const double p = elem.second;
    index_pairs.emplace_back(exp_dist(rng) / p, elem);
  }
  std::sort(index_pairs.begin(), index_pairs.end());

  std::vector<std::pair<T, double>> indices;
  indices.reserve(std::min(pmf.size(), sample_cap));
  for (const auto& [w, pair] : index_pairs) {
    if (indices.capacity() == indices.size()) break;
    indices.emplace_back(pair);
  }
  return indices;
}

}  // namespace MCVI
