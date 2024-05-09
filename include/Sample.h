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
        max_val(std::numeric_limits<double>::lowest()),
        min_val(std::numeric_limits<double>::max()) {}

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

/// @brief Sample from a PMF
int64_t SamplePMF(const std::unordered_map<int64_t, double>& pmf,
                  std::mt19937_64& rng);

std::vector<std::pair<int64_t, double>> weightedShuffle(
    const std::unordered_map<int64_t, double>& pmf, std::mt19937_64& rng,
    size_t sample_cap = std::numeric_limits<size_t>::max());
}  // namespace MCVI
