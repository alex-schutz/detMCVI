#include "Sample.h"

namespace MCVI {

static double SumPDF(const std::unordered_map<int64_t, double>& pdf) {
  double sum_p = 0.0;
  for (const auto& [s, p] : pdf) sum_p += p;
  return sum_p;
}

int64_t SamplePDF(const std::unordered_map<int64_t, double>& pdf) {
  static std::mt19937_64 rng(std::random_device{}());
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

std::pair<int64_t, double> SamplePDFDestructive(
    std::unordered_map<int64_t, double>& pdf) {
  if (pdf.size() == 0) return {-1, 0.0};
  const int64_t s = SamplePDF(pdf);
  const double prob = pdf[s];
  pdf.erase(s);
  return {s, prob};
}

}  // namespace MCVI
