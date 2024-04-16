#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>

namespace MCVI {

/// @brief Sample from a PDF (probabilities must sum to 1)
int64_t SamplePDF(const std::unordered_map<int64_t, double>& pdf);

/// @brief Generate a CDF from a PDF
std::vector<std::pair<int64_t, double>> CreateCDF(
    const std::unordered_map<int64_t, double>& pdf);

/// @brief Sample from a cdf (automatically normalised)
/// @return The index of the sampled item in the cdf
size_t SampleCDF(const std::vector<std::pair<int64_t, double>>& cdf);

/// @brief Sample a CDF and delete the sampled element from the cdf. Used for
/// sampling without replacement.
/// @return The sampled state and original probability of that state
/// (non-cumulative, non-normalised)
std::pair<int64_t, double> SampleCDFDestructive(
    std::vector<std::pair<int64_t, double>>& cdf);

}  // namespace MCVI
