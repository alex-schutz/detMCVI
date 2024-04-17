#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>

namespace MCVI {

/// @brief Sample from a PDF
int64_t SamplePDF(const std::unordered_map<int64_t, double>& pdf);

/// @brief Sample a PDF and delete the sampled element from the pdf. Used for
/// sampling without replacement.
/// @return The sampled state and original probability of that state
std::pair<int64_t, double> SamplePDFDestructive(
    std::unordered_map<int64_t, double>& pdf);

}  // namespace MCVI
