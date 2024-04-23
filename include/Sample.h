#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>

namespace MCVI {

/// @brief Sample from a PDF
int64_t SamplePDF(const std::unordered_map<int64_t, double>& pdf,
                  std::mt19937_64& rng);

std::vector<std::pair<int64_t, double>> weightedShuffle(
    const std::unordered_map<int64_t, double>& pdf, std::mt19937_64& rng,
    size_t sample_cap = std::numeric_limits<size_t>::max());
}  // namespace MCVI
