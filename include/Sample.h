#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>

namespace MCVI {

/// @brief Sample from a PMF
int64_t SamplePMF(const std::unordered_map<int64_t, double>& pmf,
                  std::mt19937_64& rng);

std::vector<std::pair<int64_t, double>> weightedShuffle(
    const std::unordered_map<int64_t, double>& pmf, std::mt19937_64& rng,
    size_t sample_cap = std::numeric_limits<size_t>::max());
}  // namespace MCVI
