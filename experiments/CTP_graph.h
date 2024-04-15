// For use with CTP.cpp
// Use this file to define the graph used in the Canadian Traveller Problem
// example This file contains a pre-generated graph, but new graphs can be
// generated using CTP_generator.py

#pragma once
#include <unordered_map>
#include <vector>

struct pairhash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};
const std::vector<int> CTPNodes = {0, 1, 2,  3,  4,  5,  6, 7,
                                   8, 9, 10, 11, 12, 13, 14};
const std::unordered_map<std::pair<int, int>, double, pairhash> CTPEdges = {
    {{0, 2}, 1},   {{0, 12}, 1}, {{0, 4}, 1},   {{0, 9}, 1},   {{0, 3}, 1},
    {{1, 13}, 1},  {{2, 9}, 1},  {{2, 14}, 1},  {{2, 12}, 1},  {{3, 4}, 1},
    {{3, 5}, 1},   {{3, 6}, 1},  {{4, 5}, 1},   {{4, 9}, 1},   {{5, 6}, 1},
    {{5, 13}, 1},  {{5, 8}, 1},  {{5, 9}, 1},   {{6, 13}, 1},  {{7, 10}, 1},
    {{7, 8}, 1},   {{7, 11}, 1}, {{7, 14}, 1},  {{8, 13}, 1},  {{8, 14}, 1},
    {{8, 9}, 1},   {{9, 14}, 1}, {{10, 12}, 1}, {{10, 11}, 1}, {{10, 14}, 1},
    {{11, 12}, 1}, {{12, 14}, 1}};
const std::unordered_map<std::pair<int, int>, double, pairhash> CTPStochEdges =
    {{{0, 4}, 0.51},  {{1, 13}, 0.01}, {{2, 9}, 0.62},  {{2, 12}, 0.19},
     {{3, 4}, 0.55},  {{3, 5}, 0.93},  {{5, 13}, 0.24}, {{5, 8}, 0.34},
     {{6, 13}, 0.13}, {{7, 8}, 0.15},  {{7, 14}, 0.76}, {{8, 14}, 0.84},
     {{9, 14}, 0.96}};
const int CTPOrigin = 1;
const int CTPGoal = 12;
