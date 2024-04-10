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

const std::vector<int> CTPNodes = {0, 1, 2, 3, 4, 5, 6, 7};
const std::unordered_map<std::pair<int, int>, double, pairhash> CTPEdges = {
    {{0, 5}, 4.0},  {{0, 7}, 3.0},  {{0, 2}, 2.24}, {{0, 6}, 1000},
    {{1, 5}, 3.16}, {{2, 7}, 1.41}, {{2, 5}, 3.61}, {{2, 3}, 3.61},
    {{3, 5}, 5.1},  {{3, 4}, 1.0},  {{3, 6}, 2.24}, {{3, 7}, 3.61},
    {{4, 7}, 4.24}, {{5, 6}, 6.08}};
const std::unordered_map<std::pair<int, int>, double, pairhash> CTPStochEdges =
    {{{0, 7}, 0.91}, {{1, 5}, 0.85}, {{2, 7}, 0.45},
     {{2, 5}, 0.1},  {{2, 3}, 0.37}, {{3, 7}, 0.67}};
const int CTPOrigin = 6;
const int CTPGoal = 0;
