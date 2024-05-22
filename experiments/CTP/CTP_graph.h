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

/*
// ========= Randomly generated graph ========================================
const std::vector<int64_t> CTPNodes = {0, 1, 2,  3,  4,  5,  6, 7,
                                       8, 9, 10, 11, 12, 13, 14};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPEdges = {{{0, 2}, 1},   {{0, 12}, 1},  {{0, 4}, 1},   {{0, 9}, 1},
                {{0, 3}, 1},   {{1, 13}, 1},  {{2, 9}, 1},   {{2, 14}, 1},
                {{2, 12}, 1},  {{3, 4}, 1},   {{3, 5}, 1},   {{3, 6}, 1},
                {{4, 5}, 1},   {{4, 9}, 1},   {{5, 6}, 1},   {{5, 13}, 1},
                {{5, 8}, 1},   {{5, 9}, 1},   {{6, 13}, 1},  {{7, 10}, 1},
                {{7, 8}, 1},   {{7, 11}, 1},  {{7, 14}, 1},  {{8, 13}, 1},
                {{8, 14}, 1},  {{8, 9}, 1},   {{9, 14}, 1},  {{10, 12}, 1},
                {{10, 11}, 1}, {{10, 14}, 1}, {{11, 12}, 1}, {{12, 14}, 1}};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPStochEdges = {{{0, 4}, 0.51},  {{1, 13}, 0.01}, {{2, 9}, 0.62},
                     {{2, 12}, 0.19}, {{3, 4}, 0.55},  {{3, 5}, 0.93},
                     {{5, 13}, 0.24}, {{5, 8}, 0.34},  {{6, 13}, 0.13},
                     {{7, 8}, 0.15},  {{7, 14}, 0.76}, {{8, 14}, 0.84},
                     {{9, 14}, 0.96}};
const int64_t CTPOrigin = 1;
const int64_t CTPGoal = 12;
*/
/*
// ========= Trivial example ==================================================
// Change probability of stochastic edge to influence the direction of the
// policy. Higher -> policy always chooses 0->1->3, lower -> policy prefers
// 0->2->3/0->2->0->1->3
const std::vector<int64_t> CTPNodes = {0, 1, 2, 3};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPEdges = {{{0, 1}, 1}, {{0, 2}, 1}, {{1, 3}, 3}, {{2, 3}, 1}};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPStochEdges = {{{2, 3}, 0.5}};
const int64_t CTPOrigin = 0;
const int64_t CTPGoal = 3;
*/

/*
// ========= Many edge example ================================================
// There are N stochastic edges fanning out from node 1, which connect to a node
// and then straight to the goal. Edges are weighted in increasing order. The
// blockage probabilities are set so that each edge has a 1/N chance of being
// the best choice (i.e. all lower weighted edges are blocked). Weights are
// normalised so that the expected value of moving from 1 to the middle row
// is 1. Thus the average value of the optimal policy should be 1 + γ + γ².
//
// .      0      .
// .      |      .
// .      1      .
// .    / | \    .
// .   /  |  \   .
// .  3   4   5  .
// .   \  |  /   .
// .    \ | /    .
// .      2      .
#define NUM_FAN_EDGES_CTP 10
const int64_t CTPOrigin = 0;
const int64_t CTPGoal = 2;
std::vector<int64_t> CTPNodes = {0, 1, 2};
std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> CTPEdges = {
    {{0, 1}, 1}};
std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> CTPStochEdges;
struct CTPDataInitializer {
  CTPDataInitializer() {
    double prob_prod = 1.0;
    const double k = 2.0 / (NUM_FAN_EDGES_CTP + 1);
    for (int64_t i = 1; i < NUM_FAN_EDGES_CTP; ++i) {
      const int64_t node = 2 + i;
      CTPNodes.push_back(node);
      const double p = 1 - 1.0 / (NUM_FAN_EDGES_CTP * prob_prod);
      prob_prod *= p;
      CTPEdges[{1, node}] = k * i;
      CTPEdges[{2, node}] = 1.0;
      CTPStochEdges[{1, node}] = p;
    }
    CTPEdges[{1, 2 + NUM_FAN_EDGES_CTP}] = k * NUM_FAN_EDGES_CTP;
    CTPEdges[{2, 2 + NUM_FAN_EDGES_CTP}] = 1.0;
    CTPNodes.push_back(2 + NUM_FAN_EDGES_CTP);
  }
};
static CTPDataInitializer ctpDataInitializer;
*/

/*
// ========= Superfluous edge example ==========================================
// Reasonably simple problem with a section of stochastic edges that do not
// contribute to the best paths. This inflates the observation space without
// adding any value. A clever solver might prune these edges before approaching
// the problem.
const std::vector<int64_t> CTPNodes = {0, 1, 2,  3,  4,  5,  6, 7,
                                       8, 9, 10, 11, 12, 13, 14};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPEdges = {
        {{0, 1}, 1},   {{0, 2}, 1},   {{0, 3}, 1},   {{4, 7}, 1},
        {{5, 7}, 1},   {{6, 7}, 1},   {{4, 9}, 1},   {{1, 4}, 1},
        {{1, 5}, 1},   {{2, 5}, 1},   {{2, 6}, 1},   {{3, 6}, 1},
        {{8, 9}, 1},   {{8, 10}, 1},  {{8, 11}, 1},  {{8, 12}, 1},
        {{8, 13}, 1},  {{8, 14}, 1},  {{9, 10}, 1},  {{9, 14}, 1},
        {{10, 11}, 1}, {{11, 12}, 1}, {{12, 13}, 1}, {{13, 14}, 1},
};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPStochEdges = {
        {{1, 4}, 0.5},   {{1, 5}, 0.5},   {{2, 5}, 0.5},   {{2, 6}, 0.5},
        {{3, 6}, 0.5},   {{8, 9}, 0.5},   {{8, 10}, 0.5},  {{8, 11}, 0.5},
        {{8, 12}, 0.5},  {{8, 13}, 0.5},  {{8, 14}, 0.5},  {{9, 10}, 0.5},
        {{9, 14}, 0.5},  {{10, 11}, 0.5}, {{11, 12}, 0.5}, {{12, 13}, 0.5},
        {{13, 14}, 0.5},
};
const int64_t CTPOrigin = 0;
const int64_t CTPGoal = 7;
*/

/*
// ========= Diamond example ===================================================
const std::vector<int64_t> CTPNodes = {0, 1, 2,  3,  4,  5,  6,  7,
                                       8, 9, 10, 11, 12, 13, 14, 15};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPEdges = {
        {{0, 1}, 1},   {{0, 2}, 1},   {{1, 2}, 1},   {{1, 3}, 1},
        {{1, 4}, 1},   {{2, 4}, 1},   {{2, 5}, 1},   {{3, 4}, 1},
        {{3, 6}, 1},   {{3, 7}, 1},   {{4, 5}, 1},   {{4, 7}, 1},
        {{4, 8}, 1},   {{5, 8}, 1},   {{5, 9}, 1},   {{6, 7}, 1},
        {{6, 10}, 1},  {{7, 8}, 1},   {{7, 10}, 1},  {{7, 11}, 1},
        {{8, 9}, 1},   {{8, 11}, 1},  {{8, 12}, 1},  {{9, 12}, 1},
        {{10, 11}, 1}, {{10, 13}, 1}, {{11, 12}, 1}, {{11, 13}, 1},
        {{11, 14}, 1}, {{12, 14}, 1}, {{13, 14}, 1}, {{13, 15}, 1},
        {{14, 15}, 1},
};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPStochEdges = {
        {{1, 3}, 0.5},   {{1, 4}, 0.5},   {{4, 5}, 0.5},   {{4, 7}, 0.5},
        {{4, 8}, 0.5},   {{5, 8}, 0.5},   {{5, 9}, 0.5},   {{7, 10}, 0.5},
        {{7, 11}, 0.5},  {{8, 12}, 0.5},  {{9, 12}, 0.5},  {{10, 13}, 0.5},
        {{11, 13}, 0.5}, {{11, 14}, 0.5}, {{14, 15}, 0.5},
};
const int64_t CTPOrigin = 0;
const int64_t CTPGoal = 15;
*/

/*
// ========= Small observation space example ===================================
const std::vector<int64_t> CTPNodes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPEdges = {
        {{0, 1}, 1}, {{1, 2}, 1}, {{1, 3}, 1},  {{1, 5}, 1},
        {{2, 6}, 1}, {{3, 4}, 1}, {{3, 5}, 1},  {{4, 9}, 1},
        {{5, 6}, 1}, {{5, 8}, 1}, {{5, 9}, 1},  {{5, 10}, 1},
        {{6, 7}, 1}, {{7, 8}, 1}, {{8, 10}, 1}, {{9, 10}, 1},
};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPStochEdges = {{{1, 5}, 0.5}, {{5, 10}, 0.5}};
const int64_t CTPOrigin = 0;
const int64_t CTPGoal = 10;
*/

// ========= Replan adversarial example
// ================================================== Change probability of
// stochastic edge to influence the direction of the policy. Higher -> policy
// always chooses 0->1->3, lower -> policy prefers 0->2->3/0->2->0->1->3
const std::vector<int64_t> CTPNodes = {0, 1, 2, 3, 4};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPEdges = {{{0, 1}, 1},  {{0, 2}, 1.01}, {{1, 3}, 0.5}, {{2, 3}, 0.5},
                {{1, 4}, 10}, {{2, 4}, 9},    {{3, 4}, 0.5}};
const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
    CTPStochEdges = {{{2, 4}, 0.5}, {{3, 4}, 0.8}};
const int64_t CTPOrigin = 0;
const int64_t CTPGoal = 4;
