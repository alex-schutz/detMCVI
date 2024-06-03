#include <ShortestPath.h>
#include <gtest/gtest.h>

#include <iostream>

using namespace MCVI;

class SmallGraph : public ShortestPathFasterAlgorithm {
 public:
  SmallGraph() = default;

  std::vector<std::tuple<State, double, int64_t>> getEdges(
      const State& node) const override {
    if (node.at(0) == 1) {
      return {{{2}, 3.0, 1}, {{3}, 6.0, 2}};
    } else if (node.at(0) == 2) {
      return {{{3}, 2.0, 3}};
    } else if (node.at(0) == 3) {
      return {{{1}, -1.0, 4}, {{3}, 0.5, 5}};
    }
    return {};
  }
};

class LargeGraph : public ShortestPathFasterAlgorithm {
 public:
  LargeGraph() = default;

  std::vector<std::tuple<State, double, int64_t>> getEdges(
      const State& node) const override {
    if (node.at(0) == 1) {
      return {{{2}, 3.0, 1}, {{3}, -2.0, 1}, {{4}, 5.0, 1}, {{7}, 2.0, 1}};
    } else if (node.at(0) == 2) {
      return {{{3}, 2.0, 2}, {{5}, 1.0, 2}, {{6}, -1.0, 2}};
    } else if (node.at(0) == 3) {
      return {{{4}, 2.0, 3}, {{6}, 3.0, 3}};
    } else if (node.at(0) == 4) {
      return {{{5}, -1.0, 4}, {{7}, 4.0, 4}, {{9}, 3.0, 4}, {{3}, 2.0, 4}};
    } else if (node.at(0) == 5) {
      return {{{6}, 2.0, 5}, {{8}, -2.0, 5}, {{9}, 1.0, 5}, {{2}, -1.0, 5}};
    } else if (node.at(0) == 6) {
      return {{{7}, 1.0, 6}, {{9}, 3.0, 6}};
    } else if (node.at(0) == 7) {
      return {{{8}, 1.0, 7}, {{10}, 2.0, 7}, {{1}, 2.0, 7}};
    } else if (node.at(0) == 8) {
      return {{{9}, 4.0, 8}, {{10}, 3.0, 8}};
    } else if (node.at(0) == 9) {
      return {{{10}, 2.0, 9}, {{4}, 3.0, 9}};
    } else if (node.at(0) == 10) {
      return {{{1}, 2.0, 10}, {{2}, 1.0, 10}, {{6}, -1.0, 10}};
    }
    return {};
  }
};

TEST(ShortestPathFasterAlgorithmTest, Constructor) {
  SmallGraph graph;
  {
    const auto [costs, pred] = graph.calculate({1}, 10);
    EXPECT_EQ(costs.size(), 3);
    EXPECT_EQ(costs.at({1}), 0.0);
    EXPECT_EQ(costs.at({2}), 3.0);
    EXPECT_EQ(costs.at({3}), 5.0);

    const auto [costs2, pred2] = graph.calculate({1}, 1);
    EXPECT_EQ(costs2.size(), 3);
    EXPECT_EQ(costs.at({1}), 0.0);
    EXPECT_EQ(costs2.at({2}), 3.0);
    EXPECT_EQ(costs2.at({3}), 6.0);
  }

  {
    const auto [costs, pred] = graph.calculate({3}, 10);
    EXPECT_EQ(costs.size(), 3);
    EXPECT_EQ(costs.at({1}), -1.0);
    EXPECT_EQ(costs.at({2}), 2.0);
    EXPECT_EQ(costs.at({3}), 0.0);
  }

  {
    const auto [costs, pred] = graph.calculate({4}, 10);
    EXPECT_EQ(costs.size(), 1);
    EXPECT_EQ(costs.at({4}), 0.0);
  }
  {
    LargeGraph lgraph;
    const auto [costs, pred] = lgraph.calculate({1}, 20);
    EXPECT_EQ(costs.size(), 10);
    EXPECT_EQ(costs.at({1}), 0.0);
    EXPECT_EQ(costs.at({2}), -2.0);
    EXPECT_EQ(costs.at({3}), -2.0);
    EXPECT_EQ(costs.at({4}), 0.0);
    EXPECT_EQ(costs.at({5}), -1.0);
    EXPECT_EQ(costs.at({6}), -3.0);
    EXPECT_EQ(costs.at({7}), -2.0);
    EXPECT_EQ(costs.at({8}), -3.0);
    EXPECT_EQ(costs.at({9}), 0.0);
    EXPECT_EQ(costs.at({10}), 0.0);
  }
}

TEST(ShortestPathFasterAlgorithmTest, ReconstructPath) {
  SmallGraph graph;
  const auto [costs, pred] = graph.calculate({1}, 10);
  const auto path = graph.reconstructPath({3}, pred);
  EXPECT_EQ(path.size(), 3);
  EXPECT_EQ(path[0].first, State({1}));
  EXPECT_EQ(path[0].second, 1);
  EXPECT_EQ(path[1].first, State({2}));
  EXPECT_EQ(path[1].second, 3);
  EXPECT_EQ(path[2].first, State({3}));
  EXPECT_EQ(path[2].second, -1);
}
