#include <QLearning.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>

class TestPOMDP : public SimInterface {
 private:
  std::vector<string> states{"s0", "s1", "s2", "s3", "s4", "sg"};
  std::vector<string> actions{"a00", "a01", "a1", "a21",      "a20",
                              "a40", "a41", "a3", "self_loop"};
  std::vector<std::string> observations{"ob"};

 public:
  double GetDiscount() const override { return 0.99; }
  int GetNbAgent() const { return 1; }
  int GetSizeOfA() const override { return actions.size(); }
  int GetSizeOfObs() const override { return observations.size(); }
  int SampleStartState() { return 0; }
  tuple<int, int, double, bool> Step(int sI, int aI) override {
    switch (sI) {
      case 0:
        if (aI == 0)
          return {2, 0, -1.0, false};
        else if (aI == 1)
          return {1, 0, -1.0, false};
        return {sI, 0, -50.0, false};
      case 1:
        if (aI == 2) return {2, 0, -1.0, false};
        return {sI, 0, -50.0, false};
      case 2:
        if (aI == 3)
          return {1, 0, -1.0, false};
        else if (aI == 4)
          return {4, 0, -1.0, false};
        return {sI, 0, -50.0, false};
      case 3:
        if (aI == 7) return {4, 0, -1.0, false};
        return {sI, 0, -50.0, false};
      case 4:
        if (aI == 5)
          return {5, 0, -5.0, true};
        else if (aI == 6) {
          std::mt19937_64 rng(random_device{}());
          uniform_real_distribution<double> unif(0, 1);
          const double u = unif(rng);
          const int s_next = u < 0.4 ? 3 : 5;
          return {s_next, 0, -2.0, s_next == 5};
        }
        return {sI, 0, -50.0, false};
      case 5:
        if (aI == 8) return {5, 0, 0.0, true};
        return {sI, 0, -50.0, true};
    }
    return {sI, 0, -50.0, false};
  }
};

TEST(QLearningTest, Learning) {
  TestPOMDP sim;
  auto q_engine = QLearning(&sim, 100000, 0.7, 0.00005, 50);

  EXPECT_NEAR(q_engine.EstimateValue(0), -6.0, 2e-1);
  EXPECT_NEAR(q_engine.EstimateValue(1), -6.0, 2e-1);
  EXPECT_NEAR(q_engine.EstimateValue(2), -5.0, 2e-1);
  EXPECT_NEAR(q_engine.EstimateValue(3), -5.0, 2e-1);
  EXPECT_NEAR(q_engine.EstimateValue(4), -4.0, 2e-1);
  EXPECT_NEAR(q_engine.EstimateValue(5), 0.0, 1e-5);
}
