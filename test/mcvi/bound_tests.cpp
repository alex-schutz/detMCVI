#include <Bound.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>

using namespace MCVI;

class TestPOMDP : public SimInterface {
 private:
  std::vector<std::string> states{"s0", "s1", "s2", "s3", "s4", "sg"};
  std::vector<std::string> actions{"a00", "a01", "a1", "a21",      "a20",
                                   "a40", "a41", "a3", "self_loop"};
  std::vector<std::string> observations{"ob"};

 public:
  double GetDiscount() const override { return 0.99; }
  int GetNbAgent() const override { return 1; }
  int GetSizeOfA() const override { return actions.size(); }
  int GetSizeOfObs() const override { return observations.size(); }
  int SampleStartState() override { return 0; }
  bool IsTerminal(int sI) const override { return sI == 5; }
  std::tuple<int, int, double, bool> Step(int sI, int aI) override {
    if (aI == 9) return {sI, 0, -13.0, false};
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
          std::mt19937_64 rng(std::random_device{}());
          std::uniform_real_distribution<double> unif(0, 1);
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

TEST(MCVITest, FindRLower) {
  TestPOMDP sim;

  const auto belief = BeliefDistribution({{0, 0.166666},
                                          {1, 0.166666},
                                          {2, 0.166666},
                                          {3, 0.166667},
                                          {4, 0.166667},
                                          {5, 0.166667}});

  const double R_lower_all = FindRLower(&sim, belief, 9, 0.0001, 100);
  EXPECT_NEAR(R_lower_all, -50 / 0.01, 1e-9);

  const double R_lower_9 = FindRLower(&sim, belief, 10, 0.0001, 100);
  EXPECT_NEAR(R_lower_9, -13 / 0.01, 1e-9);
}
