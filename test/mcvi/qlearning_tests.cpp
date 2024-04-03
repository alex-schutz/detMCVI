#include <QLearning.h>
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
  int GetNbAgent() const { return 1; }
  int GetSizeOfA() const override { return actions.size(); }
  int GetSizeOfObs() const override { return observations.size(); }
  int SampleStartState() { return 0; }
  std::tuple<int, int, double, bool> Step(int sI, int aI) override {
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

TEST(QLearningTest, Learning) {
  TestPOMDP sim;
  auto q_engine = QLearning(&sim, {0.7, 0.05, 50, 20, 100, 1000, 0.001});

  const auto belief = BeliefParticles({0, 1, 2, 3, 4, 5});
  q_engine.Train(belief, std::cerr);

  EXPECT_NEAR(get<0>(q_engine.MaxQ(0)), -5.851, 2e-2);  // -6
  EXPECT_NEAR(get<0>(q_engine.MaxQ(1)), -5.851, 2e-2);  // -6
  EXPECT_NEAR(get<0>(q_engine.MaxQ(2)), -4.898, 2e-2);  // -5
  EXPECT_NEAR(get<0>(q_engine.MaxQ(3)), -4.898, 2e-2);  // -5
  EXPECT_NEAR(get<0>(q_engine.MaxQ(4)), -3.945, 2e-2);  // -4
  EXPECT_NEAR(get<0>(q_engine.MaxQ(5)), 0.0, 1e-5);     // 0
}
