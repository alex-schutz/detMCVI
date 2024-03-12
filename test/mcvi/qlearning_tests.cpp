#include <QLearning.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>

class TestPOMDP : public PomdpInterface {
 private:
  std::vector<string> states{"s0", "s1", "s2", "s3", "s4", "sg"};
  std::vector<string> actions{"a00", "a01", "a1", "a21",      "a20",
                              "a40", "a41", "a3", "self_loop"};
  std::vector<std::string> observations{"ob"};

 public:
  double GetDiscount() const override { return 0.99; }
  int GetSizeOfS() const override { return states.size(); }
  int GetSizeOfA() const override { return actions.size(); }
  int GetSizeOfObs() const override { return observations.size(); }
  double TransFunc(int sI, int aI, int s_newI) const override {
    switch (sI) {
      case 0:
        if (aI == 0)
          return (s_newI == 2) ? 1.0 : 0.0;
        else if (aI == 1)
          return (s_newI == 1) ? 1.0 : 0.0;
        break;
      case 1:
        if (aI == 2) return (s_newI == 2) ? 1.0 : 0.0;
        break;
      case 2:
        if (aI == 3)
          return (s_newI == 1) ? 1.0 : 0.0;
        else if (aI == 4)
          return (s_newI == 4) ? 1.0 : 0.0;
        break;
      case 3:
        if (aI == 7) return (s_newI == 4) ? 1.0 : 0.0;
        break;
      case 4:
        if (aI == 5)
          return (s_newI == 5) ? 1.0 : 0.0;
        else if (aI == 6) {
          if (s_newI == 3) return 0.4;
          if (s_newI == 5) return 0.6;
          return 0.0;
        }
        break;
      case 5:
        if (aI == 8) return (s_newI == 5) ? 1.0 : 0.0;
        break;
    }
    // invalid action
    if (sI == s_newI) return 1.0;
    return 0.0;
  }
  double ObsFunc(int oI, int s_newI, int aI) const override {
    (void)s_newI;
    (void)aI;
    if (oI == 0) return 1.0;
    return 0.0;
  }
  double Reward(int sI, int aI) const override {
    switch (sI) {
      case 0:
        if (aI == 0)
          return 1;
        else if (aI == 1)
          return -1;
        return -50;
      case 1:
        if (aI == 2) return -1;
        return -50;
      case 2:
        if (aI == 3)
          return -1;
        else if (aI == 4)
          return -1;
        return -50;
      case 3:
        if (aI == 7) return -1;
        return -50;
      case 4:
        if (aI == 5)
          return -5;
        else if (aI == 6)
          return -2;
        return -50;
    }
    return -50;
  }
  const std::vector<string> &GetAllStates() const override { return states; }
  const std::vector<string> &GetAllActions() const override { return actions; }
  const std::vector<string> &GetAllObservations() const override {
    return observations;
  }
  tuple<int, int, double, bool> Step(int sI, int aI) const override {
    std::mt19937_64 rng(random_device{}());
    uniform_real_distribution<double> unif(0, 1);

    // sample next state
    const double u_s = unif(rng);
    int s_next = -1;
    double p_s = 0.0;
    for (int s = 0; s < GetSizeOfS(); ++s) {
      p_s += TransFunc(sI, aI, s);
      if (p_s > u_s) {
        s_next = s;
        break;
      }
    }

    return {s_next, 0, Reward(sI, aI), (s_next == 5)};
  }
};

TEST(QLearningTest, Learning) {
  TestPOMDP sim;
  auto q_engine = QLearning(&sim, 0.00001, 0.1, 0.00005, 30000);

  EXPECT_NEAR(q_engine.EstimateValue(0), -6.0, 1e-5);
  EXPECT_NEAR(q_engine.EstimateValue(1), -6.0, 1e-5);
  EXPECT_NEAR(q_engine.EstimateValue(2), -5.0, 1e-5);
  EXPECT_NEAR(q_engine.EstimateValue(3), -5.0, 1e-5);
  EXPECT_NEAR(q_engine.EstimateValue(4), -4.0, 1e-5);
  EXPECT_NEAR(q_engine.EstimateValue(5), 0.0, 1e-5);
}
