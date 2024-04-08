#include "MCVI.h"
#include "SimInterface.h"

using namespace MCVI;

class TigerPOMDP : public MCVI::SimInterface {
 private:
  double noise;
  mutable std::mt19937_64 rng;

 public:
  TigerPOMDP(double noise) : noise(noise), rng(std::random_device{}()) {}

  std::tuple<int, int, double, bool> Step(int sI, int aI) override {
    if (aI < 0 || aI > 2)
      throw std::range_error("Action " + std::to_string(aI) + " out of range");
    if (sI < 0 || sI > 1)
      throw std::range_error("State " + std::to_string(sI) + " out of range");
    if (aI == 2) {  // listen action
      return {sI, NoisyObservation(sI), -1, false};
    }
    // open door action
    const int sI_next = SampleStartState();  // reset if a door is opened
    const int oI = CoinFlip();               // random observation
    const double reward = (aI == sI) ? -100 : 10;
    return {sI_next, oI, reward, false};
  }
  int SampleStartState() override { return CoinFlip(); }
  int GetSizeOfObs() const override { return 2; }
  int GetSizeOfA() const override { return 3; }
  double GetDiscount() const override { return 0.95; }
  int GetNbAgent() const override { return 1; }

  std::vector<std::string> actions = {"open-left", "open-right", "listen"};
  std::vector<std::string> observations = {"tiger-left", "tiger-right"};
  std::vector<std::string> states = {"tiger-left", "tiger-right"};

 private:
  int CoinFlip() const {
    std::uniform_int_distribution<> dist(0, 1);
    return dist(rng);
  }
  int NoisyObservation(int sI) const {
    std::uniform_real_distribution<double> unif(0, 1);
    const double u = unif(rng);
    return (u >= noise) ? sI : !sI;
  }
};

int main() {
  // Initialise the POMDP
  const double noise = 0.15;
  auto pomdp = TigerPOMDP(noise);

  // Sample the initial belief
  std::vector<int64_t> particles;
  for (int i = 0; i < 500; ++i) particles.push_back(pomdp.SampleStartState());
  const auto init_belief = BeliefParticles(particles);

  // Set the Q-learning policy
  const auto q_policy = QLearningPolicy(0.9, 0.01, 10, 10, 10, 20, 0.01);

  // Initialise the FSC
  const auto init_fsc = AlphaVectorFSC(10000, {0, 1, 2}, {0, 1});

  // Run MCVI
  auto planner = MCVIPlanner(&pomdp, init_fsc, init_belief, q_policy);
  planner.Plan(40, 1000, 0.1, 30, pomdp.actions, pomdp.observations);

  // Simulate the resultant FSC
  planner.SimulationWithFSC(20);

  return 0;
}
