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
    if (aI == 0) {  // listen action
      return {sI, NoisyObservation(sI), -1, false};
    }
    // open door action
    const int sI_next = SampleStartState();  // reset if a door is opened
    const int oI = CoinFlip();               // random observation
    const double reward = (aI == sI + 1) ? -100 : 10;
    return {sI_next, oI, reward, false};
  }
  int SampleStartState() override { return CoinFlip(); }
  int GetSizeOfObs() const override { return 2; }
  int GetSizeOfA() const override { return 3; }
  double GetDiscount() const override { return 0.95; }
  int GetNbAgent() const override { return 1; }

  std::vector<std::string> actions = {"listen", "open-left", "open-right"};
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

  const int64_t nb_particles_b0 = 10000;
  const int64_t max_node_size = 10000;

  // Sample the initial belief
  std::vector<int64_t> particles;
  for (int i = 0; i < nb_particles_b0; ++i)
    particles.push_back(pomdp.SampleStartState());
  const auto init_belief = BeliefParticles(particles);

  // Set the Q-learning policy
  const int64_t max_sim_depth = 40;
  const double learning_rate = 0.9;
  const int64_t nb_episode_size = 10;
  const int64_t nb_max_episode = 10;
  const int64_t nb_sim = 20;
  const double decay_Q_learning = 0.01;
  const double epsilon_Q_learning = 0.01;
  const auto q_policy = QLearningPolicy(
      learning_rate, decay_Q_learning, max_sim_depth, nb_max_episode,
      nb_episode_size, nb_sim, epsilon_Q_learning);

  // Initialise the FSC
  const auto init_fsc = AlphaVectorFSC(max_node_size, {0, 1, 2}, {0, 1});

  // Run MCVI
  auto planner = MCVIPlanner(&pomdp, init_fsc, init_belief, q_policy);
  const int64_t nb_sample = 1000;
  const double converge_thresh = 0.1;
  const int64_t max_iter = 30;
  const auto fsc =
      planner.Plan(max_sim_depth, nb_sample, converge_thresh, max_iter);

  fsc.GenerateGraphviz(std::cerr, pomdp.actions, pomdp.observations);

  // Simulate the resultant FSC
  //   planner.SimulationWithFSC(20);

  return 0;
}
