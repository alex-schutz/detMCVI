#include "../include/Bound.h"

double UpperBoundEvaluation(const Belief& belief, SimInterface* sim,
                            double learning_rate, double decay, int sim_depth,
                            int max_episodes, int episode_size, int num_sims,
                            double ep_convergence_threshold,
                            double random_action_pb_init,
                            double random_action_pb_final, uint64_t seed) {
  // We only use the underlying MDP dynamics of sim to compute the value
  auto q_engine =
      QLearning(sim, learning_rate, decay, sim_depth, random_action_pb_init,
                random_action_pb_final, seed);
  q_engine.Train(belief, max_episodes, episode_size, num_sims,
                 ep_convergence_threshold);

  // Calculate the upper bound
  double V_upper_b = 0.0;
  for (const auto& [state, prob] : belief)
    V_upper_b += prob * get<0>(q_engine.MaxQ(state));

  return V_upper_b;
}
