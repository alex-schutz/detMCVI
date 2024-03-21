#include "../include/Bound.h"

double UpperBoundEvaluation(const BeliefParticles& belief, SimInterface* sim,
                            double learning_rate, double decay,
                            int64_t sim_depth, int64_t max_episodes,
                            int64_t episode_size, int64_t num_sims,
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
  for (const auto& state : belief.GetParticles())
    V_upper_b += get<0>(q_engine.MaxQ(state));

  return V_upper_b / belief.GetParticleCount();
}

double LowerBoundEvaluation(const BeliefParticles& belief, SimInterface* sim,
                            AlphaVectorFSC& fsc, int64_t num_sims,
                            int64_t max_depth, double epsilon) {
  const double gamma = sim->GetDiscount();

  int64_t nI = 0;
  bool random_pi = false;
  int64_t step = 0;
  double sum_r = 0.0;
  for (int64_t i = 0; i < num_sims; ++i) {
    double sum_r_sim_i = 0.0;
    int64_t state = belief.SampleOneState();

    while ((step < max_depth) && (std::pow(gamma, step) > epsilon)) {
      if (nI == -1) random_pi = true;
      const int64_t action =
          (random_pi) ? sim->RandomAction() : fsc.GetNode(nI).GetBestAction();

      const auto [sNext, obs, reward, done] = sim->Step(state, action);

      sum_r_sim_i += std::pow(gamma, step) * reward;
      nI = fsc.GetEtaValue(nI, action, obs);

      if (done) break;
      state = sNext;
      ++step;
    }
    sum_r += sum_r_sim_i;
  }
  return sum_r / num_sims;
}
