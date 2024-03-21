/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _QLEARNING_H_
#define _QLEARNING_H_
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#include "BeliefParticles.h"
#include "SimInterface.h"

/**
 * @brief Perform Q-learning on a POMDP
 */
class QLearning {
 public:
  QLearning(SimInterface* sim, double learning_rate, double decay,
            int64_t sim_depth, double epsilon_init = 1.0,
            double epsilon_final = 0.1, uint64_t seed = random_device{}())
      : sim(sim),
        learning_rate(learning_rate),
        decay(decay),
        sim_depth(sim_depth),
        epsilon_init(epsilon_init),
        epsilon_final(epsilon_final),
        discount(sim->GetDiscount()),
        epsilon(epsilon_init),
        q_table(),
        rng(seed) {}

  /// @brief Train the Q-learning model on the given belief until improvement
  /// across the belief is less than epsilon or max_episodes is reached.
  void Train(const BeliefParticles& belief, int64_t max_episodes,
             int64_t episode_size, int64_t num_sims, double epsilon,
             ostream& os = cout);

  /// @brief Return the estimated Q-value for the given state index.
  double EstimateValue(int64_t state, int64_t n_sims);

  /// @brief Use the update equation for the q value associated with given state
  /// index and action index. Updates `q_table` internally.
  void UpdateQValue(int64_t state, int64_t action, double reward,
                    int64_t next_state);

  /// @brief Return the current q value associated with state index and
  /// action index.
  double GetQValue(int64_t state, int64_t action);

  /// @brief Return the maximum Q-value for the state index across all actions,
  /// and the best action index.
  tuple<double, int64_t> MaxQ(int64_t state);

  /// @brief Choose the index of the action to take in the state index. Chooses
  /// a random action with probability epsilon, otherwise chooses the
  /// current best action.
  int64_t ChooseAction(int64_t state);

 private:
  SimInterface* sim;
  int64_t n_sims;
  double learning_rate;
  double decay;
  int64_t sim_depth;
  double epsilon_init;
  double epsilon_final;
  double discount;
  double epsilon;
  unordered_map<int64_t, vector<double>> q_table;
  mutable std::mt19937_64 rng;

  unordered_map<int64_t, vector<double>>::iterator GetQTableRow(int64_t state);

  void DecayParameters();
};

#endif /* !_QLEARNING_H_ */
