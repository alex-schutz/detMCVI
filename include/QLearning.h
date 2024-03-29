/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#include "BeliefParticles.h"
#include "SimInterface.h"

namespace MCVI {

struct QLearningPolicy {
  double learning_rate;  // Initial learning rate
  double decay;  // Decay rate for learning rate and random action probability
  int64_t sim_depth;                // Max depth of a simulation run
  int64_t max_episodes;             // Max number of episodes to learn
  int64_t episode_size;             // Number of trials in a learning episode
  int64_t num_sims;                 // Number of simulation runs in a trial
  double ep_convergence_threshold;  // Threshold for when to stop learning
  double epsilon_init = 1.0;   // Initial probability of taking random actions
  double epsilon_final = 0.1;  // Final probability of taking random actions
};

/**
 * @brief Perform Q-learning on a POMDP
 */
class QLearning {
 public:
  QLearning(SimInterface* sim, QLearningPolicy policy,
            uint64_t seed = std::random_device{}())
      : sim(sim),
        policy(policy),
        discount(sim->GetDiscount()),
        epsilon(policy.epsilon_init),
        q_table(),
        rng(seed) {}

  /// @brief Train the Q-learning model on the given belief until improvement
  /// across the belief is less than epsilon or max_episodes is reached.
  void Train(const BeliefParticles& belief, std::ostream& os = std::cout);

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
  std::tuple<double, int64_t> MaxQ(int64_t state);

  /// @brief Choose the index of the action to take in the state index. Chooses
  /// a random action with probability epsilon, otherwise chooses the
  /// current best action.
  int64_t ChooseAction(int64_t state);

 private:
  SimInterface* sim;
  int64_t n_sims;
  QLearningPolicy policy;
  double discount;
  double epsilon;
  std::unordered_map<int64_t, std::vector<double>> q_table;
  mutable std::mt19937_64 rng;

  std::unordered_map<int64_t, std::vector<double>>::iterator GetQTableRow(
      int64_t state);

  void DecayParameters();
};

}  // namespace MCVI
