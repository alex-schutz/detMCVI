/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _QLEARNING_H_
#define _QLEARNING_H_
#include <random>
#include <tuple>
#include <vector>

#include "PomdpInterface.h"

using namespace std;

/**
 * @brief Perform Q-learning on a POMDP
 */
class QLearning {
 public:
  QLearning(const PomdpInterface* sim, double epsilon, double learning_rate,
            double pb_random_explore, int nb_restarts_simulation,
            uint64_t seed = random_device{}())
      : sim(sim),
        epsilon(epsilon),
        learning_rate(learning_rate),
        pb_random_explore(pb_random_explore),
        nb_restarts_simulation(nb_restarts_simulation),
        q_table(InitQTable()),
        rng(seed) {}

  /// @brief Return the estimated Q-value for the given state index.
  double EstimateValue(int state);

  /// @brief Use the update equation for the q value associated with given state
  /// index and action index. Updates `q_table` internally.
  void UpdateQValue(int state, int action, double reward, int next_state);

  /// @brief Return the current q value associated with state index and
  /// action index.
  double GetQValue(int state, int action) const;

  /// @brief Return the maximum Q-value for the state index across all actions,
  /// and the best action index.
  tuple<double, int> MaxQ(int state) const;

  /// @brief Choose the index of the action to take in the state index. Chooses
  /// a random action with probability pb_random_explore, otherwise chooses the
  /// current best action.
  int ChooseAction(int state) const;

 private:
  const PomdpInterface* sim;
  double epsilon;
  double learning_rate;
  double pb_random_explore;
  int nb_restarts_simulation;
  vector<vector<double>> q_table;
  mutable std::mt19937_64 rng;

  vector<vector<double>> InitQTable() const;
};

#endif /* !_QLEARNING_H_ */
