/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _BOUND_H_
#define _BOUND_H_

#include <unordered_map>

#include "QLearning.h"
#include "SimInterface.h"

using namespace std;

/**
 * @brief Return an upper bound for the value of the belief.
 *
 * The upper bound value of the belief is the expected sum of each possible
 * state's MDP value
 *
 * @param belief A map from state indices to probabilities.
 * @param sim A POMDP simulator object
 * @param learning_rate Initial learning rate to use in Q-learning
 * @param decay Decay rate for learning rate and random action probability
 * @param sim_depth Max depth of a simulation run
 * @param max_episodes Max number of episodes to learn
 * @param episode_size Number of trials in a learning episode
 * @param num_sims Number of simulation runs in a trial
 * @param ep_convergence_threshold Threshold for when to stop learning
 * @param random_action_pb_init Initial probability of taking random actions
 * @param random_action_pb_final Final probability of taking random actions
 * @param seed Random seed
 * @return double
 */
double UpperBoundEvaluation(const Belief& belief, SimInterface* sim,
                            double learning_rate, double decay, int sim_depth,
                            int max_episodes, int episode_size, int num_sims,
                            double ep_convergence_threshold,
                            double random_action_pb_init = 1.0,
                            double random_action_pb_final = 0.1,
                            uint64_t seed = random_device{}());

#endif /* !_BOUND_H_ */
