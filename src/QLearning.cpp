#include "../include/QLearning.h"

#include <algorithm>
#include <limits>

unordered_map<int, vector<double>>::iterator QLearning::GetQTableRow(
    int state) {
  const auto row = q_table.find(state);
  if (row != q_table.end()) return row;
  // Initialise row
  return q_table.insert({state, vector<double>(sim->GetSizeOfA(), 0.0)}).first;
}

double QLearning::EstimateValue(int stateInit, int n_sims) {
  for (int i = 0; i < n_sims; i++) {
    int state = stateInit;
    int depth = 0;
    while (depth < sim_depth) {
      const int action = ChooseAction(state);
      const auto [stateNext, o, reward, done] = sim->Step(state, action);
      ++depth;

      // update Q-table
      UpdateQValue(state, action, reward, stateNext);

      // update state
      state = stateNext;

      if (done) break;
    }
  }
  return get<0>(MaxQ(stateInit));
}

void QLearning::DecayParameters() {
  const double f = 1.0 - decay;
  double eps_prev = (epsilon - epsilon_final) / (epsilon_init - epsilon_final);
  epsilon = f * eps_prev * (epsilon_init - epsilon_final) + epsilon_final;
  learning_rate *= f;
}

void QLearning::UpdateQValue(int state, int action, double reward,
                             int next_state) {
  const double old_val = GetQValue(state, action);
  const double new_val = reward + discount * get<0>(MaxQ(next_state));
  const double new_Q = (1 - learning_rate) * old_val + learning_rate * new_val;
  //   std::cerr << __func__ << " state: " << state << " action: " << action
  //             << " reward: " << reward << " next state: " << next_state
  //             << " new_Q: " << new_Q << " old value " << old_val << " new
  //             value "
  //             << new_val << std::endl;
  q_table[state][action] = new_Q;
}

double QLearning::GetQValue(int state, int action) {
  const auto row = GetQTableRow(state);
  return row->second.at(action);
}

tuple<double, int> QLearning::MaxQ(int state) {
  const auto row = GetQTableRow(state);
  const auto best = max_element(row->second.cbegin(), row->second.cend());
  return make_tuple(*best, best - row->second.cbegin());
}

int QLearning::ChooseAction(int state) {
  // check if we should explore randomly
  uniform_real_distribution<double> unif(0, 1);
  const double u = unif(rng);
  if (u < epsilon) {
    uniform_int_distribution<> action_dist(0, sim->GetSizeOfA() - 1);
    return action_dist(rng);
  }

  // choose the best action
  return get<1>(MaxQ(state));
}

void QLearning::Train(Belief belief, int max_episodes, int episode_size,
                      int num_sims, double epsilon, ostream& os) {
  double improvement = numeric_limits<double>::infinity();
  double avg_curr = -numeric_limits<double>::infinity();
  int i_episode = 0;
  while (improvement > epsilon && i_episode < max_episodes) {
    os << "------ Episode: " << i_episode << " ------" << endl;
    double ep_value = 0.0;
    for (int i = 0; i < episode_size; ++i) {
      double sum = 0.0;
      for (const auto& [state, prob] : belief) {
        if (prob == 0) continue;
        sum += EstimateValue(state, num_sims) * prob;
      }
      ep_value += sum;
      DecayParameters();
    }
    ep_value /= episode_size;
    improvement = abs(ep_value - avg_curr);
    avg_curr = ep_value;
    os << "Avg value: " << avg_curr << endl;
    ++i_episode;
  }
}
