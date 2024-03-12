#include "../include/QLearning.h"

#include <algorithm>

vector<vector<double>> QLearning::InitQTable() const {
  return vector<vector<double>>{sim->GetSizeOfS(),
                                vector<double>(sim->GetSizeOfA(), 0.0)};
}

double QLearning::EstimateValue(int stateInit) {
  for (int i = 0; i < nb_restarts_simulation; i++) {
    int state = stateInit;
    double total_discount = 1.0;
    while (total_discount > epsilon) {
      const int action = ChooseAction(state);
      const auto [stateNext, o, reward, done] = sim->Step(state, action);
      total_discount *= sim->GetDiscount();

      // update Q-table
      UpdateQValue(state, action, reward, stateNext);

      // update state
      state = stateNext;

      if (done) break;
    }
  }
  return get<0>(MaxQ(stateInit));
}

void QLearning::UpdateQValue(int state, int action, double reward,
                             int next_state) {
  const double new_Q =
      (1 - learning_rate) * GetQValue(state, action) +
      learning_rate * (reward + sim->GetDiscount() * get<0>(MaxQ(next_state)));
  q_table[state][action] = new_Q;
}

double QLearning::GetQValue(int state, int action) const {
  return q_table[state][action];
}

tuple<double, int> QLearning::MaxQ(int state) const {
  const auto best = max_element(q_table[state].cbegin(), q_table[state].cend());
  return make_tuple(*best, best - q_table[state].cbegin());
}

int QLearning::ChooseAction(int state) const {
  // check if we should explore randomly
  uniform_real_distribution<double> unif(0, 1);
  const double u = unif(rng);
  if (u < pb_random_explore) {
    uniform_int_distribution<> action_dist(0, sim->GetSizeOfA() - 1);
    return action_dist(rng);
  }

  // choose the best action
  return get<1>(MaxQ(state));
}
