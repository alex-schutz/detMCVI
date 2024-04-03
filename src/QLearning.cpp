#include "QLearning.h"

#include <algorithm>
#include <limits>

namespace MCVI {

std::unordered_map<int64_t, std::vector<double>>::iterator
QLearning::GetQTableRow(int64_t state) {
  const auto row = q_table.find(state);
  if (row != q_table.end()) return row;
  // Initialise row
  return q_table.insert({state, std::vector<double>(sim->GetSizeOfA(), 0.0)})
      .first;
}

double QLearning::EstimateValue(int64_t stateInit, int64_t n_sims) {
  for (int64_t i = 0; i < n_sims; i++) {
    int64_t state = stateInit;
    int64_t depth = 0;
    while (depth < policy.sim_depth) {
      const int64_t action = ChooseAction(state);
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
  const double f = 1.0 - policy.decay;
  double eps_prev = (epsilon - policy.epsilon_final) /
                    (policy.epsilon_init - policy.epsilon_final);
  epsilon = f * eps_prev * (policy.epsilon_init - policy.epsilon_final) +
            policy.epsilon_final;
  policy.learning_rate *= f;
}

void QLearning::UpdateQValue(int64_t state, int64_t action, double reward,
                             int64_t next_state) {
  const double old_val = GetQValue(state, action);
  const double new_val = reward + discount * get<0>(MaxQ(next_state));
  const double new_Q =
      (1 - policy.learning_rate) * old_val + policy.learning_rate * new_val;
  q_table[state][action] = new_Q;
}

double QLearning::GetQValue(int64_t state, int64_t action) {
  const auto row = GetQTableRow(state);
  return row->second.at(action);
}

std::tuple<double, int64_t> QLearning::MaxQ(int64_t state) {
  const auto row = GetQTableRow(state);
  const auto best = std::max_element(row->second.cbegin(), row->second.cend());
  return std::make_tuple(*best, best - row->second.cbegin());
}

int64_t QLearning::ChooseAction(int64_t state) {
  // check if we should explore randomly
  std::uniform_real_distribution<double> unif(0, 1);
  const double u = unif(rng);
  if (u < epsilon) {
    std::uniform_int_distribution<> action_dist(0, sim->GetSizeOfA() - 1);
    return action_dist(rng);
  }

  // choose the best action
  return get<1>(MaxQ(state));
}

void QLearning::Train(const BeliefParticles& belief, std::ostream& os) {
  double improvement = std::numeric_limits<double>::infinity();
  double avg_curr = -std::numeric_limits<double>::infinity();
  int64_t i_episode = 0;
  while (improvement > policy.ep_convergence_threshold &&
         i_episode < policy.max_episodes) {
    os << "------ Episode: " << i_episode << " ------" << std::endl;
    double ep_value = 0.0;
    for (int64_t i = 0; i < policy.episode_size; ++i) {
      double sum = 0.0;
      for (const auto& state : belief.GetParticles())
        sum += EstimateValue(state, policy.num_sims);
      ep_value += sum / belief.GetParticleCount();
      DecayParameters();
    }
    ep_value /= policy.episode_size;
    improvement = std::abs(ep_value - avg_curr);
    avg_curr = ep_value;
    os << "Avg value: " << avg_curr << std::endl;
    ++i_episode;
  }
}

}  // namespace MCVI
