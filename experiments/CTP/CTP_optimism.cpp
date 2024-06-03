#include <iostream>
#include <random>

#include "Bound.h"
#include "CTP.h"
#include "auto_generated_graph.h"

using namespace MCVI;

class CTP_Optimism : public CTP {
 public:
  CTP_Optimism(CTP& ctp) : CTP(ctp), ptt(&ctp) {}

  void SimulateRun(int64_t max_depth) {
    const double gamma = GetDiscount();
    State state_true = SampleStartState();
    PrintState(state_true);
    State state_optimistic = InitialiseState();
    const int64_t loc_idx = sfIdx("loc");

    double sum_r = 0.0;
    int64_t i = 0;
    while (i++ < max_depth) {
      std::cout << "---------" << std::endl;
      std::cout << "step: " << i << std::endl;
      std::cout << "true state: <";
      for (const auto& s : state_true) std::cout << s << ", ";
      std::cout << ">" << std::endl;
      std::cout << "optimistic state: <";
      for (const auto& s : state_optimistic) std::cout << s << ", ";
      std::cout << ">" << std::endl;
      std::cout << "agent loc: " << state_true.at(loc_idx) << std::endl;
      const int64_t action = GetBestAction(state_optimistic, max_depth);
      std::cout << "perform action: " << action << std::endl;

      const auto [sNext, obs, reward, done] = Step(state_true, action);
      std::cout << "receive obs: " << obs << std::endl;
      std::cout << "reward: " << reward << std::endl;
      sum_r += std::pow(gamma, i) * reward;

      if (done) {
        std::cout << "Reached terminal state." << std::endl;
        break;
      }

      state_true = sNext;

      // next state if no blocked edges observed
      State optimistic_next_state = state_optimistic;
      if (state_optimistic.at(loc_idx) == (int64_t)nodes.size())
        state_optimistic[loc_idx] = origin;
      else
        state_optimistic[loc_idx] = action;
      const int64_t intended_action =
          GetBestAction(optimistic_next_state, max_depth);
      // actual next state
      state_optimistic = ApplyObservation(state_optimistic, obs);

      const int64_t next_action = GetBestAction(state_optimistic, max_depth);
      if (next_action != intended_action)
        std::cout << "Replanning trajectory: wanted " << intended_action
                  << ", will choose " << next_action << std::endl;
    }
    std::cout << "sum reward: " << sum_r << std::endl;
  }

  void EvaluatePolicy(int64_t num_sims, int64_t max_depth) {
    EvaluationStats eval_stats;
    const double gamma = GetDiscount();

    for (int64_t sim = 0; sim < num_sims; ++sim) {
      State state_true = SampleStartState();
      const State init_state = state_true;
      State state_optimistic = InitialiseState();

      double sum_r = 0.0;
      int64_t i = 0;
      while (i++ < max_depth) {
        const int64_t action = GetBestAction(state_optimistic, max_depth);
        const auto [sNext, obs, reward, done] = Step(state_true, action);
        sum_r += std::pow(gamma, i) * reward;
        if (done) {
          eval_stats.complete.update(sum_r);
          break;
        }

        state_optimistic = ApplyObservation(state_optimistic, obs);
        state_true = sNext;
      }
      if (i == max_depth) {
        if (!ptt.is_terminal(init_state, max_depth)) {
          eval_stats.no_solution_on_policy.update(sum_r);
        } else {
          eval_stats.max_iterations.update(sum_r);
        }
      }
    }

    PrintStats(eval_stats.complete, "Optimism completed problem");
    PrintStats(eval_stats.off_policy, "Optimism exited policy");
    PrintStats(eval_stats.max_iterations, "Optimism max iterations");
    PrintStats(eval_stats.no_solution_on_policy,
               "Optimism no solution (on policy)");
    PrintStats(eval_stats.no_solution_off_policy,
               "Optimism no solution (exited policy)");
  }

 private:
  PathToTerminal ptt;

  // Update the state to set location and blocked edges according to the given
  // observation
  State ApplyObservation(const State& state, int64_t observation) const {
    State out_state = state;

    int64_t loc = observation / max_obs_width;  // int div
    out_state[sfIdx("loc")] = loc;

    const int64_t edge_bool = observation % max_obs_width;
    int64_t n = 0;
    for (const auto& edge : AdjacentStochEdges(loc)) {
      out_state[sfIdx(edge2str(edge))] = bool(edge_bool & ((int64_t)1 << n));
      ++n;
    }

    return out_state;
  }

  // Return the state at the initial position with all edges open
  State InitialiseState() const {
    std::map<std::string, int64_t> state;
    state["loc"] = (int64_t)nodes.size();
    for (const auto& [edge, p] : stoch_edges) state[edge2str(edge)] = 1;
    return names2state(state);
  }

  // Choose best action based on shortest path to goal
  // If no path to goal, take exit action
  int64_t GetBestAction(const State& state, int64_t max_depth) const {
    if (!ptt.is_terminal(state, max_depth)) {
      for (int64_t i = (int64_t)actions.size() - 1; i >= 0; --i) {
        if (actions.at(i) == "decide_goal_unreachable") return i;
      }
      throw std::logic_error("No action to decide goal unreachable");
    }
    const auto [action, reward] = ptt.path(state, max_depth);
    return action;
  }

  void PrintState(const State& state) const {
    std::cout << "State:" << std::endl;
    std::cout << "loc: " << state.at(sfIdx("loc")) << std::endl;
    for (const auto& [edge, p] : stoch_edges)
      std::cout << edge2str(edge) << ": " << state.at(sfIdx(edge2str(edge)))
                << std::endl;
  }
};

int main() {
  std::mt19937_64 rng(std::random_device{}());

  auto ctp = CTP(rng);
  auto optimism = CTP_Optimism(ctp);

  std::fstream ctp_graph("ctp_graph.dot", std::fstream::out);
  ctp.visualiseGraph(ctp_graph);
  ctp_graph.close();

  const int64_t max_eval_steps = 30;
  const int64_t n_eval_trials = 10000;

  optimism.SimulateRun(max_eval_steps);

  optimism.EvaluatePolicy(n_eval_trials, max_eval_steps);

  return 0;
}
