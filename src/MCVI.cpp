
#include "MCVI.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace MCVI {

int64_t MCVIPlanner::InsertNode(
    const AlphaVectorNode& node,
    const std::unordered_map<int64_t, int64_t>& edges) {
  const int64_t nI = _fsc.AddNode(node);
  _fsc.UpdateEdge(nI, edges);
  return nI;
}

int64_t MCVIPlanner::FindOrInsertNode(
    const AlphaVectorNode& node,
    const std::unordered_map<int64_t, int64_t>& edges) {
  const int64_t action = node.GetBestAction();
  for (int64_t nI = 0; nI < _fsc.NumNodes(); ++nI) {
    // First check the best action
    if (_fsc.GetNode(nI).GetBestAction() != action) continue;
    const auto& check_edges = _fsc.GetEdges(nI);
    if (check_edges == edges) return nI;
  }
  return InsertNode(node, edges);
}

void MCVIPlanner::BackUp(std::shared_ptr<BeliefTreeNode> Tr_node,
                         double R_lower, int64_t max_depth_sim,
                         int64_t eval_depth) {
  // Initialise node with all action children if not already done
  auto boundFunc = [this, R_lower](const BeliefDistribution& belief,
                                   int64_t belief_depth, int64_t eval_depth,
                                   SimInterface* sim) {
    return this->_fsc.LowerBoundFromFSC(belief, belief_depth, R_lower,
                                        eval_depth, sim);
  };
  for (int64_t action = 0; action < _pomdp->GetSizeOfA(); ++action)
    Tr_node->GetOrAddChildren(action, _heuristic, eval_depth, boundFunc,
                              _pomdp);

  Tr_node->BackUpActions(_fsc, R_lower, max_depth_sim, _pomdp);
  Tr_node->UpdateBestAction();

  const int64_t best_act = Tr_node->GetBestActLBound();
  auto node_new = AlphaVectorNode(best_act);
  std::unordered_map<int64_t, int64_t> node_edges;
  for (const auto& [obs, next_belief] : Tr_node->GetChildren(best_act)) {
    if (next_belief.GetBestPolicyNode() >= 0)
      node_edges[obs] = next_belief.GetBestPolicyNode();
  }

  if (node_edges.empty()) return;  // Terminal belief

  const int64_t nI = FindOrInsertNode(node_new, node_edges);
  Tr_node->SetBestPolicyNode(nI);
}

static double s_time_diff(const std::chrono::steady_clock::time_point& begin,
                          const std::chrono::steady_clock::time_point& end) {
  return (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count()) /
         1e6;
}

void MCVIPlanner::SampleBeliefs(
    std::vector<std::shared_ptr<BeliefTreeNode>>& traversal_list,
    int64_t eval_depth, double target, double R_lower, int64_t max_depth_sim) {
  const auto node = traversal_list.back();
  if (node == nullptr) throw std::logic_error("Invalid node");
  // Initialise node with all action children if not already done
  auto boundFunc = [this, R_lower](const BeliefDistribution& belief,
                                   int64_t belief_depth, int64_t eval_depth,
                                   SimInterface* sim) {
    return this->_fsc.LowerBoundFromFSC(belief, belief_depth, R_lower,
                                        eval_depth, sim);
  };
  for (int64_t action = 0; action < _pomdp->GetSizeOfA(); ++action)
    node->GetOrAddChildren(action, _heuristic, eval_depth, boundFunc, _pomdp);
  node->BackUpActions(_fsc, R_lower, max_depth_sim, _pomdp);
  node->UpdateBestAction();

  try {
    const auto next_node = node->ChooseObservation(target);
    traversal_list.push_back(next_node);
  } catch (std::logic_error& e) {
    if (std::string(e.what()) == "Failed to find best observation") return;
    throw(e);
  }
}

static bool MCVITimeExpired(const std::chrono::steady_clock::time_point& begin,
                            int64_t max_computation_ms) {
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
  if (elapsed.count() >= max_computation_ms * 1000) {
    std::cout << "MCVI planning complete, reached maximum computation time."
              << std::endl;
    return true;
  }
  return false;
}

double MCVIPlanner::MCVIIteration(std::shared_ptr<BeliefTreeNode> Tr_root,
                                  double R_lower, int64_t ms_remaining,
                                  int64_t max_depth_sim, int64_t eval_depth,

                                  std::atomic<bool>& exit_flag) {
  std::cout << "Belief Expand Process" << std::flush;
  std::vector<std::shared_ptr<BeliefTreeNode>> traversal_list = {Tr_root};
  const double precision = Tr_root->GetUpper() - Tr_root->GetLower();

  auto begin = std::chrono::steady_clock::now();
  for (int64_t depth = 0; depth < max_depth_sim; ++depth) {
    SampleBeliefs(traversal_list, eval_depth - depth, precision, R_lower,
                  max_depth_sim - depth);
    if (exit_flag.load()) break;
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
    if (elapsed.count() >= ms_remaining * 1000 / 2) break;
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << " (" << s_time_diff(begin, end) << " seconds)" << std::endl;

  std::cout << "Backup Process" << std::flush;
  begin = std::chrono::steady_clock::now();
  while (!traversal_list.empty()) {
    auto tr_node = traversal_list.back();
    BackUp(tr_node, R_lower, max_depth_sim, eval_depth);
    traversal_list.pop_back();
    if (exit_flag.load()) break;
  }
  end = std::chrono::steady_clock::now();
  std::cout << " (" << s_time_diff(begin, end) << " seconds)" << std::endl;

  _fsc.SetStartNodeIndex(Tr_root->GetBestPolicyNode());

  std::cout << "Tr_root upper bound: " << Tr_root->GetUpper() << std::endl;
  std::cout << "Tr_root lower bound: " << Tr_root->GetLower() << std::endl;
  const double new_precision = Tr_root->GetUpper() - Tr_root->GetLower();
  std::cout << "Precision: " << new_precision << std::endl;
  return new_precision;
}

int64_t MCVIPlanner::GetFirstAction(std::shared_ptr<BeliefTreeNode> Tr_node,
                                    double R_lower, int64_t max_depth_sim,
                                    int64_t eval_depth, double eval_epsilon) {
  auto boundFunc = [eval_epsilon](const BeliefDistribution& belief,
                                  int64_t belief_depth, int64_t eval_depth,
                                  SimInterface* sim) {
    return FindRLower(sim, belief, belief_depth - eval_epsilon, eval_depth);
  };
  for (int64_t action = 0; action < _pomdp->GetSizeOfA(); ++action)
    Tr_node->GetOrAddChildren(action, _heuristic, eval_depth, boundFunc,
                              _pomdp);

  Tr_node->BackUpActions(_fsc, R_lower, max_depth_sim, _pomdp);
  Tr_node->UpdateBestAction();
  return Tr_node->GetBestActUBound();
}

// fsc, root node, precision, converged, timed out, reached max iter
std::tuple<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>, double, bool, bool,
           bool>
MCVIPlanner::PlanIncrement(std::shared_ptr<BeliefTreeNode> Tr_root,
                           double R_lower, int64_t iter, int64_t ms_remaining,
                           int64_t max_depth_sim, double epsilon,
                           int64_t max_nb_iter, int64_t eval_depth,
                           double eval_epsilon, std::atomic<bool>& exit_flag) {
  if (iter == 0) {
    const auto action = GetFirstAction(Tr_root, R_lower, max_depth_sim,
                                       eval_depth, eval_epsilon);
    const auto node = AlphaVectorNode(action);
    _fsc.AddNode(node);
    _fsc.SetStartNodeIndex(0);
  }
  const double precision_before = Tr_root->GetUpper() - Tr_root->GetLower();
  if (ms_remaining <= 0)
    return {_fsc, Tr_root, precision_before, false, true, false};
  if (iter >= max_nb_iter) {
    return {_fsc, Tr_root, precision_before, false, false, true};
  }
  const auto iter_start = std::chrono::steady_clock::now();
  std::cout << "--- Iter " << iter << " ---" << std::endl;
  const double precision = MCVIIteration(Tr_root, R_lower, ms_remaining,
                                         max_depth_sim, eval_depth, exit_flag);
  return {_fsc,
          Tr_root,
          precision,
          std::abs(precision) < epsilon,
          MCVITimeExpired(iter_start, ms_remaining),
          iter >= max_nb_iter - 1};
}

std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> MCVIPlanner::Plan(
    int64_t max_depth_sim, double epsilon, int64_t max_nb_iter,
    int64_t max_computation_ms, int64_t eval_depth, double eval_epsilon,
    std::atomic<bool>& exit_flag) {
  // Calculate the lower bound
  const double R_lower = FindRLower(_pomdp, _b0, eval_epsilon, eval_depth);

  const auto init_upper =
      CalculateUpperBound(_b0, 0, eval_depth, _heuristic, _pomdp);
  std::shared_ptr<BeliefTreeNode> Tr_root =
      CreateBeliefTreeNode(_b0, 0, init_upper, R_lower);

  const auto iter_start = std::chrono::steady_clock::now();
  int64_t i = 0;

  while (!exit_flag.load()) {
    std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    const auto ms_remaining =
        max_computation_ms -
        std::chrono::duration_cast<std::chrono::microseconds>(now - iter_start)
                .count() /
            1000;
    const auto [fsc, root, precision, converged, timed_out, max_iter] =
        PlanIncrement(Tr_root, R_lower, i, ms_remaining, max_depth_sim, epsilon,
                      max_nb_iter, eval_depth, eval_epsilon, exit_flag);
    _fsc = fsc;
    Tr_root = root;
    if (converged || timed_out || max_iter) break;
    ++i;
  }
  return {_fsc, Tr_root};
}

std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>>
MCVIPlanner::PlanAndEvaluate(int64_t max_depth_sim, double epsilon,
                             int64_t max_nb_iter, int64_t max_computation_ms,
                             int64_t eval_depth, double eval_epsilon,
                             int64_t max_eval_steps, int64_t n_eval_trials,
                             int64_t nb_particles_b0, int64_t eval_interval_ms,
                             int64_t completion_threshold,
                             int64_t completion_reps,
                             std::atomic<bool>& exit_flag) {
  // Calculate the lower bound
  const double R_lower = FindRLower(_pomdp, _b0, eval_epsilon, eval_depth);

  const auto init_upper =
      CalculateUpperBound(_b0, 0, eval_depth, _heuristic, _pomdp);
  std::shared_ptr<BeliefTreeNode> Tr_root =
      CreateBeliefTreeNode(_b0, 0, init_upper, R_lower);

  int64_t i = 0;
  int64_t time_sum = 0;
  int64_t last_eval = -eval_interval_ms * 1000;
  int64_t completed_times = 0;

  while (!exit_flag.load()) {
    const std::chrono::steady_clock::time_point iter_start =
        std::chrono::steady_clock::now();
    const auto ms_remaining = max_computation_ms - time_sum / 1e3;

    const auto [fsc, root, precision, converged, timed_out, max_iter] =
        PlanIncrement(Tr_root, R_lower, i, ms_remaining, max_depth_sim, epsilon,
                      max_nb_iter, eval_depth, eval_epsilon, exit_flag);

    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - iter_start);
    time_sum += elapsed.count();

    _fsc = fsc;
    Tr_root = root;
    ++i;

    if (time_sum - last_eval >= eval_interval_ms * 1000) {
      last_eval = time_sum;
      std::cout << "Evaluation of policy (" << max_eval_steps << " steps, "
                << n_eval_trials << " trials) at time " << time_sum / 1e6 << ":"
                << std::endl;
      const int64_t completed_count = EvaluationWithSimulationFSC(
          max_eval_steps, n_eval_trials, nb_particles_b0);
      std::cout << "detMCVI policy FSC contains " << _fsc.NumNodes()
                << " nodes." << std::endl;
      if (completed_count >= completion_threshold)
        completed_times++;
      else
        completed_times = 0;
      if (completed_times >= completion_reps) {
        std::cout
            << "MCVI planning complete, reached number of completion reps."
            << std::endl;
        break;
      }
    }
    if (converged || timed_out || max_iter) {
      std::cout << "MCVI planning complete. Converged: " << converged
                << " Timed out: " << timed_out
                << " Maxed iterations: " << max_iter << std::endl;
      break;
    }
    if (time_sum >= max_computation_ms * 1000) break;
  }

  std::cout << "Evaluation of policy (" << max_eval_steps << " steps, "
            << n_eval_trials << " trials) at time " << time_sum / 1e6 << ":"
            << std::endl;
  EvaluationWithSimulationFSC(max_eval_steps, n_eval_trials, nb_particles_b0);
  std::cout << "detMCVI policy FSC contains " << _fsc.NumNodes() << " nodes."
            << std::endl;
  return {_fsc, Tr_root};
}

static int64_t GreedyBestAction(const BeliefDistribution& belief,
                                SimInterface* pomdp) {
  int64_t best_a = -1;
  double best_r = -std::numeric_limits<double>::infinity();
  for (const auto& [state, prob] : belief) {
    for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
      const auto [sNext, obs, reward, done] = pomdp->Step(state, a);
      if (reward * prob > best_r) {
        best_r = reward;
        best_a = a;
      }
    }
  }
  return best_a;
}

static BeliefDistribution NextBelief(const BeliefDistribution& belief,
                                     int64_t action, int64_t observation,
                                     SimInterface* pomdp) {
  StateMap<double> next_states;
  double total_prob = 0.0;
  for (const auto& [state, prob] : belief) {
    const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
    if (obs != observation) continue;
    next_states[sNext] += prob;
    total_prob += prob;
  }
  for (auto& [s, prob] : next_states) prob /= total_prob;
  return BeliefDistribution(next_states);
}

void MCVIPlanner::SimulationWithFSC(int64_t steps) const {
  const double gamma = _pomdp->GetDiscount();
  State state = SampleOneState(_b0, _rng);
  BeliefDistribution belief = _b0;
  double sum_r = 0.0;
  int64_t nI = _fsc.GetStartNodeIndex();
  bool end_reached = false;
  for (int64_t i = 0; i < steps; ++i) {
    if (nI == -1 && !end_reached) {
      std::cout << "Reached end of policy." << std::endl;
      end_reached = true;
    }
    const int64_t action = (nI == -1) ? GreedyBestAction(belief, _pomdp)
                                      : _fsc.GetNode(nI).GetBestAction();
    std::cout << "---------" << std::endl;
    std::cout << "step: " << i << std::endl;
    std::cout << "state: <";
    for (const auto& state_elem : state) std::cout << state_elem << ", ";
    std::cout << ">" << std::endl;
    std::cout << "perform action: " << action << std::endl;
    const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);

    std::cout << "receive obs: " << obs << std::endl;
    if (nI != -1) {
      std::cout << "nI: " << nI << std::endl;
      std::cout << "nI value: " << _fsc.GetNode(nI).V_node() << std::endl;
    }
    std::cout << "reward: " << reward << std::endl;

    sum_r += std::pow(gamma, i) * reward;
    if (nI != -1) nI = _fsc.GetEdgeValue(nI, obs);

    if (done) {
      std::cout << "Reached terminal state." << std::endl;
      break;
    }
    state = sNext;
    belief = NextBelief(belief, action, obs, _pomdp);
  }
  std::cout << "sum reward: " << sum_r << std::endl;
}

BeliefDistribution SampleInitialBelief(int64_t N, SimInterface* pomdp) {
  StateMap<int64_t> state_counts;
  for (int64_t i = 0; i < N; ++i) state_counts[pomdp->SampleStartState()] += 1;
  auto init_belief = BeliefDistribution();
  for (const auto& [state, count] : state_counts)
    init_belief[state] = (double)count / N;
  return init_belief;
}

BeliefDistribution DownsampleBelief(const BeliefDistribution& belief,
                                    int64_t max_belief_samples,
                                    std::mt19937_64& rng) {
  const auto shuffled_init = weightedShuffle(belief, rng, max_belief_samples);
  double prob_sum = 0.0;
  for (const auto& [state, prob] : shuffled_init) prob_sum += prob;
  auto b = BeliefDistribution();
  for (const auto& [state, prob] : shuffled_init) b[state] = prob / prob_sum;
  return b;
}

// static bool StateHasSolution(const State& state, const PathToTerminal& ptt,
//                              int64_t max_depth) {
//   return ptt.is_terminal(state, max_depth);
// }

int64_t MCVIPlanner::EvaluationWithSimulationFSC(
    int64_t max_steps, int64_t num_sims, int64_t init_belief_samples) const {
  const double gamma = _pomdp->GetDiscount();
  EvaluationStats eval_stats;
  const BeliefDistribution init_belief =
      SampleInitialBelief(init_belief_samples, _pomdp);
  for (int64_t sim = 0; sim < num_sims; ++sim) {
    State state = SampleOneState(init_belief, _rng);
    const State initial_state = state;
    double sum_r = 0.0;
    int64_t nI = _fsc.GetStartNodeIndex();
    int64_t i = 0;
    for (; i < max_steps; ++i) {
      if (nI == -1) {
        // if (!StateHasSolution(initial_state, _heuristic, max_steps)) {
        //   eval_stats.no_solution_off_policy.update(sum_r);
        // } else {
        eval_stats.off_policy.update(sum_r);
        // }
        break;
      }

      const int64_t action = _fsc.GetNode(nI).GetBestAction();
      const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);
      sum_r += std::pow(gamma, i) * reward;

      if (done) {
        eval_stats.complete.update(sum_r);
        break;
      }

      nI = _fsc.GetEdgeValue(nI, obs);

      state = sNext;
    }
    if (i == max_steps) {
      //   if (!StateHasSolution(initial_state, _heuristic, max_steps)) {
      //     eval_stats.no_solution_on_policy.update(sum_r);
      //   } else {
      eval_stats.max_iterations.update(sum_r);
      //   }
    }
  }
  PrintStats(eval_stats.complete, "MCVI completed problem");
  PrintStats(eval_stats.off_policy, "MCVI exited policy");
  PrintStats(eval_stats.max_iterations, "MCVI max iterations");
  PrintStats(eval_stats.no_solution_on_policy, "MCVI no solution (on policy)");
  PrintStats(eval_stats.no_solution_off_policy,
             "MCVI no solution (exited policy)");
  return eval_stats.complete.getCount();
}

int64_t MCVIPlanner::EvaluationWithSimulationFSCFixedDist(
    int64_t max_steps, std::vector<State> init_dist) const {
  const double gamma = _pomdp->GetDiscount();
  EvaluationStats eval_stats;

  for (const auto& initial_state : init_dist) {
    State state = initial_state;
    double sum_r = 0.0;
    int64_t nI = _fsc.GetStartNodeIndex();
    int64_t i = 0;
    for (; i < max_steps; ++i) {
      if (nI == -1) {
        // if (!StateHasSolution(initial_state, _heuristic, max_steps)) {
        //   eval_stats.no_solution_off_policy.update(sum_r);
        // } else {
        eval_stats.off_policy.update(sum_r);
        // }
        break;
      }

      const int64_t action = _fsc.GetNode(nI).GetBestAction();
      const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);
      sum_r += std::pow(gamma, i) * reward;

      if (done) {
        eval_stats.complete.update(sum_r);
        break;
      }

      nI = _fsc.GetEdgeValue(nI, obs);

      state = sNext;
    }
    if (i == max_steps) {
      //   if (!StateHasSolution(initial_state, _heuristic, max_steps)) {
      //     eval_stats.no_solution_on_policy.update(sum_r);
      //   } else {
      eval_stats.max_iterations.update(sum_r);
      //   }
    }
  }
  PrintStats(eval_stats.complete, "MCVI completed problem");
  PrintStats(eval_stats.off_policy, "MCVI exited policy");
  PrintStats(eval_stats.max_iterations, "MCVI max iterations");
  PrintStats(eval_stats.no_solution_on_policy, "MCVI no solution (on policy)");
  PrintStats(eval_stats.no_solution_off_policy,
             "MCVI no solution (exited policy)");
  return eval_stats.complete.getCount();
}

std::vector<State> EvaluationWithGreedyTreePolicy(
    std::shared_ptr<BeliefTreeNode> root, int64_t max_steps, int64_t num_sims,
    int64_t init_belief_samples, SimInterface* pomdp, std::mt19937_64& rng,
    const PathToTerminal& /*ptt*/, const std::string& alg_name) {
  const double gamma = pomdp->GetDiscount();
  EvaluationStats eval_stats;
  std::vector<State> success_states;
  const BeliefDistribution init_belief =
      SampleInitialBelief(init_belief_samples, pomdp);
  State initial_state = {};
  for (int64_t sim = 0; sim < num_sims; ++sim) {
    State state = SampleOneState(init_belief, rng);
    initial_state = state;
    double sum_r = 0.0;
    auto node = root;
    int64_t i = 0;
    for (; i < max_steps; ++i) {
      if (node && node->GetBestActUBound() == -1) node = nullptr;
      if (!node) {
        // if (!StateHasSolution(initial_state, ptt, max_steps)) {
        //   eval_stats.no_solution_off_policy.update(sum_r);
        // } else {
        eval_stats.off_policy.update(sum_r);
        // }
        break;
      }
      const int64_t action = node->GetBestActUBound();
      const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
      sum_r += std::pow(gamma, i) * reward;

      if (done) {
        eval_stats.complete.update(sum_r);
        success_states.push_back(initial_state);
        break;
      }

      node = node->GetChild(action, obs);

      state = sNext;
    }
    if (i == max_steps) {
      //   if (!StateHasSolution(initial_state, ptt, max_steps)) {
      //     eval_stats.no_solution_on_policy.update(sum_r);
      //   } else {
      eval_stats.max_iterations.update(sum_r);
      //   }
    }
  }
  PrintStats(eval_stats.complete, alg_name + " completed problem");
  PrintStats(eval_stats.off_policy, alg_name + " exited policy");
  PrintStats(eval_stats.max_iterations, alg_name + " max iterations");
  PrintStats(eval_stats.no_solution_on_policy,
             alg_name + " no solution (on policy)");
  PrintStats(eval_stats.no_solution_off_policy,
             alg_name + " no solution (exited policy)");
  return success_states;
}

}  // namespace MCVI
