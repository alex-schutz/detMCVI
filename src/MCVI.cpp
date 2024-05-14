
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
                         int64_t eval_depth, double eval_epsilon) {
  // Initialise node with all action children if not already done
  for (int64_t action = 0; action < _pomdp->GetSizeOfA(); ++action)
    Tr_node->GetOrAddChildren(action, _heuristic, eval_depth, eval_epsilon,
                              _pomdp);

  Tr_node->BackUpActions(_fsc, R_lower, max_depth_sim, _pomdp);
  Tr_node->UpdateBestAction();

  const int64_t best_act = Tr_node->GetBestActLBound();
  auto node_new = AlphaVectorNode(best_act);
  std::unordered_map<int64_t, int64_t> node_edges;
  for (const auto& [obs, next_belief] : Tr_node->GetChildren(best_act))
    node_edges[obs] = next_belief.GetBestPolicyNode();

  if (node_edges.empty()) return;  // Terminal belief

  const int64_t nI = FindOrInsertNode(node_new, node_edges);
  Tr_node->SetBestPolicyNode(nI);
}

int64_t MCVIPlanner::RandomAction() const {
  std::uniform_int_distribution<> action_dist(0, _pomdp->GetSizeOfA() - 1);
  return action_dist(_rng);
}

static double s_time_diff(const std::chrono::steady_clock::time_point& begin,
                          const std::chrono::steady_clock::time_point& end) {
  return (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count()) /
         1000.0;
}

void MCVIPlanner::SampleBeliefs(
    std::shared_ptr<BeliefTreeNode> node, int64_t state, int64_t depth,
    int64_t max_depth, SimInterface* pomdp, const PathToTerminal& heuristic,
    int64_t eval_depth, double eval_epsilon,
    std::vector<std::shared_ptr<BeliefTreeNode>>& traversal_list, double target,
    double R_lower, int64_t max_depth_sim) {
  if (depth >= max_depth) return;
  if (node == nullptr) throw std::logic_error("Invalid node");
  node->BackUpActions(_fsc, R_lower, max_depth_sim, _pomdp);
  node->UpdateBestAction();
  BackUp(node, R_lower, max_depth_sim, eval_depth, eval_epsilon);
  traversal_list.push_back(node);

  // TODO: identify and skip terminal states

  try {
    const auto next_node = node->ChooseObservation(target);
    SampleBeliefs(next_node, state, depth + 1, max_depth, pomdp, heuristic,
                  eval_depth, eval_epsilon, traversal_list, target, R_lower,
                  max_depth_sim);
  } catch (std::logic_error& e) {
    if (std::string(e.what()) == "Failed to find best observation") return;
    throw(e);
  }
}

std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> MCVIPlanner::Plan(
    int64_t max_depth_sim, double epsilon, int64_t max_nb_iter,
    int64_t max_computation_ms, int64_t eval_depth, double eval_epsilon) {
  // Calculate the lower bound
  const double R_lower =
      FindRLower(_pomdp, _b0, _pomdp->GetSizeOfA(), eval_epsilon, eval_depth);

  std::shared_ptr<BeliefTreeNode> Tr_root = CreateBeliefTreeNode(
      _b0, 0, _heuristic, eval_depth, eval_epsilon, _pomdp);
  const auto node = AlphaVectorNode(RandomAction());
  _fsc.AddNode(node);

  const auto iter_start = std::chrono::steady_clock::now();
  int64_t i = 0;
  while (i < max_nb_iter) {
    std::cout << "--- Iter " << i << " ---" << std::endl;
    std::cout << "Tr_root upper bound: " << Tr_root->GetUpper() << std::endl;
    std::cout << "Tr_root lower bound: " << Tr_root->GetLower() << std::endl;
    const double precision = Tr_root->GetUpper() - Tr_root->GetLower();
    std::cout << "Precision: " << precision << std::endl;
    if (std::abs(precision) < epsilon) {
      std::cout << "MCVI planning complete, reached the target precision."
                << std::endl;
      return {_fsc, Tr_root};
    }

    std::cout << "Belief Expand Process" << std::flush;
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<BeliefTreeNode>> traversal_list;
    SampleBeliefs(Tr_root, SampleOneState(_b0, _rng), 0, max_depth_sim, _pomdp,
                  _heuristic, eval_depth, eval_epsilon, traversal_list,
                  precision, R_lower, max_depth_sim);
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << " (" << s_time_diff(begin, end) << " seconds)" << std::endl;

    std::cout << "Backup Process" << std::flush;
    begin = std::chrono::steady_clock::now();
    while (!traversal_list.empty()) {
      auto tr_node = traversal_list.back();
      BackUp(tr_node, R_lower, max_depth_sim, eval_depth, eval_epsilon);
      traversal_list.pop_back();
    }
    end = std::chrono::steady_clock::now();
    std::cout << " (" << s_time_diff(begin, end) << " seconds)" << std::endl;

    _fsc.SetStartNodeIndex(Tr_root->GetBestPolicyNode());
    ++i;
    const auto iter_end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        iter_end - iter_start);
    if (elapsed.count() >= max_computation_ms) {
      std::cout << "MCVI planning complete, reached maximum computation time."
                << std::endl;
      return {_fsc, Tr_root};
    }
  }
  std::cout << "MCVI planning complete, reached the max iterations."
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
  std::unordered_map<int64_t, double> next_states;
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
  int64_t state = SampleOneState(_b0, _rng);
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
    std::cout << "state: " << state << std::endl;
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
  std::unordered_map<int64_t, int64_t> state_counts;
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

typedef struct {
  Welford complete;
  Welford off_policy;
  Welford max_iterations;
  Welford no_solution_on_policy;
  Welford no_solution_off_policy;
} EvaluationStats;

static void PrintStats(const Welford& stats, const std::string& alg_name) {
  std::cout << alg_name << " Count: " << stats.getCount() << std::endl;
  std::cout << alg_name << " Average reward: " << stats.getMean() << std::endl;
  std::cout << alg_name << " Highest reward: " << stats.getMax() << std::endl;
  std::cout << alg_name << " Lowest reward: " << stats.getMin() << std::endl;
  std::cout << alg_name << " Reward variance: " << stats.getVariance()
            << std::endl;
}

static bool StateHasSolution(int64_t state, const PathToTerminal& ptt,
                             int64_t max_depth) {
  return ptt.is_terminal(state, max_depth);
}

void MCVIPlanner::EvaluationWithSimulationFSC(
    int64_t max_steps, int64_t num_sims, int64_t init_belief_samples) const {
  const double gamma = _pomdp->GetDiscount();
  EvaluationStats eval_stats;
  int64_t initial_state = -1;
  const BeliefDistribution init_belief =
      SampleInitialBelief(init_belief_samples, _pomdp);
  for (int64_t sim = 0; sim < num_sims; ++sim) {
    BeliefDistribution belief = init_belief;
    int64_t state = SampleOneState(belief, _rng);
    initial_state = state;
    double sum_r = 0.0;
    int64_t nI = _fsc.GetStartNodeIndex();
    int64_t i = 0;
    for (; i < max_steps; ++i) {
      if (nI == -1) {
        if (!StateHasSolution(initial_state, _heuristic, max_steps)) {
          eval_stats.no_solution_off_policy.update(sum_r);
        } else {
          eval_stats.off_policy.update(sum_r);
        }
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
      belief = NextBelief(belief, action, obs, _pomdp);
    }
    if (i == max_steps) {
      if (!StateHasSolution(initial_state, _heuristic, max_steps)) {
        eval_stats.no_solution_on_policy.update(sum_r);
      } else {
        eval_stats.max_iterations.update(sum_r);
      }
    }
    initial_state = -1;
  }
  PrintStats(eval_stats.complete, "MCVI completed problem");
  PrintStats(eval_stats.off_policy, "MCVI exited policy");
  PrintStats(eval_stats.max_iterations, "MCVI max iterations");
  PrintStats(eval_stats.no_solution_on_policy, "MCVI no solution (on policy)");
  PrintStats(eval_stats.no_solution_off_policy,
             "MCVI no solution (exited policy)");
}

void EvaluationWithGreedyTreePolicy(std::shared_ptr<BeliefTreeNode> root,
                                    int64_t max_steps, int64_t num_sims,
                                    int64_t init_belief_samples,
                                    SimInterface* pomdp, std::mt19937_64& rng,
                                    const PathToTerminal& ptt,
                                    const std::string& alg_name) {
  const double gamma = pomdp->GetDiscount();
  EvaluationStats eval_stats;
  const BeliefDistribution init_belief =
      SampleInitialBelief(init_belief_samples, pomdp);
  int64_t initial_state = -1;
  for (int64_t sim = 0; sim < num_sims; ++sim) {
    BeliefDistribution belief = init_belief;
    int64_t state = SampleOneState(belief, rng);
    initial_state = state;
    double sum_r = 0.0;
    auto node = root;
    int64_t i = 0;
    for (; i < max_steps; ++i) {
      if (node && node->GetBestActUBound() == -1) node = nullptr;
      if (!node) {
        if (!StateHasSolution(initial_state, ptt, max_steps)) {
          eval_stats.no_solution_off_policy.update(sum_r);
        } else {
          eval_stats.off_policy.update(sum_r);
        }
        break;
      }
      const int64_t action = node->GetBestActUBound();
      const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
      sum_r += std::pow(gamma, i) * reward;

      if (done) {
        eval_stats.complete.update(sum_r);
        break;
      }

      node = node->GetChild(action, obs);

      belief = node->GetBelief();
      state = sNext;
    }
    if (i == max_steps) {
      if (!StateHasSolution(initial_state, ptt, max_steps)) {
        eval_stats.no_solution_on_policy.update(sum_r);
      } else {
        eval_stats.max_iterations.update(sum_r);
      }
    }
    initial_state = -1;
  }
  PrintStats(eval_stats.complete, alg_name + " completed problem");
  PrintStats(eval_stats.off_policy, alg_name + " exited policy");
  PrintStats(eval_stats.max_iterations, alg_name + " max iterations");
  PrintStats(eval_stats.no_solution_on_policy,
             alg_name + " no solution (on policy)");
  PrintStats(eval_stats.no_solution_off_policy,
             alg_name + " no solution (exited policy)");
}

}  // namespace MCVI
