
#include "MCVI.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_set>

namespace MCVI {

double MCVIPlanner::SimulateTrajectory(int64_t nI, int64_t state,
                                       int64_t max_depth,
                                       double R_lower) const {
  const double gamma = _pomdp->GetDiscount();
  double V_n_s = 0.0;
  int64_t nI_current = nI;
  for (int64_t step = 0; step < max_depth; ++step) {
    if (nI_current == -1) {
      const double reward = std::pow(gamma, max_depth) * R_lower;
      V_n_s += std::pow(gamma, step) * reward;
      break;
    }

    const int64_t action = _fsc.GetNode(nI_current).GetBestAction();
    const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);
    nI_current = _fsc.GetEdgeValue(nI_current, obs);
    V_n_s += std::pow(gamma, step) * reward;
    if (done) break;
    state = sNext;
  }

  return V_n_s;
}

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
                         int64_t max_samples) {
  const double gamma = _pomdp->GetDiscount();
  const BeliefDistribution& belief = Tr_node->GetBelief();
  auto node_new = AlphaVectorNode(RandomAction());

  std::unordered_map<int64_t, int64_t> node_edges;
  double best_V = -std::numeric_limits<double>::infinity();
  int64_t best_a = -1;
  for (int64_t action = 0; action < _pomdp->GetSizeOfA(); ++action) {
    auto belief_cdf = CreateCDF(belief);
    for (int64_t sample = 0; sample < max_samples; ++sample) {
      const auto [state, prob] = SampleCDFDestructive(belief_cdf);
      if (state == -1) break;  // Sampled all states in belief
      const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);
      node_new.AddR(action, reward * prob);

      for (int64_t nI = 0; nI < _fsc.NumNodes(); ++nI) {
        const double V_nI_sNext =
            GetNodeAlpha(sNext, nI, R_lower, max_depth_sim);
        node_new.AddValue(action, obs, nI, V_nI_sNext * prob);
      }
    }

    const auto& [edges, sum_v] = node_new.BestNodePerObs(action);
    node_new.AddQ(action, gamma * sum_v + node_new.GetR(action));
    const double Q = node_new.GetQ(action);
    if (Q > best_V) {
      best_a = action;
      best_V = Q;
      node_edges = edges;
    }
  }

  node_new.UpdateBestValue(best_a, Tr_node);
  const int64_t nI = FindOrInsertNode(node_new, node_edges);
  Tr_node->SetFSCNodeIndex(nI);
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
    double R_lower, int64_t max_depth_sim, int64_t max_samples) {
  if (depth >= max_depth) return;
  if (node == nullptr) throw std::logic_error("Invalid node");
  node->SetUpper(
      UpperBoundUpdate(node->GetBelief(), R_lower, max_depth_sim, max_samples));
  BackUp(node, R_lower, max_depth_sim, max_samples);
  traversal_list.push_back(node);

  const int64_t action = node->GetBestAction();
  auto children = node->GetChildren(action);
  if (children.size() == 0) {
    const auto [o, next_beliefs] = BeliefUpdate(node, action, pomdp);
    for (const auto& [ob, b_next] : next_beliefs)
      CreateBeliefTreeNode(node, action, ob, b_next, pomdp->GetSizeOfA(),
                           heuristic, eval_depth, eval_epsilon, pomdp);
    children = node->GetChildren(action);
    if (children.size() == 0) throw std::logic_error("Still no children!");
  }

  const int64_t obs =
      ChooseObservation(children, node->GetWeights(action), target);

  SampleBeliefs(children.at(obs), state, depth + 1, max_depth, pomdp, heuristic,
                eval_depth, eval_epsilon, traversal_list, target, R_lower,
                max_depth_sim, max_samples);
}

AlphaVectorFSC MCVIPlanner::Plan(int64_t max_depth_sim, double epsilon,
                                 int64_t max_nb_iter, int64_t eval_depth,
                                 double eval_epsilon, int64_t max_samples) {
  // Calculate the lower bound
  const double R_lower =
      FindRLower(_pomdp, _b0, _pomdp->GetSizeOfA(), eval_epsilon, eval_depth);

  std::shared_ptr<BeliefTreeNode> Tr_root = CreateBeliefRootNode(
      _b0, _pomdp->GetSizeOfA(), _heuristic, eval_depth, eval_epsilon, _pomdp);
  const auto node = AlphaVectorNode(RandomAction());
  _fsc.AddNode(node);
  Tr_root->SetFSCNodeIndex(_fsc.NumNodes() - 1);

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
      return _fsc;
    }

    std::cout << "Belief Expand Process" << std::flush;
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<BeliefTreeNode>> traversal_list;
    SampleBeliefs(Tr_root, SampleOneState(_b0), 0, max_depth_sim, _pomdp,
                  _heuristic, eval_depth, eval_epsilon, traversal_list,
                  precision, R_lower, max_depth_sim, max_samples);
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << " (" << s_time_diff(begin, end) << " seconds)" << std::endl;

    std::cout << "Backup Process" << std::flush;
    begin = std::chrono::steady_clock::now();
    while (!traversal_list.empty()) {
      auto tr_node = traversal_list.back();
      traversal_list.pop_back();
      tr_node->SetUpper(UpperBoundUpdate(tr_node->GetBelief(), R_lower,
                                         max_depth_sim, max_samples));
      BackUp(tr_node, R_lower, max_depth_sim, max_samples);
    }
    end = std::chrono::steady_clock::now();
    std::cout << " (" << s_time_diff(begin, end) << " seconds)" << std::endl;

    _fsc.SetStartNodeIndex(Tr_root->GetFSCNodeIndex());
    ++i;
  }
  std::cout << "MCVI planning complete, reached the max iterations."
            << std::endl;
  return _fsc;
}

void MCVIPlanner::SimulationWithFSC(int64_t steps) const {
  const double gamma = _pomdp->GetDiscount();
  int64_t state = SampleOneState(_b0);
  double sum_r = 0.0;
  int64_t nI = _fsc.GetStartNodeIndex();
  for (int64_t i = 0; i < steps; ++i) {
    const int64_t action = _fsc.GetNode(nI).GetBestAction();
    std::cout << "---------" << std::endl;
    std::cout << "step: " << i << std::endl;
    std::cout << "state: " << state << std::endl;
    std::cout << "perform action: " << action << std::endl;
    const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);

    std::cout << "receive obs: " << obs << std::endl;
    std::cout << "nI: " << nI << std::endl;
    std::cout << "nI value: " << _fsc.GetNode(nI).V_node() << std::endl;
    std::cout << "reward: " << reward << std::endl;

    sum_r += std::pow(gamma, i) * reward;
    nI = _fsc.GetEdgeValue(nI, obs);

    if (done) break;
    state = sNext;
  }
  std::cout << "sum reward: " << sum_r << std::endl;
}

double MCVIPlanner::GetNodeAlpha(int64_t state, int64_t nI, double R_lower,
                                 int64_t max_depth_sim) {
  const std::optional<double> val = _fsc.GetNode(nI).GetAlpha(state);
  if (val.has_value()) return val.value();
  const double V = SimulateTrajectory(nI, state, max_depth_sim, R_lower);
  _fsc.GetNode(nI).SetAlpha(state, V);
  return V;
}

double MCVIPlanner::UpperBoundUpdate(const BeliefDistribution& belief,
                                     double R_lower, int64_t max_depth_sim,
                                     int64_t max_belief_samples) {
  double V_upper_bound = 0.0;
  auto belief_cdf = CreateCDF(belief);
  for (int64_t sample = 0; sample < max_belief_samples; ++sample) {
    const auto [state, prob] = SampleCDFDestructive(belief_cdf);
    if (state == -1) break;  // Sampled all states in belief
    double best_val = -std::numeric_limits<double>::infinity();
    for (int64_t action = 0; action < _pomdp->GetSizeOfA(); ++action) {
      const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);
      double best_alpha = -std::numeric_limits<double>::infinity();
      for (int64_t nI = 0; nI < _fsc.NumNodes(); ++nI) {
        const double V_nI_sNext =
            GetNodeAlpha(sNext, nI, R_lower, max_depth_sim);
        if (V_nI_sNext > best_alpha) best_alpha = V_nI_sNext;
      }
      double val = reward + _pomdp->GetDiscount() * best_alpha;
      if (val > best_val) best_val = val;
    }
    V_upper_bound += prob * best_val;
  }
  return V_upper_bound;
}

}  // namespace MCVI
