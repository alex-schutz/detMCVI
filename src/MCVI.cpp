
#include "MCVI.h"

#include <algorithm>
#include <chrono>
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
    int64_t eval_depth, double eval_epsilon) {
  // Calculate the lower bound
  const double R_lower =
      FindRLower(_pomdp, _b0, _pomdp->GetSizeOfA(), eval_epsilon, eval_depth);

  std::shared_ptr<BeliefTreeNode> Tr_root = CreateBeliefTreeNode(
      _b0, 0, _heuristic, eval_depth, eval_epsilon, _pomdp);
  const auto node = AlphaVectorNode(RandomAction());
  _fsc.AddNode(node);

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
  }
  std::cout << "MCVI planning complete, reached the max iterations."
            << std::endl;
  return {_fsc, Tr_root};
}

static int64_t GreedyBestAction(int64_t state, SimInterface* pomdp) {
  int64_t best_a = -1;
  double best_r = -std::numeric_limits<double>::infinity();
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    const auto [sNext, obs, reward, done] = pomdp->Step(state, a);
    if (reward > best_r) {
      best_r = reward;
      best_a = a;
    }
  }
  return best_a;
}

void MCVIPlanner::SimulationWithFSC(int64_t steps) const {
  const double gamma = _pomdp->GetDiscount();
  int64_t state = SampleOneState(_b0, _rng);
  double sum_r = 0.0;
  int64_t nI = _fsc.GetStartNodeIndex();
  bool end_reached = false;
  for (int64_t i = 0; i < steps; ++i) {
    if (nI == -1 && !end_reached) {
      std::cout << "Reached end of policy." << std::endl;
      end_reached = true;
    }
    const int64_t action = (nI == -1) ? GreedyBestAction(state, _pomdp)
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
  }
  std::cout << "sum reward: " << sum_r << std::endl;
}

void MCVIPlanner::EvaluationWithSimulationFSC(int64_t max_steps,
                                              int64_t num_sims) const {
  const double gamma = _pomdp->GetDiscount();
  double total_reward = 0;
  double max_reward = -std::numeric_limits<double>::infinity();
  double min_reward = std::numeric_limits<double>::infinity();
  for (int64_t sim = 0; sim < num_sims; ++sim) {
    int64_t state = SampleOneState(_b0, _rng);
    double sum_r = 0.0;
    int64_t nI = _fsc.GetStartNodeIndex();
    for (int64_t i = 0; i < max_steps; ++i) {
      const int64_t action = (nI != -1) ? _fsc.GetNode(nI).GetBestAction()
                                        : GreedyBestAction(state, _pomdp);
      const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);
      sum_r += std::pow(gamma, i) * reward;
      if (nI != -1) nI = _fsc.GetEdgeValue(nI, obs);

      if (done) break;
      state = sNext;
    }
    total_reward += sum_r;
    if (max_reward < sum_r) max_reward = sum_r;
    if (min_reward > sum_r) min_reward = sum_r;
  }
  std::cout << "Average reward: " << total_reward / num_sims << std::endl;
  std::cout << "Highest reward: " << max_reward << std::endl;
  std::cout << "Lowest reward: " << min_reward << std::endl;
}

void MCVIPlanner::EvaluationWithGreedyTreePolicy(
    std::shared_ptr<BeliefTreeNode> root, int64_t max_steps,
    int64_t num_sims) const {
  const double gamma = _pomdp->GetDiscount();
  double total_reward = 0;
  double max_reward = -std::numeric_limits<double>::infinity();
  double min_reward = std::numeric_limits<double>::infinity();
  for (int64_t sim = 0; sim < num_sims; ++sim) {
    int64_t state = SampleOneState(root->GetBelief(), _rng);
    double sum_r = 0.0;
    auto node = root;
    for (int64_t i = 0; i < max_steps; ++i) {
      if (node && node->GetBestActUBound() == -1) node = nullptr;
      const int64_t action =
          (node) ? node->GetBestActUBound() : GreedyBestAction(state, _pomdp);
      const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);
      sum_r += std::pow(gamma, i) * reward;
      if (node) node = node->GetChild(action, obs);

      if (done) break;
      state = sNext;
    }
    total_reward += sum_r;
    if (max_reward < sum_r) max_reward = sum_r;
    if (min_reward > sum_r) min_reward = sum_r;
  }
  std::cout << "Average reward: " << total_reward / num_sims << std::endl;
  std::cout << "Highest reward: " << max_reward << std::endl;
  std::cout << "Lowest reward: " << min_reward << std::endl;
}

}  // namespace MCVI
