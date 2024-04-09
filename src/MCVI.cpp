
#include "MCVI.h"

#include <algorithm>
#include <chrono>
#include <limits>

namespace MCVI {

static bool CmpPair(const std::pair<int64_t, double>& p1,
                    const std::pair<int64_t, double>& p2) {
  return p1.second < p2.second;
}

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
    nI_current = _fsc.GetEdgeValue(nI_current, action, obs);
    V_n_s += std::pow(gamma, step) * reward;
    if (done) break;
    state = sNext;
  }

  return V_n_s;
}

std::pair<double, int64_t> MCVIPlanner::FindMaxValueNode(
    const AlphaVectorNode& node, int64_t a, int64_t o) const {
  const auto& v = node.GetActionObservationValues(a, o);
  if (v.empty()) return {0.0, -1};
  const auto it = std::max_element(std::begin(v), std::end(v), CmpPair);
  return {it->second, it->first};
}

int64_t MCVIPlanner::InsertNode(const AlphaVectorNode& node,
                                const AlphaVectorFSC::EdgeMap& edges) {
  const int64_t nI = _fsc.AddNode(node);
  _fsc.UpdateEdge(nI, edges);
  return nI;
}

int64_t MCVIPlanner::FindOrInsertNode(
    const AlphaVectorNode& node, const AlphaVectorFSC::EdgeMap& edges,
    const std::vector<int64_t>& observation_space) {
  const int64_t action = node.GetBestAction();
  for (int64_t nI = 0; nI < _fsc.NumNodes(); ++nI) {
    // First check the best action
    if (_fsc.GetNode(nI).GetBestAction() != action) continue;
    bool match = true;
    for (const auto& obs : observation_space) {
      const int64_t edge_node = _fsc.GetEdgeValue(nI, action, obs);
      if (edge_node == -1 || edge_node != edges.at({action, obs})) {
        match = false;
        break;
      }
    }
    if (match) return nI;
  }
  return InsertNode(node, edges);
}

void MCVIPlanner::BackUp(std::shared_ptr<BeliefTreeNode> Tr_node,
                         double R_lower, int64_t max_depth_sim,
                         int64_t nb_sample,
                         const std::vector<int64_t>& action_space,
                         const std::vector<int64_t>& observation_space) {
  const double gamma = _pomdp->GetDiscount();
  const BeliefParticles& belief = Tr_node->GetParticles();
  auto node_new = AlphaVectorNode(action_space, observation_space,
                                  RandomAction(action_space));

  AlphaVectorFSC::EdgeMap node_edges;
  for (const auto& action : action_space) {
    for (int64_t i = 0; i < nb_sample; ++i) {
      const int64_t state = belief.SampleOneState();
      const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);
      node_new.AddR(action, reward);
      for (int64_t nI = 0; nI < _fsc.NumNodes(); ++nI) {
        const double V_nI_sNext =
            SimulateTrajectory(nI, sNext, max_depth_sim, R_lower);
        node_new.AddValue(action, obs, nI, V_nI_sNext);
      }
    }

    for (const auto& obs : observation_space) {
      const auto [V_a_o, nI_a_o] = FindMaxValueNode(node_new, action, obs);
      if (nI_a_o == -1) continue;
      node_edges[{action, obs}] = nI_a_o;
      node_new.AddQ(action, gamma * V_a_o);
    }
    node_new.AddQ(action, node_new.GetR(action));
    node_new.NormaliseQ(action, nb_sample);
  }

  node_new.UpdateBestValue(Tr_node);
  const int64_t nI = FindOrInsertNode(node_new, node_edges, observation_space);
  Tr_node->SetFSCNodeIndex(nI);
}

int64_t MCVIPlanner::RandomAction(
    const std::vector<int64_t>& action_space) const {
  std::uniform_int_distribution<> action_dist(0, action_space.size() - 1);
  return action_space[action_dist(_rng)];
}

static double s_time_diff(const std::chrono::steady_clock::time_point& begin,
                          const std::chrono::steady_clock::time_point& end) {
  return (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count()) /
         1000.0;
}

AlphaVectorFSC MCVIPlanner::Plan(int64_t max_depth_sim, int64_t nb_sample,
                                 double epsilon, int64_t max_nb_iter) {
  std::vector<int64_t> action_space, observation_space;
  for (int64_t a = 0; a < _pomdp->GetSizeOfA(); ++a) action_space.push_back(a);
  for (int64_t o = 0; o < _pomdp->GetSizeOfObs(); ++o)
    observation_space.push_back(o);

  // Calculate the lower bound
  const double R_lower =
      FindRLower(_pomdp, _b0, action_space, _heuristic.GetPolicy().episode_size,
                 _heuristic.GetPolicy().ep_convergence_threshold,
                 _heuristic.GetPolicy().sim_depth);

  std::shared_ptr<BeliefTreeNode> Tr_root =
      CreateBeliefRootNode(_b0, action_space, _heuristic, _pomdp);
  const auto node = AlphaVectorNode(action_space, observation_space,
                                    RandomAction(action_space));
  _fsc.AddNode(node);
  Tr_root->SetFSCNodeIndex(_fsc.NumNodes() - 1);

  int64_t i = 0;
  while (i < max_nb_iter) {
    std::cout << "--- Iter " << i << " ---" << std::endl;
    UpdateUpperBound(Tr_root, _pomdp->GetDiscount(), 0);
    std::cout << "Tr_root upper bound: " << Tr_root->GetUpper() << std::endl;
    std::cout << "Tr_root lower bound: " << Tr_root->GetLower() << std::endl;
    const double precision = Tr_root->GetUpper() - Tr_root->GetLower();
    std::cout << "Precision: " << precision << std::endl;
    if (precision < epsilon) {
      std::cout << "MCVI planning complete, reached the target precision."
                << std::endl;
      return _fsc;
    }

    std::cout << "Belief Expand Process" << std::flush;
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<BeliefTreeNode>> traversal_list;
    SampleBeliefs(Tr_root, _b0.SampleOneState(), 0, max_depth_sim, nb_sample,
                  action_space, _pomdp, _heuristic, traversal_list);
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << " (" << s_time_diff(begin, end) << " seconds)" << std::endl;

    std::cout << "Backup Process" << std::flush;
    begin = std::chrono::steady_clock::now();
    while (!traversal_list.empty()) {
      auto tr_node = traversal_list.back();
      traversal_list.pop_back();
      BackUp(tr_node, R_lower, max_depth_sim, nb_sample, action_space,
             observation_space);
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
  int64_t state = _b0.SampleOneState();
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
    nI = _fsc.GetEdgeValue(nI, action, obs);

    if (done) break;
    state = sNext;
  }
  std::cout << "sum reward: " << sum_r << std::endl;
}

}  // namespace MCVI
