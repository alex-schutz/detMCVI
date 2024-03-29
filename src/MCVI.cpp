
#include "MCVI.h"

#include <algorithm>
#include <limits>

namespace MCVI {

static bool CmpPair(const std::pair<int64_t, double>& p1,
                    const std::pair<int64_t, double>& p2) {
  return p1.second < p2.second;
}

double SimulateTrajectory(int64_t nI, AlphaVectorFSC& fsc, int64_t state,
                          int64_t max_depth, SimInterface* pomdp) {
  const double gamma = pomdp->GetDiscount();
  double V_n_s = 0.0;
  int64_t nI_current = nI;
  for (int64_t step = 0; step < max_depth; ++step) {
    const int64_t action = (nI_current != -1)
                               ? fsc.GetNode(nI_current).GetBestAction()
                               : pomdp->RandomAction();
    const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
    if (nI_current != -1)
      nI_current = fsc.GetEdgeValue(nI_current, action, obs);

    V_n_s += std::pow(gamma, step) * reward;
    if (done) break;
    state = sNext;
  }

  return V_n_s;
}

std::pair<double, int64_t> FindMaxValueNode(const AlphaVectorNode& node,
                                            int64_t a, int64_t o) {
  const auto& v = node.GetActionObservationValues(a, o);
  const auto it = std::max_element(std::begin(v), std::end(v), CmpPair);
  return {it->second, it->first};
}

int64_t InsertNode(const AlphaVectorNode& node,
                   const AlphaVectorFSC::EdgeMap& edges, AlphaVectorFSC& fsc) {
  const int64_t nI = fsc.AddNode(node);
  fsc.UpdateEdge(nI, edges);
  return nI;
}

int64_t FindOrInsertNode(const AlphaVectorNode& node,
                         const AlphaVectorFSC::EdgeMap& edges,
                         const std::vector<int64_t>& observation_space,
                         AlphaVectorFSC& fsc) {
  const int64_t action = node.GetBestAction();
  for (int64_t nI = 0; nI < fsc.NumNodes(); ++nI) {
    // First check the best action
    if (fsc.GetNode(nI).GetBestAction() == action) {
      for (const auto& obs : observation_space) {
        const int64_t edge_node = fsc.GetEdgeValue(nI, action, obs);
        if (edge_node == -1 || edge_node != edges.at({action, obs}))
          return InsertNode(node, edges, fsc);
      }
      return nI;
    }
  }
  return InsertNode(node, edges, fsc);
}

void BackUp(std::shared_ptr<BeliefTreeNode> Tr_node, AlphaVectorFSC& fsc,
            int64_t max_depth_sim, int64_t nb_sample, SimInterface* pomdp,
            const std::vector<int64_t>& action_space,
            const std::vector<int64_t>& observation_space) {
  const double gamma = pomdp->GetDiscount();
  const BeliefParticles& belief = Tr_node->GetParticles();
  auto node_new = AlphaVectorNode(action_space, observation_space);

  AlphaVectorFSC::EdgeMap node_edges;
  for (const auto& action : action_space) {
    for (int64_t i = 0; i < nb_sample; ++i) {
      const int64_t state = belief.SampleOneState();
      const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
      node_new.AddR(action, reward);
      for (int64_t nI = 0; nI < fsc.NumNodes(); ++nI) {
        const double V_nI_sNext =
            SimulateTrajectory(nI, fsc, sNext, max_depth_sim, pomdp);
        node_new.UpdateValue(action, obs, nI, V_nI_sNext);
      }
    }

    for (const auto& obs : observation_space) {
      const auto [V_a_o, nI_a_o] = FindMaxValueNode(node_new, action, obs);
      node_edges[{action, obs}] = nI_a_o;
      node_new.AddQ(action, gamma * V_a_o);
    }
    node_new.AddQ(action, node_new.GetR(action));
    node_new.NormaliseQ(action, nb_sample);
  }

  node_new.UpdateBestValue(Tr_node);
  const int64_t nI =
      FindOrInsertNode(node_new, node_edges, observation_space, fsc);
  Tr_node->SetFSCNodeIndex(nI);
}

AlphaVectorFSC MCVIPlanning(const BeliefParticles& b0, AlphaVectorFSC fsc,
                            SimInterface* pomdp, int64_t max_depth_sim,
                            int64_t nb_sample, int64_t nb_iter,
                            const QLearningPolicy policy) {
  std::vector<int64_t> action_space, observation_space;
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) action_space.push_back(a);
  for (int64_t o = 0; o < pomdp->GetSizeOfObs(); ++o)
    observation_space.push_back(o);
  std::shared_ptr<BeliefTreeNode> Tr_root =
      CreateBeliefRootNode(b0, action_space, policy, pomdp);
  const auto node = AlphaVectorNode(action_space, observation_space);
  fsc.AddNode(node);
  Tr_root->SetFSCNodeIndex(fsc.NumNodes() - 1);

  for (int64_t i = 0; i < nb_iter; ++i) {
    std::cout << "--- Iter " << i << " ---" << std::endl;
    std::cout << "Tr_root upper bound: " << Tr_root->GetUpper() << std::endl;
    std::cout << "Tr_root lower bound: " << Tr_root->GetLower() << std::endl;
    std::cout << "Belief Expand Process" << std::endl;

    std::vector<std::shared_ptr<BeliefTreeNode>> traversal_list;
    SampleBeliefs(Tr_root, b0.SampleOneState(), 0, max_depth_sim, nb_sample,
                  action_space, pomdp, policy, traversal_list);

    std::cout << "Backup Process" << std::endl;
    while (!traversal_list.empty()) {
      auto tr_node = traversal_list.back();
      traversal_list.pop_back();
      BackUp(tr_node, fsc, max_depth_sim, nb_sample, pomdp, action_space,
             observation_space);
    }
  }

  fsc.SetStartNodeIndex(Tr_root->GetFSCNodeIndex());
  return fsc;
}

void SimulationWithFSC(const BeliefParticles& b0, SimInterface* pomdp,
                       AlphaVectorFSC& fsc, int64_t steps) {
  const double gamma = pomdp->GetDiscount();
  int64_t state = b0.SampleOneState();
  double sum_r = 0.0;
  int64_t nI = fsc.GetStartNodeIndex();
  for (int64_t i = 0; i < steps; ++i) {
    const int64_t action = fsc.GetNode(nI).GetBestAction();
    std::cout << "---------" << std::endl;
    std::cout << "step: " << i << std::endl;
    std::cout << "state: " << state << std::endl;
    std::cout << "perform action: " << action << std::endl;
    const auto [sNext, obs, reward, done] = pomdp->Step(state, action);

    std::cout << "receive obs: " << obs << std::endl;
    std::cout << "nI: " << nI << std::endl;
    std::cout << "nI value: " << fsc.GetNode(nI).V_node() << std::endl;
    std::cout << "reward: " << reward << std::endl;

    sum_r += std::pow(gamma, i) * reward;
    nI = fsc.GetEdgeValue(nI, action, obs);

    if (done) break;
    state = sNext;
  }
  std::cout << "sum reward: " << sum_r << std::endl;
}

}  // namespace MCVI
