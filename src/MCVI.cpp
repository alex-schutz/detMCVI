
#include "../include/MCVI.h"

#include <algorithm>
#include <limits>

#include "../include/AlphaVectorFSC.h"
#include "../include/BeliefParticles.h"

double FindRLower(SimInterface* pomdp, const BeliefParticles& b0,
                  const std::vector<int>& action_space, int64_t max_restarts,
                  double epsilon, int64_t max_depth) {
  std::unordered_map<int, double> action_min_reward;
  for (const auto& action : action_space) {
    double min_reward = std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < max_restarts; ++i) {
      int64_t state = b0.SampleOneState();
      int64_t step = 0;
      while ((step < max_depth) &&
             (std::pow(pomdp->GetDiscount(), step) > epsilon)) {
        const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
        if (reward < min_reward) {
          action_min_reward[action] = reward;
          min_reward = reward;
        }
        if (done) break;
        state = sNext;
        ++step;
      }
    }
  }
  const double max_min_reward =
      std::max_element(std::begin(action_min_reward),
                       std::end(action_min_reward),
                       [](const std::pair<int, double>& p1,
                          const std::pair<int, double>& p2) {
                         return p1.second < p2.second;
                       })
          ->second;
  return max_min_reward / (1 - pomdp->GetDiscount());
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
    if (nI_current != -1) nI_current = fsc.GetEtaValue(nI_current, action, obs);

    V_n_s += std::pow(gamma, step) * reward;
    if (done) break;
    state = sNext;
  }

  return V_n_s;
}

std::pair<double, int64_t> FindMaxValueNode(int64_t n, AlphaVectorFSC& fsc,
                                            int64_t a, int64_t o) {
  const auto& v = fsc.GetNode(n).GetActionObservationValues(a, o);
  const auto it = std::max_element(std::begin(v), std::end(v),
                                   [](const std::pair<int64_t, double>& p1,
                                      const std::pair<int64_t, double>& p2) {
                                     return p1.second < p2.second;
                                   });
  return {it->second, it->first};
}

void BackUp(int64_t nI_new, AlphaVectorFSC& fsc, int64_t max_depth_sim,
            int64_t nb_sample, SimInterface* pomdp,
            const std::vector<int64_t>& action_space,
            const std::vector<int64_t>& observation_space) {
  // a new node (alpha-vector in MCVI)
  const double gamma = pomdp->GetDiscount();
  auto& node = fsc.GetNode(nI_new);
  double V_nI_new = node.V_node();
  std::cout << "nI_new: " << nI_new << ", V: " << V_nI_new << std::endl;

  node.ReInit(action_space, observation_space);

  for (const auto& action : action_space) {
    for (int64_t i = 0; i < nb_sample; ++i) {
      const int64_t state = node.SampleParticle();
      const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
      node.AddR(action, reward);
      for (int64_t nI = 0; nI < fsc.NumNodes(); ++nI) {
        const double V_nI_sNext =
            SimulateTrajectory(nI, fsc, sNext, max_depth_sim, pomdp);
        node.UpdateValue(action, obs, nI, V_nI_sNext);
      }
    }

    for (const auto& obs : observation_space) {
      const auto [V_a_o, nI_a_o] = FindMaxValueNode(nI_new, fsc, action, obs);
      fsc.UpdateEta(nI_new, action, obs, nI_a_o);
      node.AddQ(action, gamma * V_a_o);
    }
    node.AddQ(action, node.GetR(action));
    node.NormaliseQ(action, nb_sample);
  }

  node.UpdateBestValue();
}

void MCVIPlanning(int64_t nb_particles, const BeliefParticles& b0,
                  AlphaVectorFSC fsc, SimInterface* pomdp, double epsilon) {
  fsc.AddNode(b0);

  while (true) {
    // TODO: C++ implementation of the solver
  }
}
