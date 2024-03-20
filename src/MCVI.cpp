
#include "../include/MCVI.h"

#include <algorithm>
#include <limits>

#include "../include/AlphaVectorFSC.h"

// Naive way of sampling a discrete distribution
int SampleBelief(Belief belief) {
  std::mt19937_64 rng;
  uniform_real_distribution<double> unif(0, 1);
  const double u = unif(rng);
  double p_sum = 0.0;
  for (const auto& [state, prob] : belief) {
    p_sum += prob;
    if (u < p_sum) return state;
  }
  throw std::runtime_error("Failed to sample belief");
}

double FindRLower(SimInterface* pomdp, Belief b0, std::vector<int> action_space,
                  int max_restarts, double epsilon, int max_depth) {
  std::unordered_map<int, double> action_min_reward;
  for (const auto& action : action_space) {
    double min_reward = std::numeric_limits<double>::infinity();
    for (int i = 0; i < max_restarts; ++i) {
      int state = SampleBelief(b0);
      int step = 0;
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

// gen function should be implemented!!!

double SimulateTrajectory(int nI, AlphaVectorFSC fsc, double s, int L,
                          POMDP pomdp) {
  double gamma = std::pow(pomdp.discount(), 1.0);
  double V_n_s = 0.0;
  int nI_current = nI;
  for (int step = 0; step < L; ++step) {
    double a = GetBestAction(fsc.nodes[nI_current]);
    std::vector<double> sp, o, r;
    // TODO: C++ implementation of the`gen` function

    if (fsc.eta[nI_current].find(std::make_pair(a, o)) !=
        fsc.eta[nI_current].end()) {
      nI_current = fsc.eta[nI_current][std::make_pair(a, o)];
    }
    V_n_s += (gamma ^ step) * r;
    s = sp;
  }

  return V_n_s;
}

int FindMaxValueNode(int n, AlphaVectorFSC fsc, int a, int o) {
  int max_V = std::numeric_limits<int>::min();
  int max_nI = 1;
  for (int nI = 1; nI < length(fsc.nodes); ++nI) {
    double V_temp = fsc.nodes[n].V_a_o_n[a][o][nI];
    if (V_temp > max_V) {
      max_V = V_temp;
      max_nI = nI;
    }
  }
  return max_V, max_nI;
}

void BackUp(double b, AlphaVectorFSC fsc, int L, int nb_sample, POMDP pomdp) {
  std::vector<int> action_space = pomdp.action_space();
  std::vector<int> obs_space = pomdp.obs_space();

  // TODO: C++ implementation of the`CreatNode` function

  for (int a = 0; a < action_space.size(); ++a) {
    for (int o = 0; o < obs_space.size(); ++o) {
      for (int nI = 0; nI < length(fsc.nodes); ++nI) {
        fsc.nodes[nI].V_a_o_n[a][o] = 0;
      }
    }
  }

  for (int a = 0; a < action_space.size(); ++a) {
    for (int i = 0; i < nb_sample; ++i) {
      double s = rand(fsc.nodes[0]._state_particles);
      std::vector<double> sp, o, r;
      // TODO: C++ implementation of the`gen` function

      fsc.nodes[0]._R_action[a] += r;
      for (int nI = 0; nI < length(fsc.nodes); ++nI) {
        double V_nI = SimulateTrajectory(nI, fsc, sp, L, pomdp);
        fsc.nodes[nI].V_a_o_n[a][o] += V_nI;
      }
    }

    for (int o = 0; o < obs_space.size(); ++o) {
      int V_a_o, nI_a_o = FindMaxValueNode(fsc.nodes[0], fsc, a, o);
      fsc.eta[length(fsc.nodes)][a][o] = nI_a_o;
      fsc.nodes[length(fsc.nodes)].Q_action[a] += pomdp.discount() * V_a_o;
    }
    fsc.nodes[0].Q_action[a] /= nb_sample;
  }

  // TODO: C++ implementation of the`GetBestAction` function
}

void MCVIPlanning(int nb_particles, double b0, AlphaVectorFSC fsc, POMDP pomdp,
                  double epsilon) {
  AlphaVectorFSC node_start = CreatNode(b0, fsc._action_space, fsc._obs_space);
  fsc.nodes.push_back(node_start);

  // TODO: C++ implementation of the`CreatNode` function

  while (true) {
    // TODO: C++ implementation of the solver
  }
}
