#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>

#include "CTP.h"
#include "MCVI.h"

using namespace MCVI;

std::atomic<bool> exit_flag = false;

class CTP_Online : public CTP {
 public:
  CTP_Online(CTP& ctp) : CTP(ctp) {}

  State InitialBeliefState() {
    State out_state = SampleStartState();
    for (const auto& [edge, prob] : stoch_edges) {
      out_state[sfIdx(edge2str(edge))] = -1;
    }
    return out_state;
  }

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

  State SampleStartStateFromBeliefState(const State& state) const {
    std::uniform_real_distribution<> unif(0, 1);
    State state_new = state;
    // stochastic edge status
    for (const auto& [edge, p] : stoch_edges) {
      const auto e_idx = sfIdx(edge2str(edge));
      if (state_new.at(e_idx) == -1) state_new[e_idx] = (unif(rng)) < p ? 0 : 1;
    }
    return state_new;
  }

  BeliefDistribution SampleFromBeliefState(int64_t N, const State& state) {
    StateMap<int64_t> state_counts;
    for (int64_t i = 0; i < N; ++i)
      state_counts[SampleStartStateFromBeliefState(state)] += 1;
    auto init_belief = BeliefDistribution();
    for (const auto& [state, count] : state_counts)
      init_belief[state] = (double)count / N;
    return init_belief;
  }
};

static double s_time_diff(const std::chrono::steady_clock::time_point& begin,
                          const std::chrono::steady_clock::time_point& end) {
  return (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count()) /
         1000.0;
}

std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> runMCVI(
    CTP* pomdp, const BeliefDistribution& init_belief, std::mt19937_64& rng,
    int64_t max_sim_depth, int64_t max_node_size, int64_t eval_depth,
    int64_t eval_epsilon, double converge_thresh, int64_t max_iter,
    int64_t max_computation_ms, const PathToTerminal& ptt) {
  // Initialise the FSC
  const auto init_fsc = AlphaVectorFSC(max_node_size);

  // Run MCVI
  std::cout << "Running MCVI" << std::endl;
  const std::chrono::steady_clock::time_point mcvi_begin =
      std::chrono::steady_clock::now();
  auto planner = MCVIPlanner(pomdp, init_fsc, init_belief, ptt, rng);
  const auto [fsc, root] =
      planner.Plan(max_sim_depth, converge_thresh, max_iter, max_computation_ms,
                   eval_depth, eval_epsilon, exit_flag);
  const std::chrono::steady_clock::time_point mcvi_end =
      std::chrono::steady_clock::now();
  std::cout << "MCVI complete (" << s_time_diff(mcvi_begin, mcvi_end)
            << " seconds)" << std::endl;

  std::cout << "detMCVI policy FSC contains " << fsc.NumNodes() << " nodes."
            << std::endl;
  std::cout << std::endl;

  // Draw FSC plot
  std::fstream fsc_graph("fsc.dot", std::fstream::out);
  fsc.GenerateGraphviz(fsc_graph, pomdp->getActions(), pomdp->getObs());
  fsc_graph.close();

  // Draw the internal belief tree
  std::fstream belief_tree("belief_tree.dot", std::fstream::out);
  root->DrawBeliefTree(belief_tree);
  belief_tree.close();
  return {fsc, root};
}

void parseCommandLine(int argc, char* argv[], int64_t& runtime_ms) {
  if (argc > 1) {
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--runtime" && i + 1 < argc) {
        runtime_ms = std::stoi(argv[i + 1]);
        break;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  std::mt19937_64 rng(std::random_device{}());

  // Initialise the POMDP
  std::cout << "Initialising CTP" << std::endl;
  auto ctp = CTP(rng);
  auto pomdp = CTP_Online(ctp);

  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  std::fstream ctp_graph("ctp_graph.dot", std::fstream::out);
  pomdp.visualiseGraph(ctp_graph);
  ctp_graph.close();

  // Initial belief parameters
  const int64_t nb_particles_b0 = 100000;
  const int64_t max_belief_samples = 2000;

  // MCVI parameters
  const int64_t max_sim_depth = 100;
  const int64_t max_node_size = 10000;
  const int64_t eval_depth = 100;
  const int64_t eval_epsilon = 0.005;
  const double converge_thresh = 0.005;
  const int64_t max_iter = 10;
  int64_t max_time_ms = 1000 * 30;

  parseCommandLine(argc, argv, max_time_ms);

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(nb_particles_b0, &pomdp);
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;
  init_belief = DownsampleBelief(init_belief, max_belief_samples, rng);

  // Run MCVI
  auto mcvi_ctp = new CTP_Online(pomdp);
  PathToTerminal ptt(mcvi_ctp);

  const double gamma = mcvi_ctp->GetDiscount();
  State state = SampleOneState(init_belief, rng);
  State belief_state = mcvi_ctp->InitialBeliefState();
  std::shared_ptr<BeliefTreeNode> tree_node = nullptr;
  double sum_r = 0.0;
  int64_t nI = -1;
  AlphaVectorFSC fsc = AlphaVectorFSC(max_node_size);
  for (int64_t i = 0; i < eval_depth; ++i) {
    if (nI == -1 || tree_node == nullptr) {
      std::cout << "Reached end of policy. Recalculating." << std::endl;

      auto belief =
          mcvi_ctp->SampleFromBeliefState(nb_particles_b0, belief_state);
      belief = DownsampleBelief(belief, max_belief_samples, rng);
      std::cout << "Belief size " << belief.size() << std::endl;

      const auto a = runMCVI(mcvi_ctp, belief, rng, max_sim_depth,
                             max_node_size, eval_depth, eval_epsilon,
                             converge_thresh, max_iter, max_time_ms, ptt);
      fsc = a.first;
      nI = fsc.GetStartNodeIndex();
      tree_node = a.second;
    }
    const int64_t action = fsc.GetNode(nI).GetBestAction();
    std::cout << "---------" << std::endl;
    std::cout << "step: " << i << std::endl;
    std::cout << "state: <";
    for (const auto& state_elem : state) std::cout << state_elem << ", ";
    std::cout << ">" << std::endl;
    std::cout << "perform action: " << action << std::endl;
    const auto [sNext, obs, reward, done] = mcvi_ctp->Step(state, action);

    std::cout << "receive obs: " << obs << std::endl;
    belief_state = mcvi_ctp->ApplyObservation(belief_state, obs);

    std::cout << "reward: " << reward << std::endl;
    sum_r += std::pow(gamma, i) * reward;

    nI = fsc.GetEdgeValue(nI, obs);

    if (done) {
      std::cout << "Reached terminal state." << std::endl;
      break;
    }
    state = sNext;
    tree_node = tree_node->GetChild(action, obs);
  }
  std::cout << "sum reward: " << sum_r << std::endl;

  return 0;
}
