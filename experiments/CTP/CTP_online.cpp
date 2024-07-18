#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>

#include "AOStar.h"
#include "CTP.h"
#include "MCVI.h"
#include "Sample.h"

#define RANDOM_SEED (42)

using namespace MCVI;

std::atomic<bool> exit_flag = false;

class NullBuffer : public std::streambuf {
 public:
  int overflow(int c) { return c; }
};

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

using OnlineFuncPtr = std::tuple<double, std::chrono::microseconds, bool> (*)(
    const CTP_Online&, const BeliefDistribution&, std::mt19937_64&,
    const CTPParams&, std::ostream&);

static double s_time_diff(const std::chrono::steady_clock::time_point& begin,
                          const std::chrono::steady_clock::time_point& end) {
  return (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count()) /
         1e6;
}

std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> runMCVI(
    CTP* pomdp, const BeliefDistribution& init_belief, std::mt19937_64& rng,
    int64_t max_sim_depth, int64_t max_node_size, int64_t eval_depth,
    int64_t eval_epsilon, double converge_thresh, int64_t max_computation_ms,
    const OptimalPath& solver, std::ostream& fs) {
  // Initialise the FSC
  const auto init_fsc = AlphaVectorFSC(max_node_size);

  // Run MCVI
  fs << "Running MCVI" << std::endl;
  const std::chrono::steady_clock::time_point mcvi_begin =
      std::chrono::steady_clock::now();
  auto planner = MCVIPlanner(pomdp, init_fsc, init_belief, solver, rng);
  const auto [fsc, root] = planner.Plan(
      max_sim_depth, converge_thresh, std::numeric_limits<int64_t>::max(),
      max_computation_ms, eval_depth, eval_epsilon, exit_flag);
  const std::chrono::steady_clock::time_point mcvi_end =
      std::chrono::steady_clock::now();
  fs << "MCVI complete (" << s_time_diff(mcvi_begin, mcvi_end) << " seconds)"
     << std::endl;

  fs << "detMCVI policy FSC contains " << fsc.NumNodes() << " nodes."
     << std::endl;
  fs << std::endl;

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

std::shared_ptr<BeliefTreeNode> runAOStar(
    CTP* pomdp, const BeliefDistribution& init_belief, std::mt19937_64& rng,
    int64_t eval_depth, int64_t max_computation_ms, OptimalPath& solver,
    std::ostream& fs) {
  // Create root belief node
  const double init_upper =
      CalculateUpperBound(init_belief, 0, eval_depth, solver, pomdp);
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, init_upper, -std::numeric_limits<double>::infinity());

  // Run AO*
  fs << "Running AO* on belief tree" << std::endl;
  const std::chrono::steady_clock::time_point ao_begin =
      std::chrono::steady_clock::now();
  RunAOStar(root, std::numeric_limits<int64_t>::max(), max_computation_ms,
            solver, eval_depth, rng, pomdp);
  const std::chrono::steady_clock::time_point ao_end =
      std::chrono::steady_clock::now();
  fs << "AO* complete (" << s_time_diff(ao_begin, ao_end) << " seconds)"
     << std::endl;

  std::fstream policy_tree("greedy_policy_tree.dot", std::fstream::out);
  const int64_t n_greedy_nodes = root->DrawPolicyTree(policy_tree);
  policy_tree.close();
  fs << "AO* greedy policy tree contains " << n_greedy_nodes << " nodes."
     << std::endl;

  return root;
}

std::tuple<double, std::chrono::microseconds, bool> MCVIOnline(
    const CTP_Online& pomdp, const BeliefDistribution& init_belief,
    std::mt19937_64& rng, const CTPParams& params, std::ostream& fs) {
  const auto begin = std::chrono::steady_clock::now();
  // Run MCVI
  auto mcvi_ctp = new CTP_Online(pomdp);
  OptimalPath solver(mcvi_ctp);

  bool completed = false;

  const double gamma = mcvi_ctp->GetDiscount();
  State state = SampleOneState(init_belief, rng);
  State belief_state = mcvi_ctp->InitialBeliefState();
  std::shared_ptr<BeliefTreeNode> tree_node = nullptr;
  double sum_r = 0.0;
  int64_t nI = -1;
  AlphaVectorFSC fsc = AlphaVectorFSC(params.max_node_size);
  for (int64_t i = 0; i < params.max_sim_depth; ++i) {
    if (nI == -1 || tree_node == nullptr) {
      fs << "Reached end of policy. Recalculating." << std::endl;

      auto belief =
          mcvi_ctp->SampleFromBeliefState(params.nb_particles_b0, belief_state);
      belief = DownsampleBelief(belief, params.max_belief_samples, rng);
      fs << "Belief size " << belief.size() << std::endl;

      const auto a = runMCVI(mcvi_ctp, belief, rng, params.max_sim_depth,
                             params.max_node_size, params.max_sim_depth,
                             params.eval_epsilon, params.converge_thresh,
                             params.max_time_ms, solver, fs);
      fsc = a.first;
      nI = fsc.GetStartNodeIndex();
      tree_node = a.second;
    }
    const int64_t action = fsc.GetNode(nI).GetBestAction();
    fs << "---------" << std::endl;
    fs << "step: " << i << std::endl;
    fs << "state: <";
    for (const auto& state_elem : state) fs << state_elem << ", ";
    fs << ">" << std::endl;
    fs << "perform action: " << action << std::endl;
    const auto [sNext, obs, reward, done] = mcvi_ctp->Step(state, action);

    fs << "receive obs: " << obs << std::endl;
    belief_state = mcvi_ctp->ApplyObservation(belief_state, obs);

    fs << "reward: " << reward << std::endl;
    sum_r += std::pow(gamma, i) * reward;

    nI = fsc.GetEdgeValue(nI, obs);

    if (done) {
      fs << "Reached terminal state." << std::endl;
      completed = true;
      break;
    }
    state = sNext;
    tree_node = tree_node->GetChild(action, obs);
  }
  fs << "sum reward: " << sum_r << std::endl << std::endl;
  const auto end = std::chrono::steady_clock::now();
  return std::make_tuple(
      sum_r, std::chrono::duration_cast<std::chrono::microseconds>(end - begin),
      completed);
}

std::tuple<double, std::chrono::microseconds, bool> AOStarOnline(
    const CTP_Online& pomdp, const BeliefDistribution& init_belief,
    std::mt19937_64& rng, const CTPParams& params, std::ostream& fs) {
  const auto begin = std::chrono::steady_clock::now();
  // Run AO*
  auto ao_ctp = new CTP_Online(pomdp);
  OptimalPath solver(ao_ctp);

  bool completed = false;

  const double gamma = ao_ctp->GetDiscount();
  State state = SampleOneState(init_belief, rng);
  State belief_state = ao_ctp->InitialBeliefState();
  std::shared_ptr<BeliefTreeNode> tree_node = nullptr;
  double sum_r = 0.0;
  AlphaVectorFSC fsc = AlphaVectorFSC(params.max_node_size);
  for (int64_t i = 0; i < params.max_sim_depth; ++i) {
    if (tree_node == nullptr || tree_node->GetBestActUBound() == -1) {
      fs << "Reached end of policy. Recalculating." << std::endl;

      auto belief =
          ao_ctp->SampleFromBeliefState(params.nb_particles_b0, belief_state);
      belief = DownsampleBelief(belief, params.max_belief_samples, rng);
      fs << "Belief size " << belief.size() << std::endl;

      tree_node = runAOStar(ao_ctp, init_belief, rng, params.max_sim_depth,
                            params.max_time_ms, solver, fs);
    }
    const int64_t action = tree_node->GetBestActUBound();
    fs << "---------" << std::endl;
    fs << "step: " << i << std::endl;
    fs << "state: <";
    for (const auto& state_elem : state) fs << state_elem << ", ";
    fs << ">" << std::endl;
    fs << "perform action: " << action << std::endl;
    const auto [sNext, obs, reward, done] = ao_ctp->Step(state, action);

    fs << "receive obs: " << obs << std::endl;
    belief_state = ao_ctp->ApplyObservation(belief_state, obs);

    fs << "reward: " << reward << std::endl;
    sum_r += std::pow(gamma, i) * reward;

    if (done) {
      fs << "Reached terminal state." << std::endl;
      completed = true;
      break;
    }
    state = sNext;
    tree_node = tree_node->GetChild(action, obs);
  }
  fs << "sum reward: " << sum_r << std::endl << std::endl;
  const auto end = std::chrono::steady_clock::now();
  return std::make_tuple(
      sum_r, std::chrono::duration_cast<std::chrono::microseconds>(end - begin),
      completed);
}

std::pair<Welford, Welford> RunOnlineTrials(
    OnlineFuncPtr func, int64_t n_trials, const CTP_Online& pomdp,
    const BeliefDistribution& init_belief, std::mt19937_64& rng,
    const CTPParams& params) {
  Welford reward_stats;
  Welford time_stats;

  NullBuffer null_buffer;
  std::ostream null_stream(&null_buffer);

  for (int64_t i = 0; i < n_trials; ++i) {
    const auto [reward, timing, completed] =
        func(pomdp, init_belief, rng, params, null_stream);
    if (completed) {
      reward_stats.update(reward);
      time_stats.update(timing.count());
    }
  }
  return std::make_pair(reward_stats, time_stats);
}

int main(int argc, char* argv[]) {
  const CTPParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  const int64_t n_trials = 10;

  std::vector<int64_t> nodes;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> edges;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> stoch_edges;
  int64_t origin;
  int64_t goal;
  ctpGraphFromFile(params.filename, nodes, edges, stoch_edges, origin, goal);
  auto ctp = CTP(rng, nodes, edges, stoch_edges, origin, goal);
  auto pomdp = CTP_Online(ctp);

  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  std::fstream ctp_graph("ctp_graph.dot", std::fstream::out);
  pomdp.visualiseGraph(ctp_graph);
  ctp_graph.close();

  std::cerr << "Max planning time per step: " << params.max_time_ms
            << std::endl;

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(params.nb_particles_b0, &pomdp);
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;
  init_belief = DownsampleBelief(init_belief, params.max_belief_samples, rng);

  MCVIOnline(pomdp, init_belief, rng, params, std::cout);

  AOStarOnline(pomdp, init_belief, rng, params, std::cout);

  const auto [mcvi_rw, mcvi_ts] =
      RunOnlineTrials(MCVIOnline, n_trials, pomdp, init_belief, rng, params);
  PrintStats(mcvi_rw, "MCVI Online Reward");
  PrintStats(mcvi_ts, "MCVI Online Times");

  const auto [ao_rw, ao_ts] =
      RunOnlineTrials(AOStarOnline, n_trials, pomdp, init_belief, rng, params);
  PrintStats(ao_rw, "AO* Online Reward");
  PrintStats(ao_ts, "AO* Online Times");

  return 0;
}
