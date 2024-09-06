#include "CTP.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>

#include "AOStar.h"
#include "MCVI.h"
#include "POMCP.h"
#include "Params.h"
#include "QMDPTree.h"

#define RANDOM_SEED (42)

using namespace MCVI;

std::atomic<bool> exit_flag = false;

static double s_time_diff(const std::chrono::steady_clock::time_point& begin,
                          const std::chrono::steady_clock::time_point& end) {
  return (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count()) /
         1000.0;
}

void runMCVI(CTP* pomdp, const BeliefDistribution& init_belief,
             std::mt19937_64& rng, int64_t max_sim_depth, int64_t max_node_size,
             int64_t eval_depth, int64_t eval_epsilon, double converge_thresh,
             int64_t max_iter, int64_t max_computation_ms,
             int64_t max_eval_steps, int64_t n_eval_trials,
             int64_t nb_particles_b0, OptimalPath& solver) {
  // Initialise heuristic
  OptimalPath heuristic(pomdp);

  // Initialise the FSC
  std::cout << "Initialising FSC" << std::endl;
  const auto init_fsc = AlphaVectorFSC(max_node_size);

  // Run MCVI
  std::cout << "Running MCVI" << std::endl;
  const std::chrono::steady_clock::time_point mcvi_begin =
      std::chrono::steady_clock::now();
  auto planner = MCVIPlanner(pomdp, init_fsc, init_belief, heuristic, rng);
  const auto [fsc, root] =
      planner.Plan(max_sim_depth, converge_thresh, max_iter, max_computation_ms,
                   eval_depth, eval_epsilon, exit_flag);
  const std::chrono::steady_clock::time_point mcvi_end =
      std::chrono::steady_clock::now();
  std::cout << "detMCVI complete (" << s_time_diff(mcvi_begin, mcvi_end)
            << " seconds)" << std::endl;

  // Draw FSC plot
  std::fstream fsc_graph("fsc.dot", std::fstream::out);
  fsc.GenerateGraphviz(fsc_graph, pomdp->getActions(), pomdp->getObs());
  fsc_graph.close();

  // Simulate the resultant FSC
  std::cout << "Simulation with up to " << max_eval_steps
            << " steps:" << std::endl;
  planner.SimulationWithFSC(max_eval_steps);
  std::cout << std::endl;

  // Evaluate the FSC policy
  std::cout << "Evaluation of policy (" << max_eval_steps << " steps, "
            << n_eval_trials << " trials):" << std::endl;
  EvaluationWithSimulationFSC(
      max_eval_steps, n_eval_trials, nb_particles_b0,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      pomdp, rng, fsc, solver);
  std::cout << "detMCVI policy FSC contains " << fsc.NumNodes() << " nodes."
            << std::endl;
  std::cout << std::endl;

  // Draw the internal belief tree
  std::fstream belief_tree("belief_tree.dot", std::fstream::out);
  root->DrawBeliefTree(belief_tree);
  belief_tree.close();
}

void runAOStar(CTP* pomdp, const BeliefDistribution& init_belief,
               std::mt19937_64& rng, int64_t eval_depth, int64_t max_iter,
               int64_t max_computation_ms, int64_t max_eval_steps,
               int64_t n_eval_trials, int64_t nb_particles_b0,
               OptimalPath& solver) {
  // Initialise heuristic
  OptimalPath heuristic(pomdp);

  // Create root belief node
  const double init_upper =
      CalculateUpperBound(init_belief, 0, eval_depth, heuristic, pomdp);
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, init_upper, -std::numeric_limits<double>::infinity());

  // Run AO*
  std::cout << "Running AO* on belief tree" << std::endl;
  const std::chrono::steady_clock::time_point ao_begin =
      std::chrono::steady_clock::now();
  RunAOStar(root, max_iter, max_computation_ms, heuristic, eval_depth, rng,
            pomdp);
  const std::chrono::steady_clock::time_point ao_end =
      std::chrono::steady_clock::now();
  std::cout << "AO* complete (" << s_time_diff(ao_begin, ao_end) << " seconds)"
            << std::endl;

  // Draw policy tree
  std::fstream policy_tree("greedy_policy_tree.dot", std::fstream::out);
  const int64_t n_greedy_nodes = root->DrawPolicyTree(policy_tree);
  policy_tree.close();

  // Evaluate policy
  std::cout << "Evaluation of alternative (AO* greedy) policy ("
            << max_eval_steps << " steps, " << n_eval_trials
            << " trials):" << std::endl;
  EvaluationWithGreedyTreePolicy(
      root, max_eval_steps, n_eval_trials, nb_particles_b0, pomdp, rng, solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      "AO*");
  std::cout << "AO* greedy policy tree contains " << n_greedy_nodes << " nodes."
            << std::endl;
}

void runQMDP(CTP* pomdp, const BeliefDistribution& init_belief,
             std::mt19937_64& rng, int64_t eval_depth, int64_t max_time_ms,
             int64_t max_eval_steps, int64_t n_eval_trials,
             int64_t nb_particles_b0, OptimalPath& solver) {
  // Initialise heuristic
  OptimalPath heuristic(pomdp);

  // Create root belief node
  const double init_upper =
      CalculateUpperBound(init_belief, 0, eval_depth, heuristic, pomdp);
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, init_upper, -std::numeric_limits<double>::infinity());

  // Run QMDP
  std::cout << "Running QMDP on belief tree" << std::endl;
  RunQMDPAndEvaluate(
      root, max_time_ms, heuristic, eval_depth, max_eval_steps, n_eval_trials,
      nb_particles_b0, rng, solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      pomdp);
}

void runPOMCP(CTP* pomdp, std::mt19937_64& rng, int64_t init_belief_size,
              double pomcp_c, int64_t pomcp_nb_rollout,
              std::chrono::microseconds pomcp_time_out, double pomcp_epsilon,
              int64_t pomcp_depth, int64_t max_eval_steps,
              int64_t n_eval_trials, int64_t nb_particles_b0,
              OptimalPath& solver) {
  const double gamma = pomdp->GetDiscount();
  std::cout << "Initialising POMCP" << std::endl;
  auto pomcp = POMCP::PomcpPlanner(pomdp, gamma);
  pomcp.Init(pomcp_c, pomcp_nb_rollout, pomcp_time_out, pomcp_epsilon,
             pomcp_depth);
  std::cout << "POMCP initialised" << std::endl;

  EvaluationStats eval_stats;
  std::cout << "Generating belief particles" << std::endl;
  std::vector<State> init_belief_p;
  for (int64_t n = 0; n < init_belief_size; ++n)
    init_belief_p.push_back(pomdp->SampleStartState());
  POMCP::BeliefParticles init_belief(init_belief_p);
  std::cout << "Running POMCP offline" << std::endl;
  POMCP::TreeNodePtr root_node = std::make_shared<POMCP::TreeNode>(0);
  pomcp.SearchOffline(init_belief, root_node);

  POMCP::EvaluationWithGreedyTreePolicy(
      root_node, max_eval_steps, n_eval_trials, nb_particles_b0, pomdp, rng,
      solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      "POMCP");
  std::cout << "POMCP offline policy tree contains " << CountNodes(root_node)
            << " nodes." << std::endl;
}

int main(int argc, char* argv[]) {
  const EvalParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  std::vector<int64_t> nodes;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> edges;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> stoch_edges;
  int64_t origin;
  int64_t goal;
  ctpGraphFromFile(params.datafile, nodes, edges, stoch_edges, origin, goal);

  // Initialise the POMDP
  std::cout << "Initialising CTP" << std::endl;
  auto pomdp = CTP(rng, nodes, edges, stoch_edges, origin, goal);

  std::cout << "Action space size: " << pomdp.GetSizeOfA() << std::endl;
  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  std::fstream ctp_graph("ctp_graph.dot", std::fstream::out);
  pomdp.visualiseGraph(ctp_graph);
  ctp_graph.close();

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(params.nb_particles_b0, &pomdp);
  if (params.max_belief_samples < (int64_t)init_belief.size()) {
    std::cout << "Initial belief size: " << init_belief.size() << std::endl;
    std::cout << "Downsampling belief" << std::endl;
    init_belief = DownsampleBelief(init_belief, params.max_belief_samples, rng);
  }
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;

  auto sol_ctp = new CTP(pomdp);
  OptimalPath solver(sol_ctp);

  auto qmdp_ctp = new CTP(pomdp);
  runQMDP(qmdp_ctp, init_belief, rng, params.max_sim_depth, params.max_time_ms,
          params.max_sim_depth, params.n_eval_trials,
          10 * params.nb_particles_b0, solver);
  delete qmdp_ctp;

  // Run MCVI
  auto mcvi_ctp = new CTP(pomdp);
  runMCVI(mcvi_ctp, init_belief, rng, params.max_sim_depth,
          params.max_node_size, params.max_sim_depth, params.eval_epsilon,
          params.converge_thresh, params.max_iterations, params.max_time_ms,
          params.max_sim_depth, params.n_eval_trials,
          10 * params.nb_particles_b0, solver);
  delete mcvi_ctp;

  // Compare to AO*
  auto aostar_ctp = new CTP(pomdp);
  runAOStar(aostar_ctp, init_belief, rng, params.max_sim_depth,
            params.max_iterations, params.max_time_ms, params.max_sim_depth,
            params.n_eval_trials, 10 * params.nb_particles_b0, solver);
  delete aostar_ctp;

  auto pomcp_ctp = new CTP(pomdp);
  const double pomcp_c = 2.0;
  const int64_t pomcp_nb_rollout = 200;
  const std::chrono::microseconds pomcp_time_out =
      std::chrono::milliseconds(params.max_time_ms);
  const double pomcp_epsilon = 0.01;
  const int64_t pomcp_depth = params.max_sim_depth;
  runPOMCP(pomcp_ctp, rng, params.max_belief_samples, pomcp_c, pomcp_nb_rollout,
           pomcp_time_out, pomcp_epsilon, pomcp_depth, params.max_sim_depth,
           params.n_eval_trials, 10 * params.nb_particles_b0, solver);
  delete pomcp_ctp;
  delete sol_ctp;

  return 0;
}
