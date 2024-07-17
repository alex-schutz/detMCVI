#include "CTP.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>

#include "AOStar.h"
#include "MCVI.h"
#include "POMCP.h"

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
             int64_t nb_particles_b0) {
  // Initialise heuristic
  PathToTerminal ptt(pomdp);

  // Initialise the FSC
  std::cout << "Initialising FSC" << std::endl;
  const auto init_fsc = AlphaVectorFSC(max_node_size);
  //   const auto init_fsc =
  //       InitialiseFSC(ptt, init_belief, max_sim_depth, max_node_size,
  //       &pomdp);
  //   init_fsc.GenerateGraphviz(std::cerr, pomdp.getActions(), pomdp.getObs());

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
  planner.EvaluationWithSimulationFSC(max_eval_steps, n_eval_trials,
                                      nb_particles_b0);
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
               int64_t n_eval_trials, int64_t nb_particles_b0) {
  // Initialise heuristic
  PathToTerminal ptt(pomdp);

  // Create root belief node
  const double init_upper =
      CalculateUpperBound(init_belief, 0, eval_depth, ptt, pomdp);
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, init_upper, -std::numeric_limits<double>::infinity());

  // Run AO*
  std::cout << "Running AO* on belief tree" << std::endl;
  const std::chrono::steady_clock::time_point ao_begin =
      std::chrono::steady_clock::now();
  RunAOStar(root, max_iter, max_computation_ms, ptt, eval_depth, rng, pomdp);
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
  EvaluationWithGreedyTreePolicy(root, max_eval_steps, n_eval_trials,
                                 nb_particles_b0, pomdp, rng, ptt, "AO*");
  std::cout << "AO* greedy policy tree contains " << n_greedy_nodes << " nodes."
            << std::endl;
}

void runPOMCP(CTP* pomdp, std::mt19937_64& rng, double pomcp_c,
              int64_t pomcp_nb_rollout,
              std::chrono::microseconds pomcp_time_out, double pomcp_epsilon,
              int64_t pomcp_depth, int64_t max_eval_steps,
              int64_t n_eval_trials, int64_t nb_particles_b0) {
  const double gamma = pomdp->GetDiscount();
  std::cerr << "Initialising POMCP" << std::endl;
  auto pomcp = POMCP::PomcpPlanner(pomdp, gamma);
  pomcp.Init(pomcp_c, pomcp_nb_rollout, pomcp_time_out, pomcp_epsilon,
             pomcp_depth);
  std::cerr << "POMCP initialised" << std::endl;

  EvaluationStats eval_stats;
  std::cerr << "Generating belief particles" << std::endl;
  std::vector<State> init_belief_p;
  for (int64_t n = 0; n < nb_particles_b0 / 10; ++n)
    init_belief_p.push_back(pomdp->SampleStartState());
  POMCP::BeliefParticles init_belief(init_belief_p);

  std::cerr << "Generating evaluation set" << std::endl;
  const BeliefDistribution init_belief_eval =
      SampleInitialBelief(nb_particles_b0, pomdp);
  State initial_state = {};
  for (int64_t sim = 0; sim < n_eval_trials; ++sim) {
    State state = SampleOneState(init_belief_eval, rng);
    initial_state = state;
    double sum_r = 0.0;
    POMCP::BeliefParticles curr_belief = init_belief;
    int64_t i = 0;
    std::cerr << "Running trial " << sim << std::endl;
    for (; i < max_eval_steps; ++i) {
      const int64_t action = pomcp.Search(curr_belief);
      const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
      sum_r += std::pow(gamma, i) * reward;

      std::cout << "---------" << std::endl;
      std::cout << "step: " << i << std::endl;
      std::cout << "state: <";
      for (const auto& state_elem : state) std::cout << state_elem << ", ";
      std::cout << ">" << std::endl;
      std::cout << "perform action: " << action << std::endl;
      std::cout << "receive obs: " << obs << std::endl;
      std::cout << "reward: " << reward << std::endl;

      if (done) {
        eval_stats.complete.update(sum_r);
        break;
      }

      state = sNext;

      std::vector<State> new_belief_p;
      for (const auto& s : curr_belief.getParticles()) {
        State sNext;
        pomdp->applyActionToState(s, action, sNext);
        new_belief_p.push_back(sNext);
      }
      curr_belief = POMCP::BeliefParticles(new_belief_p);
    }
    if (i == max_eval_steps) {
      //   if (!StateHasSolution(initial_state, ptt, max_steps)) {
      //     eval_stats.no_solution_on_policy.update(sum_r);
      //   } else {
      eval_stats.max_iterations.update(sum_r);
      //   }
    }
  }
  std::cout << "Evaluation of POMCP (online) policy (" << max_eval_steps
            << " steps, " << n_eval_trials << " trials):" << std::endl;
  const std::string alg_name = "POMCP";
  PrintStats(eval_stats.complete, alg_name + " completed problem");
  PrintStats(eval_stats.off_policy, alg_name + " exited policy");
  PrintStats(eval_stats.max_iterations, alg_name + " max iterations");
  PrintStats(eval_stats.no_solution_on_policy,
             alg_name + " no solution (on policy)");
  PrintStats(eval_stats.no_solution_off_policy,
             alg_name + " no solution (exited policy)");
}

int main(int argc, char* argv[]) {
  const CTPParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  std::vector<int64_t> nodes;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> edges;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> stoch_edges;
  int64_t origin;
  int64_t goal;
  ctpGraphFromFile(params.filename, nodes, edges, stoch_edges, origin, goal);

  // Initialise the POMDP
  std::cout << "Initialising CTP" << std::endl;
  auto pomdp = CTP(rng, nodes, edges, stoch_edges, origin, goal);

  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  std::fstream ctp_graph("ctp_graph.dot", std::fstream::out);
  pomdp.visualiseGraph(ctp_graph);
  ctp_graph.close();

  // Evaluation parameters
  const int64_t n_eval_trials = 10000;

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(params.nb_particles_b0, &pomdp);
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;
  if (params.max_belief_samples < (int64_t)init_belief.size()) {
    std::cout << "Downsampling belief" << std::endl;
    init_belief = DownsampleBelief(init_belief, params.max_belief_samples, rng);
  }
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;

  // Run MCVI
  auto mcvi_ctp = new CTP(pomdp);
  runMCVI(mcvi_ctp, init_belief, rng, params.max_sim_depth,
          params.max_node_size, params.max_sim_depth, params.eval_epsilon,
          params.converge_thresh, params.max_iter, params.max_time_ms,
          params.max_sim_depth, n_eval_trials, 10 * params.nb_particles_b0);

  // Compare to AO*
  auto aostar_ctp = new CTP(pomdp);
  runAOStar(aostar_ctp, init_belief, rng, params.max_sim_depth, params.max_iter,
            params.max_time_ms, params.max_sim_depth, n_eval_trials,
            10 * params.nb_particles_b0);

  auto pomcp_ctp = new CTP(pomdp);
  const double pomcp_c = 2.0;
  const int64_t pomcp_nb_rollout = 200;
  const std::chrono::microseconds pomcp_time_out = std::chrono::microseconds(
      params.max_time_ms * 1000 / params.max_sim_depth);
  const double pomcp_epsilon = 0.01;
  const int64_t pomcp_depth = params.max_sim_depth;
  runPOMCP(pomcp_ctp, rng, pomcp_c, pomcp_nb_rollout, pomcp_time_out,
           pomcp_epsilon, pomcp_depth, params.max_sim_depth, n_eval_trials,
           10 * params.nb_particles_b0);

  return 0;
}
