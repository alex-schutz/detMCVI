#include "wumpus.h"

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

void runMCVI(Wumpus* pomdp, const BeliefDistribution& init_belief,
             std::mt19937_64& rng, int64_t max_sim_depth, int64_t max_node_size,
             int64_t eval_depth, int64_t eval_epsilon, double converge_thresh,
             int64_t max_iter, int64_t max_computation_ms,
             int64_t max_eval_steps, int64_t n_eval_trials,
             int64_t nb_particles_b0) {
  // Initialise heuristic
  OptimalPath solver(pomdp);

  // Initialise the FSC
  std::cout << "Initialising FSC" << std::endl;
  const auto init_fsc = AlphaVectorFSC(max_node_size);
  //   const auto init_fsc =
  //       InitialiseFSC(solver, init_belief, max_sim_depth, max_node_size,
  //       &pomdp);
  //   init_fsc.GenerateGraphviz(std::cerr, pomdp.getActions(), pomdp.getObs());

  // Run MCVI
  std::cout << "Running MCVI" << std::endl;
  const std::chrono::steady_clock::time_point mcvi_begin =
      std::chrono::steady_clock::now();
  auto planner = MCVIPlanner(pomdp, init_fsc, init_belief, solver, rng);
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
                                      nb_particles_b0, std::nullopt);
  std::cout << "detMCVI policy FSC contains " << fsc.NumNodes() << " nodes."
            << std::endl;
  std::cout << std::endl;

  // Draw the internal belief tree
  std::fstream belief_tree("belief_tree.dot", std::fstream::out);
  root->DrawBeliefTree(belief_tree);
  belief_tree.close();
}

void runAOStar(Wumpus* pomdp, const BeliefDistribution& init_belief,
               std::mt19937_64& rng, int64_t eval_depth, int64_t max_iter,
               int64_t max_computation_ms, int64_t max_eval_steps,
               int64_t n_eval_trials, int64_t nb_particles_b0) {
  // Initialise heuristic
  OptimalPath solver(pomdp);

  // Create root belief node
  const double init_upper =
      CalculateUpperBound(init_belief, 0, eval_depth, solver, pomdp);
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, init_upper, -std::numeric_limits<double>::infinity());

  // Run AO*
  std::cout << "Running AO* on belief tree" << std::endl;
  const std::chrono::steady_clock::time_point ao_begin =
      std::chrono::steady_clock::now();
  RunAOStar(root, max_iter, max_computation_ms, solver, eval_depth, rng, pomdp);
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
                                 nb_particles_b0, pomdp, rng, solver,
                                 std::nullopt, "AO*");
  std::cout << "AO* greedy policy tree contains " << n_greedy_nodes << " nodes."
            << std::endl;
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

void runPOMCP(Wumpus* pomdp, std::mt19937_64& rng, int64_t init_belief_size,
              double pomcp_c, int64_t pomcp_nb_rollout,
              std::chrono::microseconds pomcp_time_out, double pomcp_epsilon,
              int64_t pomcp_depth, int64_t max_eval_steps,
              int64_t n_eval_trials, int64_t nb_particles_b0) {
  OptimalPath solver(pomdp);

  const double gamma = pomdp->GetDiscount();
  std::cerr << "Initialising POMCP" << std::endl;
  auto pomcp = POMCP::PomcpPlanner(pomdp, gamma);
  pomcp.Init(pomcp_c, pomcp_nb_rollout, pomcp_time_out, pomcp_epsilon,
             pomcp_depth);
  std::cerr << "POMCP initialised" << std::endl;

  EvaluationStats eval_stats;
  std::cerr << "Generating belief particles" << std::endl;
  std::vector<State> init_belief_p;
  for (int64_t n = 0; n < init_belief_size / 10; ++n)
    init_belief_p.push_back(pomdp->SampleStartState());
  POMCP::BeliefParticles init_belief(init_belief_p);
  std::cerr << "Running POMCP offline" << std::endl;
  POMCP::TreeNodePtr root_node = pomcp.SearchOffline(init_belief);

  std::cerr << "Generating evaluation set" << std::endl;
  const BeliefDistribution init_belief_eval =
      SampleInitialBelief(nb_particles_b0, pomdp);
  State initial_state = {};
  for (int64_t sim = 0; sim < n_eval_trials; ++sim) {
    State state = SampleOneState(init_belief_eval, rng);
    initial_state = state;
    double sum_r = 0.0;
    int64_t i = 0;
    const auto [optimal, path] =
        solver.getMaxReward(initial_state, max_eval_steps);
    const auto final_state = std::get<1>(path.back());
    const bool has_soln = final_state[pomdp->sfIdx("player_gold")] > 0;
    // std::cerr << "Running trial " << sim << std::endl;
    POMCP::TreeNodePtr tr_node = root_node;
    for (; i < max_eval_steps; ++i) {
      const int64_t action = (tr_node) ? POMCP::BestAction(tr_node) : -1;
      if (!tr_node || action == -1) {
        if (!has_soln) {
          eval_stats.no_solution_off_policy.update(sum_r - optimal);
        } else {
          eval_stats.off_policy.update(sum_r);
        }
        break;
      }
      const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
      sum_r += std::pow(gamma, i) * reward;

      //   std::cout << "---------" << std::endl;
      //   std::cout << "step: " << i << std::endl;
      //   std::cout << "state: <";
      //   for (const auto& state_elem : state) std::cout << state_elem << ", ";
      //   std::cout << ">" << std::endl;
      //   std::cout << "perform action: " << action << std::endl;
      //   std::cout << "receive obs: " << obs << std::endl;
      //   std::cout << "reward: " << reward << std::endl;

      if (done) {
        eval_stats.complete.update(sum_r);
        break;
      }

      state = sNext;
      tr_node = tr_node->GetChildNode(action, obs);
    }
    if (i == max_eval_steps) {
      if (!has_soln) {
        eval_stats.no_solution_on_policy.update(sum_r - optimal);
      } else {
        eval_stats.max_depth.update(sum_r - optimal);
      }
    }
  }
  std::cout << "Evaluation of POMCP (offline) policy (" << max_eval_steps
            << " steps, " << n_eval_trials << " trials):" << std::endl;
  const std::string alg_name = "POMCP";
  PrintStats(eval_stats.complete, alg_name + " completed problem");
  PrintStats(eval_stats.off_policy, alg_name + " exited policy");
  PrintStats(eval_stats.max_depth, alg_name + " max depth");
  PrintStats(eval_stats.no_solution_on_policy,
             alg_name + " no solution (on policy)");
  PrintStats(eval_stats.no_solution_off_policy,
             alg_name + " no solution (exited policy)");
}

int main(int argc, char* argv[]) {
  std::mt19937_64 rng(RANDOM_SEED);

  // Initialise the POMDP
  std::cout << "Initialising Wumpus" << std::endl;
  auto pomdp = Wumpus(4, rng);

  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  // Initial belief parameters
  const int64_t nb_particles_b0 = 100000;
  const int64_t max_belief_samples = 20000;

  // MCVI parameters
  const int64_t max_sim_depth = 100;
  const int64_t max_node_size = 10000;
  const int64_t eval_depth = 50;
  const int64_t eval_epsilon = 0.005;
  const double converge_thresh = 0.005;
  const int64_t max_iter = 500;
  int64_t max_time_ms = 10000;

  // Evaluation parameters
  const int64_t max_eval_steps = 100;
  const int64_t n_eval_trials = 10000;

  parseCommandLine(argc, argv, max_time_ms);

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(nb_particles_b0, &pomdp);
  if (max_belief_samples < init_belief.size()) {
    std::cout << "Initial belief size: " << init_belief.size() << std::endl;
    std::cout << "Downsampling belief" << std::endl;
    init_belief = DownsampleBelief(init_belief, max_belief_samples, rng);
  }
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;

  // Run POMCP
  auto pomcp = new Wumpus(pomdp);
  const double pomcp_c = 2.0;
  const int64_t pomcp_nb_rollout = 200;
  const std::chrono::microseconds pomcp_time_out =
      std::chrono::milliseconds(max_time_ms);
  const double pomcp_epsilon = 0.01;
  const int64_t pomcp_depth = max_sim_depth;
  runPOMCP(pomcp, rng, max_belief_samples, pomcp_c, pomcp_nb_rollout,
           pomcp_time_out, pomcp_epsilon, pomcp_depth, max_sim_depth,
           n_eval_trials, nb_particles_b0);

  // Run MCVI
  auto mcvi = new Wumpus(pomdp);
  runMCVI(mcvi, init_belief, rng, max_sim_depth, max_node_size, eval_depth,
          eval_epsilon, converge_thresh, max_iter, max_time_ms, max_eval_steps,
          n_eval_trials, nb_particles_b0);

  // Compare to AO*
  auto aostar = new Wumpus(pomdp);
  runAOStar(aostar, init_belief, rng, eval_depth, max_iter, max_time_ms,
            max_eval_steps, n_eval_trials, nb_particles_b0);

  return 0;
}