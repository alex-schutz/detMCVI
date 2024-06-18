#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>

#include "AOStar.h"
#include "CTP.h"
#include "MCVI.h"

#define RANDOM_SEED (42)

using namespace MCVI;

std::atomic<bool> exit_flag = false;

void runMCVIIncrements(CTP* pomdp, const BeliefDistribution& init_belief,
                       std::mt19937_64& rng, int64_t max_sim_depth,
                       int64_t max_node_size, int64_t eval_depth,
                       int64_t eval_epsilon, double converge_thresh,
                       int64_t max_time_ms, int64_t max_eval_steps,
                       int64_t n_eval_trials, int64_t nb_particles_b0,
                       int64_t eval_interval_ms, int64_t completion_threshold,
                       int64_t completion_reps) {
  // Initialise heuristic
  PathToTerminal ptt(pomdp);

  // Initialise the FSC
  const auto init_fsc = AlphaVectorFSC(max_node_size);

  // Run MCVI
  std::cout << "Running MCVI" << std::endl;
  auto planner = MCVIPlanner(pomdp, init_fsc, init_belief, ptt, rng);
  const auto [fsc, root] = planner.PlanAndEvaluate(
      max_sim_depth, converge_thresh, 100000000000, max_time_ms, eval_depth,
      eval_epsilon, max_eval_steps, n_eval_trials, nb_particles_b0,
      eval_interval_ms, completion_threshold, completion_reps, exit_flag);
}

void runAOStarIncrements(CTP* pomdp, const BeliefDistribution& init_belief,
                         std::mt19937_64& rng, int64_t eval_depth,
                         int64_t eval_epsilon, int64_t max_time_ms,
                         int64_t max_eval_steps, int64_t n_eval_trials,
                         int64_t nb_particles_b0, int64_t eval_interval_ms,
                         int64_t completion_threshold,
                         int64_t completion_reps) {
  // Initialise heuristic
  PathToTerminal ptt(pomdp);

  // Create root belief node
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, ptt, eval_depth, eval_epsilon, pomdp);

  // Run AO*
  std::cout << "Running AO* on belief tree" << std::endl;
  RunAOStarAndEvaluate(root, 100000000000, max_time_ms, ptt, eval_depth,
                       eval_epsilon, max_eval_steps, n_eval_trials,
                       nb_particles_b0, eval_interval_ms, completion_threshold,
                       completion_reps, rng, ptt, pomdp);
}

void parseSeriesArgs(int argc, char** argv, int64_t& n_eval_trials,
                     int64_t& eval_interval_ms, int64_t& completion_threshold,
                     int& completion_reps) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--n_eval_trials") == 0 && i + 1 < argc) {
      n_eval_trials = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--eval_interval_ms") == 0 && i + 1 < argc) {
      eval_interval_ms = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--completion_threshold") == 0 && i + 1 < argc) {
      completion_threshold = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--completion_reps") == 0 && i + 1 < argc) {
      completion_reps = std::stoi(argv[++i]);
    }
  }
}

int main(int argc, char* argv[]) {
  const CTPParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  // Initialise the POMDP
  std::cout << "Initialising CTP" << std::endl;
  auto pomdp = CTP(rng);

  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  std::fstream ctp_graph("ctp_graph.dot", std::fstream::out);
  pomdp.visualiseGraph(ctp_graph);
  ctp_graph.close();

  // Evaluation parameters
  int64_t n_eval_trials = 10000;
  int64_t eval_interval_ms = 10;
  int64_t completion_threshold = 9900;
  int completion_reps = 3;
  parseSeriesArgs(argc, argv, n_eval_trials, eval_interval_ms,
                  completion_threshold, completion_reps);

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(params.nb_particles_b0, &pomdp);
  if (params.max_belief_samples < (int64_t)init_belief.size()) {
    std::cout << "Downsampling belief" << std::endl;
    init_belief = DownsampleBelief(init_belief, params.max_belief_samples, rng);
  }
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;

  // Run MCVI
  auto mcvi_ctp = new CTP(pomdp);
  runMCVIIncrements(mcvi_ctp, init_belief, rng, params.max_sim_depth,
                    params.max_node_size, params.max_sim_depth,
                    params.eval_epsilon, params.converge_thresh,
                    params.max_time_ms, params.max_sim_depth, n_eval_trials,
                    10 * params.nb_particles_b0, eval_interval_ms,
                    completion_threshold, completion_reps);

  // Compare to AO*
  auto aostar_ctp = new CTP(pomdp);
  runAOStarIncrements(aostar_ctp, init_belief, rng, params.max_sim_depth,
                      params.eval_epsilon, params.max_time_ms,
                      params.max_sim_depth, n_eval_trials,
                      10 * params.nb_particles_b0, eval_interval_ms,
                      completion_threshold, completion_reps);

  return 0;
}
