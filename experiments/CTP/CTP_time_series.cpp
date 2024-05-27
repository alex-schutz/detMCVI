#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include "AOStar.h"
#include "CTP.h"
#include "MCVI.h"

using namespace MCVI;

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
      eval_interval_ms, completion_threshold, completion_reps);
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

int main() {
  std::mt19937_64 rng(std::random_device{}());

  // Initialise the POMDP
  std::cout << "Initialising CTP" << std::endl;
  auto pomdp = CTP(rng);

  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  std::fstream ctp_graph("ctp_graph.dot", std::fstream::out);
  pomdp.visualiseGraph(ctp_graph);
  ctp_graph.close();

  // Initial belief parameters
  const int64_t nb_particles_b0 = 100000;
  const int64_t max_belief_samples = 20000;

  // MCVI parameters
  const int64_t max_sim_depth = 30;
  const int64_t max_node_size = 10000;
  const int64_t eval_depth = 30;
  const int64_t eval_epsilon = 0.005;
  const double converge_thresh = 0.005;
  const int64_t max_time_ms = 1000 * 60 * 60 * 5;

  // Evaluation parameters
  const int64_t max_eval_steps = 30;
  const int64_t n_eval_trials = 10000;
  const int64_t eval_interval_ms = 10;
  const int64_t completion_threshold = 9900;
  const int completion_reps = 3;

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(nb_particles_b0, &pomdp);
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;
  if (max_belief_samples < init_belief.size()) {
    std::cout << "Downsampling belief" << std::endl;
    init_belief = DownsampleBelief(init_belief, max_belief_samples, rng);
  }

  // Run MCVI
  auto mcvi_ctp = new CTP(pomdp);
  runMCVIIncrements(mcvi_ctp, init_belief, rng, max_sim_depth, max_node_size,
                    eval_depth, eval_epsilon, converge_thresh, max_time_ms,
                    max_eval_steps, n_eval_trials, 10 * nb_particles_b0,
                    eval_interval_ms, completion_threshold, completion_reps);

  // Compare to AO*
  auto aostar_ctp = new CTP(pomdp);
  runAOStarIncrements(aostar_ctp, init_belief, rng, eval_depth, eval_epsilon,
                      max_time_ms, max_eval_steps, n_eval_trials,
                      10 * nb_particles_b0, eval_interval_ms,
                      completion_threshold, completion_reps);

  return 0;
}
