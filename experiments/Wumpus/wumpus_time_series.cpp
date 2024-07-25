#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>

#include "AOStar.h"
#include "MCVI.h"
#include "POMCP.h"
#include "Params.h"
#include "wumpus.h"

#define RANDOM_SEED (42)

using namespace MCVI;

std::atomic<bool> exit_flag = false;

void runMCVIIncrements(Wumpus* pomdp, const BeliefDistribution& init_belief,
                       std::mt19937_64& rng, int64_t max_sim_depth,
                       int64_t max_node_size, int64_t eval_depth,
                       int64_t eval_epsilon, double converge_thresh,
                       int64_t max_time_ms, int64_t max_eval_steps,
                       int64_t n_eval_trials, int64_t nb_particles_b0,
                       int64_t eval_interval_ms, int64_t completion_threshold,
                       int64_t completion_reps) {
  // Initialise heuristic
  OptimalPath solver(pomdp);

  // Initialise the FSC
  const auto init_fsc = AlphaVectorFSC(max_node_size);

  // Run MCVI
  std::cout << "Running MCVI" << std::endl;
  auto planner = MCVIPlanner(pomdp, init_fsc, init_belief, solver, rng);
  const auto [fsc, root] = planner.PlanAndEvaluate(
      max_sim_depth, converge_thresh, std::numeric_limits<int64_t>::max(),
      max_time_ms, eval_depth, eval_epsilon, max_eval_steps, n_eval_trials,
      nb_particles_b0, eval_interval_ms, completion_threshold, completion_reps,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      exit_flag);
}

void runAOStarIncrements(Wumpus* pomdp, const BeliefDistribution& init_belief,
                         std::mt19937_64& rng, int64_t eval_depth,
                         int64_t max_time_ms, int64_t max_eval_steps,
                         int64_t n_eval_trials, int64_t nb_particles_b0,
                         int64_t eval_interval_ms, int64_t completion_threshold,
                         int64_t completion_reps) {
  // Initialise heuristic
  OptimalPath solver(pomdp);

  // Create root belief node
  const double init_upper =
      CalculateUpperBound(init_belief, 0, eval_depth, solver, pomdp);
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, init_upper, -std::numeric_limits<double>::infinity());

  // Run AO*
  std::cout << "Running AO* on belief tree" << std::endl;
  RunAOStarAndEvaluate(
      root, std::numeric_limits<int64_t>::max(), max_time_ms, solver,
      eval_depth, max_eval_steps, n_eval_trials, nb_particles_b0,
      eval_interval_ms, completion_threshold, completion_reps, rng, solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      pomdp);
}

void runPOMCPIncrements(Wumpus* pomdp, std::mt19937_64& rng,
                        int64_t init_belief_size, double pomcp_c,
                        int64_t pomcp_nb_rollout, double pomcp_epsilon,
                        int64_t pomcp_depth, int64_t max_computation_time_ms,
                        int64_t max_eval_steps, int64_t n_eval_trials,
                        int64_t nb_particles_b0, int64_t eval_interval_ms,
                        int64_t completion_threshold, int64_t completion_reps) {
  // Initialise heuristic
  OptimalPath solver(pomdp);

  // Create root node
  POMCP::TreeNodePtr root_node = std::make_shared<POMCP::TreeNode>(0);

  // Generate initial belief particles
  std::vector<State> init_belief_p;
  for (int64_t n = 0; n < init_belief_size; ++n)
    init_belief_p.push_back(pomdp->SampleStartState());
  POMCP::BeliefParticles init_belief(init_belief_p);

  // Run POMCP
  std::cout << "Running POMCP on belief tree" << std::endl;
  POMCP::RunPOMCPAndEvaluate(
      init_belief, pomcp_c, pomcp_nb_rollout, pomcp_epsilon, pomcp_depth,
      max_computation_time_ms, max_eval_steps, n_eval_trials, nb_particles_b0,
      eval_interval_ms, completion_threshold, completion_reps, rng, solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      pomdp);
}

int main(int argc, char* argv[]) {
  const EvalParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  // Initialise the POMDP
  std::cout << "Initialising Wumpus" << std::endl;
  int64_t grid_size = 4;
  ReadWumpusParams(params.datafile, grid_size);
  const size_t heuristic_samples = params.max_belief_samples / 10;
  auto pomdp = Wumpus(grid_size, heuristic_samples, rng);

  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(params.nb_particles_b0, &pomdp);
  if (params.max_belief_samples < (int64_t)init_belief.size()) {
    std::cout << "Initial belief size: " << init_belief.size() << std::endl;
    std::cout << "Downsampling belief" << std::endl;
    init_belief = DownsampleBelief(init_belief, params.max_belief_samples, rng);
  }
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;

  // Run MCVI
  auto mcvi_wumpus = new Wumpus(pomdp);
  runMCVIIncrements(
      mcvi_wumpus, init_belief, rng, params.max_sim_depth, params.max_node_size,
      params.max_sim_depth, params.eval_epsilon, params.converge_thresh,
      params.max_time_ms, params.max_sim_depth, params.n_eval_trials,
      10 * params.nb_particles_b0, params.eval_interval_ms,
      params.completion_threshold, params.completion_reps);
  delete mcvi_wumpus;

  // Compare to AO*
  auto aostar_wumpus = new Wumpus(pomdp);
  runAOStarIncrements(aostar_wumpus, init_belief, rng, params.max_sim_depth,
                      params.max_time_ms, params.max_sim_depth,
                      params.n_eval_trials, 10 * params.nb_particles_b0,
                      params.eval_interval_ms, params.completion_threshold,
                      params.completion_reps);
  delete aostar_wumpus;

  // Compare to POMCP offline
  auto pomcp_wumpus = new Wumpus(pomdp);
  const double pomcp_c = 10.0;
  const int64_t pomcp_nb_rollout = 1000;
  const double pomcp_epsilon = 0.01;
  runPOMCPIncrements(pomcp_wumpus, rng, params.max_belief_samples, pomcp_c,
                     pomcp_nb_rollout, pomcp_epsilon, params.max_sim_depth,
                     params.max_time_ms, params.max_sim_depth,
                     params.n_eval_trials, 10 * params.nb_particles_b0,
                     params.eval_interval_ms, params.completion_threshold,
                     params.completion_reps);
  delete pomcp_wumpus;

  return 0;
}
