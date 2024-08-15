#include <iostream>
#include <random>

#include "AlphaVectorFSC.h"
#include "MCVI.h"
#include "Params.h"
#include "maze.h"

#define RANDOM_SEED (42)

using namespace MCVI;

int main(int argc, char* argv[]) {
  const EvalParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  // Initialise the POMDP
  std::cout << "Initialising Maze" << std::endl;
  std::vector<std::string> maze;
  ReadMazeParams(params.datafile, maze);
  auto pomdp = Maze(maze, rng);

  const auto fsc = ParseDotFile("policy.dot");

  OptimalPath solver(&pomdp);

  std::cout << "Evaluation of SARSOP policy (" << params.max_sim_depth
            << " steps, " << params.n_eval_trials << " trials) at time " << 0
            << ":" << std::endl;
  EvaluationWithSimulationFSC(
      params.max_sim_depth, params.n_eval_trials, params.nb_particles_b0,
      [&pomdp](const State& state, int64_t value) {
        return pomdp.get_state_value(state, value);
      },
      &pomdp, rng, fsc, solver, "SARSOP");
  std::cout << "SARSOP policy FSC contains " << fsc.NumNodes() << " nodes."
            << std::endl;

  return 0;
}
