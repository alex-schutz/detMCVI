#include <iostream>
#include <random>

#include "AlphaVectorFSC.h"
#include "MCVI.h"
#include "Params.h"
#include "mastermind.h"

#define RANDOM_SEED (42)

using namespace MCVI;

int main(int argc, char* argv[]) {
  const EvalParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  int colour_count = 6;
  int peg_count = 4;
  ReadMastermindParams(params.datafile, colour_count, peg_count);

  // Initialise the POMDP
  std::cout << "Initialising Mastermind" << std::endl;
  auto pomdp = Mastermind(colour_count, peg_count, rng);

  const auto fsc = ParseDotFile("policy.dot");

  OptimalPath solver(&pomdp);

  std::cout << "Evaluation of SARSOP policy (" << params.max_sim_depth
            << " steps, " << params.n_eval_trials << " trials) at time " << 0
            << ":" << std::endl;
  EvaluationWithSimulationFSC(params.max_sim_depth, params.n_eval_trials,
                              params.nb_particles_b0, std::nullopt, &pomdp, rng,
                              fsc, solver, "SARSOP");
  std::cout << "SARSOP policy FSC contains " << fsc.NumNodes() << " nodes."
            << std::endl;

  return 0;
}
