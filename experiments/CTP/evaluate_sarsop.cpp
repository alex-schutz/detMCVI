#include <iostream>
#include <random>

#include "AlphaVectorFSC.h"
#include "CTP.h"
#include "MCVI.h"
#include "Params.h"

#define RANDOM_SEED (42)

using namespace MCVI;

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
